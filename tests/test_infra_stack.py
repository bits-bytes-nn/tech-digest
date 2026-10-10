"""The synthesized CloudFormation for the Batch stack, asserted.

``cdk synth`` in CI only proves the stack compiles. Five regressions at once —
the S3 grant widened to the whole bucket, the application-inference-profile ARN
dropped (every Bedrock call AccessDenied), the EXIT-on-1 retry rule removed,
Spot demoted, the FAILED rule pointed elsewhere — synthesized cleanly and passed
the suite. These tests read the template the deploy would actually ship.

Builds the stack from the committed ci-batch config, offline (dummy account and
AZs). Needs Node for jsii, so it is skipped where Node is absent.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="aws-cdk-lib needs Node (jsii)"
)

CONFIG = (
    Path(__file__).resolve().parent.parent / "app" / "configs" / "config-ci-batch.yaml"
)


@pytest.fixture(scope="module")
def template() -> dict:
    os.environ["CDK_SYNTH_DUMMY_AZS"] = "1"
    os.environ.setdefault("JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION", "1")
    import aws_cdk as core
    from aws_cdk.assertions import Template

    from app.configs.config import Config
    from scripts.deploy_infra import NewsletterStack

    config = Config.from_yaml(str(CONFIG))
    app = core.App()
    stack = NewsletterStack(
        app,
        "TestStack",
        project_name=config.resources.project_name,
        stage=config.resources.stage,
        s3_bucket_name=config.resources.s3_bucket_name,
        s3_prefix=config.resources.s3_prefix,
        lambda_or_batch="batch",
        cron_expression=config.resources.cron_expression,
        email_addresses=["sender@example.com"],
        parameters={"end_date": "null", "language": "ko", "recipients": "null"},
        env=core.Environment(account="123456789012", region="ap-northeast-2"),
    )
    return Template.from_stack(stack).to_json()


def _resources(template: dict, kind: str) -> list[dict]:
    return [r for r in template["Resources"].values() if r["Type"] == kind]


def _statements(template: dict) -> list[dict]:
    return [
        statement
        for policy in _resources(template, "AWS::IAM::Policy")
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
    ]


def _actions(statement: dict) -> list[str]:
    action = statement["Action"]
    return action if isinstance(action, list) else [action]


def test_s3_object_access_is_limited_to_the_prefix(template):
    config_bucket = "tech-digest-ci-placeholder-bucket"
    objects = [s for s in _statements(template) if "s3:GetObject" in _actions(s)]
    assert [s["Resource"] for s in objects] == [
        f"arn:aws:s3:::{config_bucket}/tech-digest/*"
    ]


def test_bedrock_invoke_covers_every_resource_type_the_runtime_resolves(template):
    [invoke] = [
        s for s in _statements(template) if "bedrock:InvokeModel" in _actions(s)
    ]
    resources = json.dumps(invoke["Resource"])
    for fragment in (
        "foundation-model/anthropic.*",
        "foundation-model/openai.*",
        ":inference-profile/*",
        ":application-inference-profile/*",
    ):
        assert fragment in resources, fragment


def test_a_failed_run_is_not_retried_but_an_infrastructure_death_is(template):
    [job_definition] = _resources(template, "AWS::Batch::JobDefinition")
    assert job_definition["Properties"]["RetryStrategy"] == {
        "Attempts": 2,
        "EvaluateOnExit": [{"Action": "EXIT", "OnExitCode": "1"}],
    }


def test_a_job_stuck_in_runnable_is_cancelled_so_the_failed_rule_fires(template):
    [queue] = _resources(template, "AWS::Batch::JobQueue")
    actions = queue["Properties"]["JobStateTimeLimitActions"]
    assert {a["Reason"] for a in actions} == {
        "CAPACITY:INSUFFICIENT_INSTANCE_CAPACITY",
        "MISCONFIGURATION:COMPUTE_ENVIRONMENT_MAX_RESOURCE",
        "MISCONFIGURATION:JOB_RESOURCE_REQUIREMENT",
    }
    assert {(a["Action"], a["State"]) for a in actions} == {("CANCEL", "RUNNABLE")}


def test_spot_is_tried_first(template):
    [queue] = _resources(template, "AWS::Batch::JobQueue")
    order = {
        json.dumps(entry["ComputeEnvironment"]): entry["Order"]
        for entry in queue["Properties"]["ComputeEnvironmentOrder"]
    }
    spot = [k for k in order if "Spot" in k]
    assert spot and order[spot[0]] == 1


def test_failed_jobs_reach_the_topic(template):
    rules = [
        r
        for r in _resources(template, "AWS::Events::Rule")
        if r["Properties"].get("EventPattern", {}).get("detail", {}).get("status")
        == ["FAILED"]
    ]
    assert len(rules) == 1
    [target] = rules[0]["Properties"]["Targets"]
    assert "NewsletterTopic" in json.dumps(target["Arn"])


def test_the_job_role_and_the_instance_role_are_separate(template):
    roles = _resources(template, "AWS::IAM::Role")
    principals = {
        json.dumps(
            r["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]
        )
        for r in roles
    }
    assert '{"Service": "ecs-tasks.amazonaws.com"}' in principals
    assert '{"Service": "ec2.amazonaws.com"}' in principals
    assert not any("ec2.amazonaws.com" in p and "ecs-tasks" in p for p in principals)
