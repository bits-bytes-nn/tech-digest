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
def template(tmp_path_factory) -> dict:
    # Restored afterwards so the rest of the session does not inherit it.
    previous = os.environ.get("CDK_SYNTH_DUMMY_AZS")
    os.environ["CDK_SYNTH_DUMMY_AZS"] = "1"
    os.environ.setdefault("JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION", "1")
    try:
        yield _synthesize(tmp_path_factory.mktemp("cdk"))
    finally:
        if previous is None:
            os.environ.pop("CDK_SYNTH_DUMMY_AZS", None)
        else:
            os.environ["CDK_SYNTH_DUMMY_AZS"] = previous


def test_image_assets_never_ship_recipient_lists_or_secrets():
    from scripts.deploy_infra import IMAGE_ASSET_EXCLUDES

    for pattern in ("assets/recipients-*.txt", "**/.env", "**/.env.*"):
        assert pattern in IMAGE_ASSET_EXCLUDES, pattern


def test_the_staged_image_context_holds_no_recipient_list(template, tmp_path_factory):
    # The synthesized asset itself: recipients-*.txt exist only on a machine that
    # has them (they are gitignored), so this bites locally, where they live.
    staged = [
        path
        for root in Path(tmp_path_factory.getbasetemp()).glob("cdk*/asset.*")
        for path in root.rglob("*")
    ]
    assert staged, "no staged asset found"
    names = {path.name for path in staged}
    assert not [n for n in names if n.startswith("recipients-")]
    assert ".env" not in names and ".mypy_cache" not in names
    assert not [p for p in staged if p.name == "__pycache__"]


def _synthesize(outdir: Path) -> dict:
    import aws_cdk as core
    from aws_cdk.assertions import Template

    from app.configs.config import Config
    from scripts.deploy_infra import NewsletterStack

    config = Config.from_yaml(str(CONFIG))
    # An explicit outdir under pytest's tmp: the default is a fresh system temp
    # directory per run, each holding a full copy of app/ that nothing removed.
    app = core.App(outdir=str(outdir))
    stack = NewsletterStack(
        app,
        "TestStack",
        project_name=config.resources.project_name,
        stage=config.resources.stage,
        s3_bucket_name=config.resources.s3_bucket_name,
        s3_prefix=config.resources.s3_prefix,
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


def test_the_role_can_read_inference_profiles(template):
    # langchain-aws calls GetInferenceProfile when it is given an application
    # profile ARN; removing it made every run fail before filtering.
    actions = {a for s in _statements(template) for a in _actions(s)}
    assert {"bedrock:ListInferenceProfiles", "bedrock:GetInferenceProfile"} <= actions


# Actions granted on Resource "*". Each has no resource-level scoping in IAM
# (SendRawEmail is held to the sender by a ses:FromAddress condition instead), so
# a new entry here is a deliberate widening, not an accident.
ACCOUNT_WIDE_ACTIONS = {
    "bedrock:GetInferenceProfile",
    "bedrock:ListInferenceProfiles",
    "ecr:GetAuthorizationToken",
    "ses:ListSuppressedDestinations",
    "ses:SendRawEmail",
}


def test_no_wildcard_actions_and_only_profile_listing_is_account_wide(template):
    statements = _statements(template)
    assert not [a for s in statements for a in _actions(s) if "*" in a]
    account_wide = {a for s in statements if s["Resource"] == "*" for a in _actions(s)}
    assert account_wide == ACCOUNT_WIDE_ACTIONS


def test_a_failed_run_is_not_retried_but_an_infrastructure_death_is(template):
    [job_definition] = _resources(template, "AWS::Batch::JobDefinition")
    from app.src.constants import BATCH_ATTEMPTS

    assert job_definition["Properties"]["RetryStrategy"] == {
        "Attempts": BATCH_ATTEMPTS,
        "EvaluateOnExit": [{"Action": "EXIT", "OnExitCode": "1"}],
    }


def test_the_job_runs_the_app_package(template):
    [job_definition] = _resources(template, "AWS::Batch::JobDefinition")
    command = job_definition["Properties"]["ContainerProperties"]["Command"]
    assert command[:3] == ["python", "-m", "app.main"]


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


def test_s3_grants_no_destructive_action(template):
    s3_actions = {
        a for s in _statements(template) for a in _actions(s) if a.startswith("s3:")
    }
    assert s3_actions == {
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:GetBucketLocation",
    }


def test_kms_decrypt_only_through_ssm(template):
    [kms] = [s for s in _statements(template) if "kms:Decrypt" in _actions(s)]
    assert kms["Condition"] == {
        "StringEquals": {"kms:ViaService": "ssm.ap-northeast-2.amazonaws.com"}
    }


def test_ses_send_only_as_the_configured_sender(template):
    [ses] = [s for s in _statements(template) if "ses:SendRawEmail" in _actions(s)]
    assert ses["Condition"] == {
        "StringEquals": {"ses:FromAddress": "sender@example.com"}
    }


def test_the_suppression_list_is_readable(template):
    assert any(
        "ses:ListSuppressedDestinations" in _actions(s) for s in _statements(template)
    )


def test_ssm_reads_only_this_projects_parameters(template):
    [ssm] = [s for s in _statements(template) if "ssm:GetParameter" in _actions(s)]
    assert json.dumps(ssm["Resource"]).endswith(':parameter/tech-digest/dev/*"')


def test_the_job_role_carries_no_managed_policy(template):
    roles = _resources(template, "AWS::IAM::Role")
    [job_role] = [
        r
        for r in roles
        if r["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]
        == {"Service": "ecs-tasks.amazonaws.com"}
        and r["Properties"].get("RoleName") == "tech-digest-dev-newsletter"
    ]
    assert not job_role["Properties"].get("ManagedPolicyArns")


def test_instances_require_imdsv2_with_one_hop(template):
    [launch_template] = _resources(template, "AWS::EC2::LaunchTemplate")
    options = launch_template["Properties"]["LaunchTemplateData"]["MetadataOptions"]
    assert options["HttpTokens"] == "required"
    assert options["HttpPutResponseHopLimit"] == 1


def test_a_dropped_schedule_lands_in_the_dlq(template):
    [schedule] = [
        r
        for r in _resources(template, "AWS::Events::Rule")
        if "ScheduleExpression" in r["Properties"]
    ]
    [target] = schedule["Properties"]["Targets"]
    assert "DeadLetterConfig" in target
    assert target["RetryPolicy"]["MaximumRetryAttempts"] == 2


def test_job_logs_expire_and_jobs_time_out(template):
    groups = _resources(template, "AWS::Logs::LogGroup")
    assert groups and all(g["Properties"].get("RetentionInDays") for g in groups)
    [job_definition] = _resources(template, "AWS::Batch::JobDefinition")
    from app.src.constants import BATCH_ATTEMPT_TIMEOUT_MINUTES

    assert job_definition["Properties"]["Timeout"] == {
        "AttemptDurationSeconds": BATCH_ATTEMPT_TIMEOUT_MINUTES * 60
    }


def test_compute_environments_use_a_supported_ami(template):
    # Batch's ECS Amazon Linux 2 AMI reached end of support on 2026-06-30.
    for env in _resources(template, "AWS::Batch::ComputeEnvironment"):
        images = env["Properties"]["ComputeResources"]["Ec2Configuration"]
        assert [i["ImageType"] for i in images] == ["ECS_AL2023"]


def test_the_cdk_app_makes_no_aws_call_and_ignores_local_tracing(monkeypatch, tmp_path):
    # `cdk synth` runs this app exactly as `cdk deploy` does, with the operator's
    # .env loaded. Neither the LangChain key nor the local tracing switch may
    # reach AWS or the deployed stack, and a synth must make no AWS call at all.
    import botocore.client

    from scripts import deploy_infra

    calls: list[str] = []

    def _no_aws(self, operation_name, api_params):
        calls.append(operation_name)
        raise AssertionError(f"AWS call during synth: {operation_name}")

    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", _no_aws)
    monkeypatch.setenv("CONFIG_FILE_SUFFIX", "ci-batch")
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
    monkeypatch.setenv("CDK_SYNTH_DUMMY_AZS", "1")
    # Not CDK_OUTDIR: the jsii process already started with the session's env.
    app_class = deploy_infra.core.App
    monkeypatch.setattr(
        deploy_infra.core, "App", lambda: app_class(outdir=str(tmp_path))
    )
    monkeypatch.setenv("LANGCHAIN_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://tracing.example.com")

    deploy_infra.main()

    assert calls == []
    body = (tmp_path / "NewsletterDevStack.template.json").read_text(encoding="utf-8")
    assert "not-a-real-key" not in body
    [job_definition] = _resources(json.loads(body), "AWS::Batch::JobDefinition")
    environment = {
        e["Name"]: e["Value"]
        for e in job_definition["Properties"]["ContainerProperties"]["Environment"]
    }
    assert environment["LANGCHAIN_TRACING_V2"] == "false"
    assert "LANGCHAIN_ENDPOINT" not in environment


def test_the_cdk_app_needs_the_cli_account(monkeypatch):
    from scripts import deploy_infra

    monkeypatch.setenv("CONFIG_FILE_SUFFIX", "ci-batch")
    monkeypatch.delenv("CDK_DEFAULT_ACCOUNT", raising=False)
    with pytest.raises(SystemExit):
        deploy_infra.main()
