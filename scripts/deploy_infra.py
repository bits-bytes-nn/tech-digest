import os
import sys
from pathlib import Path
from typing import Any

import aws_cdk as core
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_batch as batch,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_cloudwatch_actions as cloudwatch_actions,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as subscriptions,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from aws_cdk.aws_events_targets import BatchJob, SnsTopic
from constructs import Construct

sys.path.append(str(Path(__file__).parent.parent))
from app.configs import Config
from app.domain.constants import (
    BATCH_ATTEMPT_TIMEOUT_MINUTES,
    BATCH_ATTEMPTS,
    BATCH_RUNNABLE_LIMIT_MINUTES,
    AppConstants,
    EnvVars,
    SSMParams,
)
from app.logger import logger

# Paths kept out of the image build context, shared by both image assets so they
# cannot drift apart. Patterns are relative to the context directory (``app/``).
#
# The root `.dockerignore` does NOT apply to these builds — the context is `app/`,
# so Docker would look for `app/.dockerignore`, and CDK passes this list instead.
# A local run downloads the recipient list (real email addresses) to
# `<repo>/assets/`, outside this context; the recipients pattern guards against a
# copy placed in `app/assets/` (beside the logos, which do ship), which
# `COPY . app/` in Dockerfile-batch would otherwise bake into the image.
IMAGE_ASSET_EXCLUDES: list[str] = [
    "cdk.out",
    ".git",
    # Local recipient lists hold real addresses; the runtime always reads the
    # list from S3, so no copy may ship in an image.
    "assets/recipients-*.txt",
    # Local tool caches and secrets. A cache changes on every run, so shipping
    # it would also change the asset hash and turn a no-op deploy into a new
    # image and job-definition revision. The ** forms matter: a bare pattern
    # only matches at the root of app/.
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "**/__pycache__",
    "**/*.pyc",
    "**/.DS_Store",
    "**/.env",
    "**/.env.*",
]


class NewsletterStack(Stack):
    # Runtime logs are for debugging the last few runs, not an archive. Left
    # unset (the default) a log group never expires, so a weekly job pays storage
    # on logs nobody will read again — and the growth is invisible.
    LOG_RETENTION = logs.RetentionDays.THREE_MONTHS

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_name: str,
        stage: str = "dev",
        s3_bucket_name: str = "",
        s3_prefix: str | None = None,
        vpc_id: str | None = None,
        subnet_ids: list[str] | None = None,
        cron_expression: str | None = None,
        email_addresses: list[str] | None = None,
        environment_vars: dict[str, str] | None = None,
        parameters: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not s3_bucket_name:
            # Fail closed: an empty bucket name would otherwise scope the S3
            # IAM policy to arn:aws:s3:::* (every bucket in the account).
            raise ValueError(
                "s3_bucket_name is required to scope the S3 IAM policy; "
                "set resources.s3_bucket_name in the config."
            )
        self.project_name = project_name
        self.stage = stage
        self._bucket_name = s3_bucket_name
        self._s3_prefix = (s3_prefix or "").strip("/")
        if not self._s3_prefix:
            # Fail closed, as for the bucket name: without a prefix the object
            # grant below would cover the whole shared SageMaker bucket.
            raise ValueError(
                "s3_prefix is required to scope the S3 IAM policy; "
                "set resources.s3_prefix in the config."
            )
        # The SES sender identity to scope send permissions to (operator email).
        self._sender = email_addresses[0] if email_addresses else None
        self._add_tags(scope)

        self._configure_vpc(vpc_id, subnet_ids)
        self.security_group = self._create_security_group()
        self.role = self._create_iam_role()
        self.topic = self._create_sns_topic(email_addresses)
        # Dead-letter queue for scheduled-invocation delivery failures. Without
        # it, if EventBridge cannot deliver the weekly trigger after its retries
        # the event is dropped silently and the digest is missed with no signal.
        self.schedule_dlq = self._create_schedule_dlq()

        common_env_vars = self._get_common_environment_vars(
            self.topic.topic_arn,
            **(environment_vars or {}),
        )

        job_queue_name, job_definition_name = self._setup_batch_resources(
            common_env_vars, cron_expression, parameters
        )
        self._store_ssm_parameters(job_queue_name, job_definition_name)

    @property
    def availability_zones(self) -> list[str]:
        # When synthesizing without AWS credentials (CI validation), return
        # deterministic AZs so VPC creation doesn't trigger an account lookup.
        # Real deploys have credentials and use the default lookup.
        if os.environ.get("CDK_SYNTH_DUMMY_AZS"):
            return [f"{self.region}a", f"{self.region}c"]
        return super().availability_zones

    def _get_resource_name(self, suffix: str) -> str:
        return f"{self.project_name}-{self.stage}-{suffix}"

    def _add_tags(self, scope: Construct) -> None:
        # `Project` matches the key the sibling projects (omnisummary, paper-bridge,
        # scholar-lens) tag with, and the key put_inference_profiles.py puts on the
        # Bedrock application inference profiles. A cost-allocation tag is activated
        # once per payer account BY KEY, so sharing one key means one activation
        # covers every workload instead of one per project. `ProjectName` is kept
        # alongside it so any existing Cost Explorer filter or report keeps working.
        for key, value in {
            "ProjectName": self.project_name,
            "Stage": self.stage,
            "CostCenter": self.project_name,
            "ManagedBy": "CDK",
        }.items():
            Tags.of(scope).add(key, value)
        # `Project` is added separately, EXCLUDING Batch compute environments.
        # `AWS::Batch::ComputeEnvironment` treats Tags as create-only, so adding a
        # key to an existing one requires replacement — and because the compute
        # environments are custom-named, CloudFormation cannot replace them
        # ("cannot update a stack when a custom-named resource requires
        # replacing"), which fails the whole deploy. Excluding them costs nothing
        # worth having: the EC2 minutes are cents, while the spend this tag exists
        # to attribute is Bedrock, and that is tagged on the application inference
        # profiles by scripts/put_inference_profiles.py rather than by CDK.
        Tags.of(scope).add(
            "Project",
            self.project_name,
            exclude_resource_types=["AWS::Batch::ComputeEnvironment"],
        )

    def _configure_vpc(self, vpc_id: str | None, subnet_ids: list[str] | None) -> None:
        if vpc_id and subnet_ids:
            self.vpc = ec2.Vpc.from_lookup(self, "BaseVPC", vpc_id=vpc_id)
            self.vpc_subnets = ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(self, f"BaseSubnet-{i}", subnet_id)
                    for i, subnet_id in enumerate(subnet_ids)
                ]
            )
        else:
            self.vpc = ec2.Vpc(
                self,
                "BaseVPC",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public",
                        subnet_type=ec2.SubnetType.PUBLIC,
                        cidr_mask=24,
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                        cidr_mask=24,
                    ),
                ],
            )
            self.vpc_subnets = ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            )

    def _create_security_group(self) -> ec2.SecurityGroup:
        sg = ec2.SecurityGroup(
            self,
            "NewsletterSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=True,
            description="Security group for Newsletter",
            security_group_name=self._get_resource_name("newsletter"),
        )
        return sg

    def _create_iam_role(self) -> iam.Role:
        # Least-privilege (AWS Well-Architected, Security pillar): instead of
        # broad *FullAccess managed policies, grant only the specific actions
        # the application performs, scoped to this stack's resources.
        #
        # This is the JOB role: only the container assumes it. The EC2 hosts get
        # their own instance role (``_create_instance_role``), so neither carries
        # the other's rights.
        role = iam.Role(
            self,
            "NewsletterRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Least-privilege IAM role for Tech Digest Newsletter",
            role_name=self._get_resource_name("newsletter"),
        )
        for statement in self._app_policy_statements():
            role.add_to_policy(statement)
        return role

    def _create_instance_role(self) -> iam.Role:
        """The Batch EC2 hosts' role: ECS registration, image pull and the
        awslogs driver, which on EC2 writes with the host's credentials. Nothing
        the application does runs under it."""
        return iam.Role(
            self,
            "NewsletterInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Batch EC2 instance role for Tech Digest Newsletter",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                )
            ],
            role_name=self._get_resource_name("newsletter-instance"),
        )

    def _app_policy_statements(self) -> list[iam.PolicyStatement]:
        """Scoped permissions for the application's actual AWS operations."""
        region = self.region
        account = self.account
        bucket_arn = f"arn:aws:s3:::{self._bucket_name}"
        # Every key the app reads or writes sits under resources.s3_prefix
        # (recipients/, newsletters/, articles/, deliveries/). The configured
        # bucket is the account's shared SageMaker default bucket, so granting
        # the whole of it let this role read and overwrite other workloads' data.
        object_arn = f"{bucket_arn}/{self._s3_prefix}/*"
        param_arn = (
            f"arn:aws:ssm:{region}:{account}:parameter/"
            f"{self.project_name}/{self.stage}/*"
        )
        return [
            # S3: read config/recipients, write newsletters/articles — object
            # actions on the bucket, plus ListBucket on the bucket itself.
            # No s3:DeleteObject: the app never deletes an object (aws_helpers
            # exposes upload and download only), and granting the one destructive
            # action of the three contradicted the least-privilege claim above.
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[object_arn],
            ),
            # ListBucket stays unconditioned: the app never lists, but S3 answers
            # a GetObject on a missing key with 404 only when the caller holds
            # ListBucket on the bucket, and that check carries no s3:prefix to
            # match a condition against. Without it, the absent delivery ledger
            # of every first send reads as AccessDenied. It exposes key names,
            # not contents.
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[bucket_arn],
            ),
            # SES: read the account suppression list so bounced addresses are
            # skipped. A read-only action without resource-level permissions.
            iam.PolicyStatement(
                actions=["ses:ListSuppressedDestinations"],
                resources=["*"],
            ),
            # SES: deliver the newsletter, restricted (via FromAddress) to the
            # configured sender so the role cannot send as arbitrary identities.
            # SendRawEmail only — that is the single API aws_helpers.send_email
            # calls (it builds a MIME message to set Message-ID and
            # List-Unsubscribe, which the simple SendEmail action cannot carry).
            iam.PolicyStatement(
                actions=["ses:SendRawEmail"],
                resources=["*"],  # SES identity ARNs are account/region-specific
                conditions=(
                    {"StringEquals": {"ses:FromAddress": self._sender}}
                    if self._sender
                    else None
                ),
            ),
            # SNS: publish run/health notifications to this stack's topic.
            iam.PolicyStatement(
                actions=["sns:Publish"],
                resources=[
                    f"arn:aws:sns:{region}:{account}:"
                    f"{self._get_resource_name('newsletter')}"
                ],
            ),
            # SSM: read this project's parameters (e.g. the LangChain API key,
            # a SecureString the operator puts outside CDK).
            iam.PolicyStatement(
                actions=["ssm:GetParameter"],
                resources=[param_arn],
            ),
            # KMS: decrypt the SecureString SSM parameter. Scoped to the
            # account's default SSM key via the ViaService condition, so the role
            # can only use it through SSM, not for arbitrary decryption.
            iam.PolicyStatement(
                actions=["kms:Decrypt"],
                resources=[f"arn:aws:kms:{region}:{account}:key/*"],
                conditions={
                    "StringEquals": {"kms:ViaService": f"ssm.{region}.amazonaws.com"}
                },
            ),
            # Bedrock: invoke Claude and OpenAI GPT models, including via cross-region
            # inference profiles. A cross-region profile (us./apac./global.)
            # fans the underlying InvokeModel out to MULTIPLE regions, so the
            # foundation-model resource MUST span regions or invocation hits
            # AccessDenied when the profile routes to a region not listed.
            # foundation-model ARNs are AWS-owned (no account id); the
            # meaningful restriction is the provider model namespace
            # (anthropic.*, openai.*), with a region wildcard. The account-scoped inference-profile ARNs remain
            # the access-controlled resource.
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],  # calls are not streamed
                resources=[
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    "arn:aws:bedrock:*::foundation-model/openai.*",
                    f"arn:aws:bedrock:*:{account}:inference-profile/*",
                    # `application-inference-profile` is a DIFFERENT IAM resource
                    # type from `inference-profile`. The runtime prefers this
                    # project's application profile when one exists (that is how
                    # Bedrock spend gets a cost-allocation tag), so omitting this
                    # makes every Bedrock call AccessDenied the moment a profile is
                    # provisioned — while working fine until then.
                    f"arn:aws:bedrock:*:{account}:application-inference-profile/*",
                ],
            ),
            iam.PolicyStatement(
                # List: our profile resolution (model_factory). Get: langchain-aws
                # calls it inside ChatBedrockConverse for an application-profile
                # ARN to learn the base model. Without it every run fails at
                # model construction with AccessDenied.
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                ],
                resources=["*"],  # profile discovery is an account-level action
            ),
            # No CloudWatch Logs statement: nothing running as this role writes
            # logs itself. Batch container output goes through the awslogs driver,
            # which on EC2 uses the INSTANCE role (its managed policy grants
            # CreateLogStream/PutLogEvents).
        ]

    def _create_sns_topic(self, email_addresses: list[str] | None) -> sns.Topic:
        topic = sns.Topic(
            self,
            "NewsletterTopic",
            topic_name=self._get_resource_name("newsletter"),
            display_name="Tech Digest Notifications",
        )
        if email_addresses:
            for email in email_addresses:
                topic.add_subscription(subscriptions.EmailSubscription(email))
        return topic

    def _create_schedule_dlq(self) -> sqs.Queue:
        """SQS dead-letter queue for the EventBridge schedule target, plus an
        alarm on it so a dropped weekly trigger surfaces to the SNS topic
        instead of vanishing silently."""
        dlq = sqs.Queue(
            self,
            "ScheduleDlq",
            queue_name=self._get_resource_name("schedule-dlq"),
            retention_period=Duration.days(14),
            enforce_ssl=True,
        )
        alarm = cloudwatch.Alarm(
            self,
            "ScheduleDlqAlarm",
            alarm_name=self._get_resource_name("schedule-dlq"),
            alarm_description=(
                "A scheduled newsletter invocation was dropped and sent to the "
                "dead-letter queue — the run likely did not start."
            ),
            metric=dlq.metric_approximate_number_of_messages_visible(
                period=Duration.minutes(5), statistic="Maximum"
            ),
            threshold=0,
            evaluation_periods=1,
            comparison_operator=(cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD),
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        alarm.add_alarm_action(cloudwatch_actions.SnsAction(self.topic))
        return dlq

    def _get_common_environment_vars(
        self, topic_arn: str, **kwargs: str
    ) -> dict[str, str]:
        # Regions are not passed: the container loads the same stage config
        # (CONFIG_FILE_SUFFIX) and reads them from there.
        env_vars = {
            EnvVars.CONFIG_FILE_SUFFIX.value: self.stage,
            EnvVars.LOG_LEVEL.value: "INFO",
            # Name this project's Bedrock application inference profiles, which is
            # what carries the cost-allocation tags.
            EnvVars.PROJECT_NAME.value: self.project_name,
            EnvVars.STAGE.value: self.stage,
            EnvVars.TOPIC_ARN.value: topic_arn,
        }
        env_vars.update(kwargs)
        return env_vars

    def _setup_batch_resources(
        self,
        environment: dict[str, str],
        cron_expression: str | None,
        parameters: dict[str, str] | None,
    ) -> tuple[str, str]:
        job_queue = self._create_job_queue()
        job_definition = self._create_job_definition(environment)
        if cron_expression:
            batch_target = BatchJob(
                job_queue_arn=job_queue.job_queue_arn,
                job_queue_scope=job_queue,
                job_definition_arn=job_definition.job_definition_arn,
                job_definition_scope=job_definition,
                job_name=self._get_resource_name("newsletter"),
                retry_attempts=2,
                dead_letter_queue=self.schedule_dlq,
                event=(
                    events.RuleTargetInput.from_object({"Parameters": parameters})
                    if parameters
                    else None
                ),
            )
            self._create_event_rule(cron_expression, target=batch_target)
        # AWS Batch emits no per-job "failed" CloudWatch metric, so watch the
        # terminal FAILED state via EventBridge and forward it to SNS. This
        # catches failures (OOM, init crash) that never reach main.py's in-app
        # alert branches.
        self._create_batch_failure_alarm(job_queue)
        return job_queue.job_queue_name, job_definition.job_definition_name

    def _create_batch_failure_alarm(self, job_queue: batch.JobQueue) -> None:
        rule = events.Rule(
            self,
            "BatchJobFailedRule",
            rule_name=self._get_resource_name("batch-job-failed"),
            description="Forward FAILED Batch jobs for this queue to SNS.",
            event_pattern=events.EventPattern(
                source=["aws.batch"],
                detail_type=["Batch Job State Change"],
                detail={
                    "status": ["FAILED"],
                    "jobQueue": [job_queue.job_queue_arn],
                },
            ),
        )
        rule.add_target(SnsTopic(self.topic))

    def _create_job_queue(self) -> batch.JobQueue:
        # A job that can never be placed sits in RUNNABLE indefinitely: the job
        # definition's attempt timeout only counts RUNNING time, and the FAILED rule
        # below never fires, so a missed issue raised no signal at all. Batch can
        # cancel such a job for the three reasons it can diagnose; a cancelled job
        # reaches FAILED, which the existing rule forwards to SNS. Two hours is
        # far past normal placement (minutes) and still leaves the run the same day.
        job_queue = batch.JobQueue(
            self,
            "NewsletterJobQueue",
            job_queue_name=self._get_resource_name("newsletter"),
            priority=1,
            compute_environments=[],
            job_state_time_limit_actions=[
                batch.JobStateTimeLimitAction(
                    max_time=Duration.minutes(BATCH_RUNNABLE_LIMIT_MINUTES),
                    reason=reason,
                    action=batch.JobStateTimeLimitActionsAction.CANCEL,
                    state=batch.JobStateTimeLimitActionsState.RUNNABLE,
                )
                for reason in (
                    batch.JobStateTimeLimitActionsReason.INSUFFICIENT_INSTANCE_CAPACITY,
                    batch.JobStateTimeLimitActionsReason.COMPUTE_ENVIRONMENT_MAX_RESOURCE,
                    batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
                )
            ],
        )

        # Require IMDSv2 (token-authenticated) and drop the hop limit to 1 so a
        # single unauthenticated GET to 169.254.169.254 can no longer read the
        # instance role's credentials — defense-in-depth that neutralizes the
        # SSRF -> credential-theft path even if the URL filter is bypassed. The
        # container's job-role credentials come from the ECS agent endpoint
        # (169.254.170.2), not IMDS, so this does not affect the app. No AMI is
        # pinned here — Batch fills in the ECS-optimized AMI and user data.
        launch_template = ec2.LaunchTemplate(
            self,
            "NewsletterLaunchTemplate",
            launch_template_name=self._get_resource_name("newsletter"),
            require_imdsv2=True,
            http_put_response_hop_limit=1,
        )

        compute_env_config: dict[str, Any] = {
            "instance_role": self._create_instance_role(),
            # Let Batch pick from the optimal C/M/R instance classes. This is the
            # supported way to express Batch "optimal"; passing
            # InstanceType("optimal") is rejected as a malformed identifier by
            # current aws-cdk-lib.
            "use_optimal_instance_classes": True,
            # Pinned: Batch's ECS-optimized Amazon Linux 2 AMI reached end of
            # support on 2026-06-30, and an environment keeps the image type it
            # was created with unless one is named here.
            "images": [
                batch.EcsMachineImage(image_type=batch.EcsMachineImageType.ECS_AL2023)
            ],
            "launch_template": launch_template,
            "vpc": self.vpc,
            "security_groups": [self.security_group],
            "vpc_subnets": self.vpc_subnets,
        }

        # A queue tries the LOWEST order first, so Spot is order 1 and on-demand
        # the fallback at order 2. If a Spot shortage does not fall through, the
        # RUNNABLE time limit above cancels the job and the FAILED rule alerts.
        # A Spot reclaim re-runs the job from the start: the delivery ledger stops
        # a re-send, but the crawl and the Bedrock calls are paid again.
        spot_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "NewsletterSpotEnv",
            allocation_strategy=batch.AllocationStrategy.SPOT_CAPACITY_OPTIMIZED,
            compute_environment_name=self._get_resource_name("newsletter-spot"),
            spot=True,
            maxv_cpus=8,
            **compute_env_config,
        )
        job_queue.add_compute_environment(spot_env, 1)

        ondemand_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "NewsletterOnDemandEnv",
            allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
            compute_environment_name=self._get_resource_name("newsletter-ondemand"),
            maxv_cpus=4,
            **compute_env_config,
        )
        job_queue.add_compute_environment(ondemand_env, 2)

        return job_queue

    def _create_job_definition(
        self, environment: dict[str, str]
    ) -> batch.EcsJobDefinition:
        docker_image = DockerImageAsset(
            self,
            "NewsletterImage",
            directory=str(Path(__file__).parent.parent / "app"),
            file="Dockerfile-batch",
            platform=Platform.LINUX_AMD64,
            exclude=IMAGE_ASSET_EXCLUDES,
            build_args={"DOCKER_BUILDKIT": "1"},
        )

        container_definition = batch.EcsEc2ContainerDefinition(
            self,
            "NewsletterContainerDef",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image),
            # The Batch image bundles headless Chrome + xvfb for the optional
            # convert_to_images render step, which routinely needs well over
            # 512 MiB; 512 MiB would OOM-kill the job. 2 vCPU / 2 GiB is a safe
            # floor.
            cpu=2,
            memory=core.Size.mebibytes(2048),
            job_role=self.role,
            command=[
                "python",
                "-m",
                "app.main",
                "--end-date",
                "Ref::end_date",
                "--language",
                "Ref::language",
                "--recipients",
                "Ref::recipients",
            ],
            environment=environment,
            logging=ecs.LogDriver.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}",
                # Without an explicit group Batch logs into /aws/batch/job with NO
                # retention, so a weekly job accumulates logs forever and the
                # project's output is mixed in with every other Batch workload's.
                log_group=logs.LogGroup(
                    self,
                    "NewsletterJobLogGroup",
                    log_group_name=f"/aws/batch/{self._get_resource_name('newsletter')}",
                    retention=self.LOG_RETENTION,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
            ),
        )

        return batch.EcsJobDefinition(
            self,
            "NewsletterJobDefinition",
            container=container_definition,
            job_definition_name=self._get_resource_name("newsletter"),
            retry_attempts=BATCH_ATTEMPTS,
            # main.py exits 1 when the run itself failed. It has already retried
            # what can be retried (throttling, timeouts) and sent its own failure
            # alert, so a fresh attempt would repeat the same error and pay for
            # the crawl and every Bedrock call again. Any other exit — a Spot
            # reclaim, an OOM kill (137), a host failure — matches no rule below
            # and Batch retries it, which is what the attempts above are for.
            retry_strategies=[
                batch.RetryStrategy.of(
                    batch.Action.EXIT, batch.Reason.custom(on_exit_code="1")
                ),
            ],
            timeout=core.Duration.minutes(BATCH_ATTEMPT_TIMEOUT_MINUTES),
        )

    def _create_event_rule(
        self, cron_expression: str, target: events.IRuleTarget
    ) -> None:
        rule = events.Rule(
            self,
            "NewsletterEventRule",
            schedule=events.Schedule.expression(cron_expression),
            description="Event rule to trigger Newsletter batch",
            # The scheduled run's language is carried in the job parameters, not
            # the rule name — don't bake a fixed "-ko" into the resource name.
            rule_name=self._get_resource_name("batch"),
        )
        rule.add_target(target)

    def _store_ssm_parameters(
        self,
        job_queue_name: str,
        job_definition_name: str,
    ) -> None:
        # The LangChain API key is NOT created here: CloudFormation can only make
        # a plaintext `String` parameter, which would put the key in the template.
        # The operator puts it as a SecureString; the runtime reads it from SSM.
        ssm_params_to_create = {
            SSMParams.BATCH_JOB_QUEUE: job_queue_name,
            SSMParams.BATCH_JOB_DEFINITION: job_definition_name,
        }

        descriptions = {
            SSMParams.BATCH_JOB_QUEUE: "AWS Batch Job Queue Name for Tech Digest Newsletter",
            SSMParams.BATCH_JOB_DEFINITION: "AWS Batch Job Definition Name for Tech Digest Newsletter",
        }

        for param_enum, param_value in ssm_params_to_create.items():
            param_name = f"/{self.project_name}/{self.stage}/{param_enum.value}"
            ssm.StringParameter(
                self,
                f"SsmParam{param_enum.name}",
                parameter_name=param_name,
                string_value=param_value,
                description=descriptions[param_enum],
                tier=ssm.ParameterTier.STANDARD,
            )


def main() -> None:
    try:
        config = Config.load()
        logger.info(
            "Synthesizing infrastructure for '%s' in '%s' stage",
            config.resources.project_name,
            config.resources.stage,
        )

        env_vars: dict[str, str] = {
            # From the stage config only: the operator's shell and .env (which
            # Config.load reads) must not switch on third-party tracing in AWS.
            EnvVars.LANGCHAIN_TRACING_V2.value: str(
                config.resources.langchain_tracing
            ).lower(),
            EnvVars.LANGCHAIN_PROJECT.value: config.resources.project_name,
        }

        parameters = {
            "end_date": AppConstants.NULL_STRING,
            "language": "ko",
            "recipients": AppConstants.NULL_STRING,
        }

        # The app makes no AWS call: `cdk synth` runs it exactly as `cdk deploy`
        # does, and a synth must neither need credentials nor touch AWS. The CLI
        # sets CDK_DEFAULT_ACCOUNT from the credentials it deploys with.
        account_id = os.environ.get("CDK_DEFAULT_ACCOUNT")
        if not account_id:
            raise RuntimeError(
                "CDK_DEFAULT_ACCOUNT is not set: run through the cdk CLI with "
                "credentials (--profile), or set it for an offline synth."
            )
        env = core.Environment(
            account=account_id,
            region=config.resources.default_region_name,
        )
        app = core.App()

        NewsletterStack(
            app,
            f"Newsletter{config.resources.stage.capitalize()}Stack",
            project_name=config.resources.project_name,
            stage=config.resources.stage,
            s3_bucket_name=config.resources.s3_bucket_name,
            s3_prefix=config.resources.s3_prefix,
            vpc_id=config.resources.vpc_id,
            subnet_ids=config.resources.subnet_ids,
            cron_expression=config.resources.cron_expression,
            # Empty when no sender is configured, never ``["None"]`` (which would
            # subscribe "None" to the topic and scope SES to a bogus sender).
            email_addresses=(
                [str(config.newsletter.sender)] if config.newsletter.sender else []
            ),
            environment_vars=env_vars,
            parameters=parameters,
            env=env,
        )
        app.synth()
    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
