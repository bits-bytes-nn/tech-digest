import os
import sys
from pathlib import Path

import aws_cdk as core
import boto3
from aws_cdk import (
    Duration,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_batch as batch,
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
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as subscriptions,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from aws_cdk.aws_events_targets import BatchJob, LambdaFunction
from constructs import Construct

sys.path.append(str(Path(__file__).parent.parent))
from app.configs import Config
from app.src import AppConstants, EnvVars, SSMParams, get_account_id, logger


class NewsletterStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_name: str,
        stage: str = "dev",
        s3_bucket_name: str = "",
        vpc_id: str | None = None,
        subnet_ids: list[str] | None = None,
        lambda_or_batch: str = "lambda",
        cron_expression: str | None = None,
        email_addresses: list[str] | None = None,
        default_region_name: str | None = None,
        bedrock_region_name: str | None = None,
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
        self._bedrock_region = bedrock_region_name or "us-west-2"
        # The SES sender identity to scope send permissions to (operator email).
        self._sender = (email_addresses or [None])[0]
        self.lambda_or_batch = lambda_or_batch
        self._add_tags(scope)

        self._configure_vpc(vpc_id, subnet_ids)
        self.security_group = self._create_security_group()
        self.role = self._create_iam_role()
        self.topic = self._create_sns_topic(email_addresses)

        common_env_vars = self._get_common_environment_vars(
            default_region_name,
            bedrock_region_name,
            self.topic.topic_arn,
            **(environment_vars or {}),
        )

        job_queue_name = None
        job_definition_name = None

        if self.lambda_or_batch == "lambda":
            self._setup_lambda_resources(common_env_vars, cron_expression)
        else:
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
        for key, value in {
            "ProjectName": self.project_name,
            "Stage": self.stage,
            "CostCenter": self.project_name,
            "ManagedBy": "CDK",
        }.items():
            Tags.of(scope).add(key, value)

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
        # the application performs, scoped to this stack's resources. The only
        # AWS-managed policies retained are the service-execution roles that
        # are required by the runtime itself (Lambda exec / ECS-on-EC2 agent).
        if self.lambda_or_batch == "lambda":
            service_principal: iam.IPrincipal = iam.ServicePrincipal(
                "lambda.amazonaws.com"
            )
            managed_policies = [
                iam.ManagedPolicy.from_aws_managed_policy_name(name)
                for name in (
                    "service-role/AWSLambdaBasicExecutionRole",
                    "service-role/AWSLambdaVPCAccessExecutionRole",
                )
            ]
        else:
            service_principal = iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            )
            # Required for the Batch EC2 compute environment to register with
            # ECS and pull container images; this is the AWS-recommended
            # scoped instance role, not a *FullAccess policy.
            managed_policies = [
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                )
            ]

        role = iam.Role(
            self,
            "NewsletterRole",
            assumed_by=service_principal,
            description="Least-privilege IAM role for Tech Digest Newsletter",
            managed_policies=managed_policies,
            role_name=self._get_resource_name("newsletter"),
        )
        for statement in self._app_policy_statements():
            role.add_to_policy(statement)
        return role

    def _app_policy_statements(self) -> list[iam.PolicyStatement]:
        """Scoped permissions for the application's actual AWS operations."""
        region = self.region
        account = self.account
        bucket_arn = f"arn:aws:s3:::{self._bucket_name}"
        param_arn = (
            f"arn:aws:ssm:{region}:{account}:parameter/"
            f"{self.project_name}/{self.stage}/*"
        )
        return [
            # S3: read config/recipients, write newsletters/articles — object
            # actions on the bucket, plus ListBucket on the bucket itself.
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[f"{bucket_arn}/*"],
            ),
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[bucket_arn],
            ),
            # SES: deliver the newsletter, restricted (via FromAddress) to the
            # configured sender so the role cannot send as arbitrary identities.
            iam.PolicyStatement(
                actions=["ses:SendRawEmail", "ses:SendEmail"],
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
            # SSM: read this project's parameters (e.g. LangChain API key,
            # stored as a SecureString — see _put_secure_ssm_parameter).
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
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
            # Bedrock: invoke Claude models, including via cross-region
            # inference profiles. A cross-region profile (us./apac./global.)
            # fans the underlying InvokeModel out to MULTIPLE regions, so the
            # foundation-model resource MUST span regions or invocation hits
            # AccessDenied when the profile routes to a region not listed.
            # foundation-model ARNs are AWS-owned (no account id); the
            # meaningful restriction is the Anthropic model namespace, with a
            # region wildcard. The account-scoped inference-profile ARNs remain
            # the access-controlled resource.
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    f"arn:aws:bedrock:*:{account}:inference-profile/*",
                ],
            ),
            iam.PolicyStatement(
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                ],
                resources=["*"],  # profile discovery is an account-level action
            ),
            # CloudWatch Logs: Batch container logging (Lambda gets this from
            # its managed exec role). Scoped to this project's Batch log groups.
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{region}:{account}:log-group:/aws/batch/*",
                    f"arn:aws:logs:{region}:{account}:log-group:"
                    f"/aws/batch/*:log-stream:*",
                    f"arn:aws:logs:{region}:{account}:log-group:"
                    f"{self.project_name}-{self.stage}*",
                ],
            ),
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

    def _get_common_environment_vars(
        self,
        default_region_name: str | None,
        bedrock_region_name: str | None,
        topic_arn: str,
        **kwargs,
    ) -> dict[str, str]:
        env_vars = {
            EnvVars.BEDROCK_REGION_NAME.value: bedrock_region_name or "us-west-2",
            EnvVars.CONFIG_FILE_SUFFIX.value: self.stage,
            EnvVars.DEFAULT_REGION_NAME.value: default_region_name or "ap-northeast-2",
            EnvVars.LOG_LEVEL.value: "INFO",
            EnvVars.TOPIC_ARN.value: topic_arn,
        }
        env_vars.update(kwargs)
        return env_vars

    def _setup_lambda_resources(
        self, environment: dict[str, str], cron_expression: str | None
    ) -> None:
        function = self._create_lambda_function(environment)
        if function.role:
            self.topic.grant_publish(function.role)
        if cron_expression:
            self._create_event_rule(cron_expression, target=LambdaFunction(function))

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
                event=(
                    events.RuleTargetInput.from_object({"Parameters": parameters})
                    if parameters
                    else None
                ),
            )
            self._create_event_rule(cron_expression, target=batch_target)
        return job_queue.job_queue_name, job_definition.job_definition_name

    def _create_lambda_function(
        self, environment: dict[str, str]
    ) -> lambda_.DockerImageFunction:
        return lambda_.DockerImageFunction(
            self,
            "NewsletterFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=str(Path(__file__).parent.parent / "app"),
                file="Dockerfile-lambda",
                platform=Platform.LINUX_AMD64,
                exclude=["cdk.out", "__pycache__", "*.pyc", ".git"],
            ),
            description="Lambda function to send newsletter",
            environment=environment,
            function_name=self._get_resource_name("newsletter"),
            memory_size=512,
            role=self.role,
            security_groups=[self.security_group],
            timeout=Duration.minutes(15),
            vpc=self.vpc,
            vpc_subnets=self.vpc_subnets,
            retry_attempts=2,
        )

    def _create_job_queue(self) -> batch.JobQueue:
        job_queue = batch.JobQueue(
            self,
            "NewsletterJobQueue",
            job_queue_name=self._get_resource_name("newsletter"),
            priority=1,
            compute_environments=[],
        )

        compute_env_config = {
            "instance_role": self.role,
            "instance_types": [ec2.InstanceType("optimal")],
            "vpc": self.vpc,
            "security_groups": [self.security_group],
            "vpc_subnets": self.vpc_subnets,
        }

        ondemand_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "NewsletterOnDemandEnv",
            allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
            compute_environment_name=self._get_resource_name("newsletter-ondemand"),
            maxv_cpus=4,
            **compute_env_config,
        )
        job_queue.add_compute_environment(ondemand_env, 1)

        spot_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "NewsletterSpotEnv",
            allocation_strategy=batch.AllocationStrategy.SPOT_CAPACITY_OPTIMIZED,
            compute_environment_name=self._get_resource_name("newsletter-spot"),
            spot=True,
            maxv_cpus=8,
            **compute_env_config,
        )
        job_queue.add_compute_environment(spot_env, 2)

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
            exclude=["cdk.out", "__pycache__", "*.pyc", ".git"],
            build_args={"DOCKER_BUILDKIT": "1"},
        )

        container_definition = batch.EcsEc2ContainerDefinition(
            self,
            "NewsletterContainerDef",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image),
            cpu=1,
            memory=core.Size.mebibytes(512),
            job_role=self.role,
            command=[
                "python",
                "main.py",
                "--end-date",
                "Ref::end_date",
                "--language",
                "Ref::language",
                "--recipients",
                "Ref::recipients",
            ],
            environment=environment,
            logging=ecs.LogDriver.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}"
            ),
        )

        return batch.EcsJobDefinition(
            self,
            "NewsletterJobDefinition",
            container=container_definition,
            job_definition_name=self._get_resource_name("newsletter"),
            retry_attempts=2,
            timeout=core.Duration.hours(3),
        )

    def _create_event_rule(
        self, cron_expression: str, target: events.IRuleTarget
    ) -> None:
        rule = events.Rule(
            self,
            "NewsletterEventRule",
            schedule=events.Schedule.expression(cron_expression),
            description=f"Event rule to trigger Newsletter {self.lambda_or_batch}",
            # The scheduled run's language is carried in the job parameters, not
            # the rule name — don't bake a fixed "-ko" into the resource name.
            rule_name=self._get_resource_name(self.lambda_or_batch),
        )
        rule.add_target(target)

    def _store_ssm_parameters(
        self,
        job_queue_name: str | None,
        job_definition_name: str | None,
    ) -> None:
        # NOTE: the LangChain API key is intentionally NOT created here. CDK can
        # only emit a plaintext `String` SSM parameter (CloudFormation cannot
        # create a `SecureString`), which would leak the key into the synthesized
        # template / CloudFormation console. It is written out-of-band as a
        # SecureString by `_put_secure_ssm_parameter` at deploy time instead.
        ssm_params_to_create = {
            SSMParams.BATCH_JOB_QUEUE: job_queue_name,
            SSMParams.BATCH_JOB_DEFINITION: job_definition_name,
        }

        descriptions = {
            SSMParams.BATCH_JOB_QUEUE: "AWS Batch Job Queue Name for Tech Digest Newsletter",
            SSMParams.BATCH_JOB_DEFINITION: "AWS Batch Job Definition Name for Tech Digest Newsletter",
        }

        for param_enum, param_value in ssm_params_to_create.items():
            if param_value:
                param_name = f"/{self.project_name}/{self.stage}/{param_enum.value}"
                ssm.StringParameter(
                    self,
                    f"SsmParam{param_enum.name}",
                    parameter_name=param_name,
                    string_value=param_value,
                    description=descriptions[param_enum],
                    tier=ssm.ParameterTier.STANDARD,
                )


def _put_secure_ssm_parameter(
    boto_session: boto3.Session, name: str, value: str
) -> None:
    """Write a secret to SSM as a SecureString (encrypted with the account's
    default SSM KMS key). Done via the API rather than CDK because
    CloudFormation cannot create a SecureString — a CDK StringParameter would
    leak the value in plaintext into the synthesized template."""
    ssm_client = boto_session.client("ssm")
    ssm_client.put_parameter(
        Name=name,
        Value=value,
        Type="SecureString",
        Overwrite=True,
        Description="LangChain API Key (Tech Digest)",
    )
    logger.info("Stored SecureString SSM parameter '%s'", name)


def main() -> None:
    try:
        config = Config.load()
        profile_name = os.environ.get(EnvVars.AWS_PROFILE_NAME.value)

        logger.info(
            "Deploying infrastructure for '%s' in '%s' stage",
            config.resources.project_name,
            config.resources.stage,
        )

        boto_session = boto3.Session(
            region_name=config.resources.default_region_name, profile_name=profile_name
        )
        # Resolve the account from STS, falling back to CDK_DEFAULT_ACCOUNT when
        # no credentials are available (e.g. `cdk synth` validation in CI).
        try:
            account_id = get_account_id(boto_session)
        except Exception:
            account_id = os.environ.get("CDK_DEFAULT_ACCOUNT")
            if not account_id:
                raise
            logger.warning(
                "STS unavailable; using CDK_DEFAULT_ACCOUNT=%s for synth", account_id
            )

        # Non-secret tracing config goes into the container environment block.
        env_vars: dict[str, str] = {
            EnvVars.LANGCHAIN_TRACING_V2.value: os.environ.get(
                EnvVars.LANGCHAIN_TRACING_V2.value, "false"
            ),
            EnvVars.LANGCHAIN_ENDPOINT.value: os.environ.get(
                EnvVars.LANGCHAIN_ENDPOINT.value, ""
            ),
            EnvVars.LANGCHAIN_PROJECT.value: config.resources.project_name,
        }

        # The LangChain API key is a secret, so it must NOT be baked into the
        # synthesized CloudFormation template or the Lambda/Batch environment
        # block. Keep it out of env_vars entirely and write it separately as an
        # SSM SecureString once the stack is deployed.
        langchain_api_key = os.environ.get(EnvVars.LANGCHAIN_API_KEY.value)

        parameters = {
            "end_date": AppConstants.NULL_STRING,
            "language": "ko",
            "recipients": AppConstants.NULL_STRING,
        }

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
            vpc_id=config.resources.vpc_id,
            subnet_ids=config.resources.subnet_ids,
            lambda_or_batch=config.resources.lambda_or_batch,
            cron_expression=config.resources.cron_expression,
            email_addresses=[str(config.newsletter.sender)],
            default_region_name=config.resources.default_region_name,
            bedrock_region_name=config.resources.bedrock_region_name,
            environment_vars=env_vars,
            parameters=parameters,
            env=env,
        )
        app.synth()

        # Write the secret out-of-band as an SSM SecureString. Skipped during a
        # credential-less synth (e.g. CI) — there's nothing to write and no
        # session to write it with.
        if langchain_api_key and os.environ.get("CDK_SYNTH_ONLY") != "1":
            _put_secure_ssm_parameter(
                boto_session,
                f"/{config.resources.project_name}/{config.resources.stage}/"
                f"{SSMParams.LANGCHAIN_API_KEY.value}",
                langchain_api_key,
            )

    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
