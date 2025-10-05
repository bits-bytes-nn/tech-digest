import os
import sys
from pathlib import Path

import aws_cdk as core
import boto3
from aws_cdk import (
    Duration,
    Stack,
    Tags,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_events as events,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
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
        vpc_id: str | None = None,
        subnet_ids: list[str] | None = None,
        lambda_or_batch: str = "lambda",
        cron_expression: str | None = None,
        email_addresses: list[str] | None = None,
        default_region_name: str | None = None,
        bedrock_region_name: str | None = None,
        langchain_api_key: str | None = None,
        environment_vars: dict[str, str] | None = None,
        parameters: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.stage = stage
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

        self._store_ssm_parameters(
            langchain_api_key, job_queue_name, job_definition_name
        )

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
        base_policies = [
            "AmazonS3FullAccess",
            "AmazonSESFullAccess",
            "AmazonSNSFullAccess",
            "AmazonSSMFullAccess",
            "AmazonBedrockFullAccess",
        ]
        if self.lambda_or_batch == "lambda":
            service_principal = iam.ServicePrincipal("lambda.amazonaws.com")
            additional_policies = [
                "service-role/AWSLambdaBasicExecutionRole",
                "service-role/AWSLambdaVPCAccessExecutionRole",
            ]
        else:
            service_principal = iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            )
            additional_policies = [
                "AmazonEC2FullAccess",
                "AmazonECS_FullAccess",
                "AWSBatchFullAccess",
                "CloudWatchLogsFullAccess",
            ]

        all_policies = base_policies + additional_policies
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name(name)
            for name in all_policies
        ]

        return iam.Role(
            self,
            "NewsletterRole",
            assumed_by=service_principal,
            description="IAM role for Newsletter",
            managed_policies=managed_policies,
            role_name=self._get_resource_name("newsletter"),
        )

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
            rule_name=self._get_resource_name(f"{self.lambda_or_batch}-ko"),
        )
        rule.add_target(target)

    def _store_ssm_parameters(
        self,
        langchain_api_key: str | None,
        job_queue_name: str | None,
        job_definition_name: str | None,
    ) -> None:
        ssm_params_to_create = {
            SSMParams.LANGCHAIN_API_KEY: langchain_api_key,
            SSMParams.BATCH_JOB_QUEUE: job_queue_name,
            SSMParams.BATCH_JOB_DEFINITION: job_definition_name,
        }

        descriptions = {
            SSMParams.LANGCHAIN_API_KEY: "Langchain API Key",
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
        account_id = get_account_id(boto_session)

        env_vars = {
            key.value: os.environ.get(key.value, default)
            for key, default in [
                (EnvVars.LANGCHAIN_TRACING_V2, "false"),
                (EnvVars.LANGCHAIN_ENDPOINT, ""),
                (EnvVars.LANGCHAIN_API_KEY, None),
            ]
        }
        env_vars[EnvVars.LANGCHAIN_PROJECT.value] = config.resources.project_name
        env_vars = {k: v for k, v in env_vars.items() if v is not None}

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
            vpc_id=config.resources.vpc_id,
            subnet_ids=config.resources.subnet_ids,
            lambda_or_batch=config.resources.lambda_or_batch,
            cron_expression=config.resources.cron_expression,
            email_addresses=[str(config.newsletter.sender)],
            default_region_name=config.resources.default_region_name,
            bedrock_region_name=config.resources.bedrock_region_name,
            langchain_api_key=env_vars.pop(EnvVars.LANGCHAIN_API_KEY.value, None),
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
