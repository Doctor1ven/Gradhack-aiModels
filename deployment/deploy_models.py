"""Deploy packaged models with a reusable custom SageMaker inference image.

Run this from SageMaker Studio/JupyterLab after building and pushing the custom
container image to Amazon ECR. Serverless Inference is the default; real-time
endpoints remain available as a fallback.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
import sagemaker


DEPLOYMENT_MODE = "serverless"
VALID_DEPLOYMENT_MODES = {"serverless", "realtime"}

SERVERLESS_MEMORY_MB = 4096
SERVERLESS_MAX_CONCURRENCY = 5

REALTIME_INSTANCE_TYPE = "ml.m5.large"
REALTIME_INITIAL_INSTANCE_COUNT = 1

REGION = "eu-west-1"
S3_BUCKET = ""
S3_PREFIX = "gradhack-health-coach/models"

ECR_REPOSITORY_NAME = "gradhack-health-coach-inference"
ECR_IMAGE_TAG = "py312"
ECR_IMAGE_URI = ""

READINESS_ENDPOINT_NAME = "recovery-readiness-endpoint"
VO2_ENDPOINT_NAME = "vo2-forecast-endpoint"

ENDPOINT_WAIT_SECONDS = 60
ENDPOINT_MAX_WAIT_SECONDS = 3600

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "readiness": {
        "archive": BASE_DIR / "readiness" / "model.tar.gz",
        "endpoint_name": READINESS_ENDPOINT_NAME,
        "model_name": "recovery-readiness-model",
        "container_env": {"MODEL_KIND": "readiness"},
    },
    "vo2": {
        "archive": BASE_DIR / "vo2" / "model.tar.gz",
        "endpoint_name": VO2_ENDPOINT_NAME,
        "model_name": "vo2-forecast-model",
        "container_env": {"MODEL_KIND": "vo2"},
    },
}


def make_boto_session() -> boto3.Session:
    return boto3.Session(region_name=REGION)


def make_sagemaker_session() -> sagemaker.Session:
    return sagemaker.Session(boto_session=make_boto_session())


def resolve_role(role_arn: str | None) -> str:
    if role_arn:
        return role_arn
    env_role = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
    if env_role:
        return env_role
    return sagemaker.get_execution_role()


def resolve_bucket(sm_session: sagemaker.Session, configured_bucket: str | None) -> str:
    return configured_bucket or S3_BUCKET or sm_session.default_bucket()


def resolve_image_uri(image_uri: str | None) -> str:
    if image_uri:
        return image_uri
    env_image_uri = os.getenv("SAGEMAKER_INFERENCE_IMAGE_URI")
    if env_image_uri:
        return env_image_uri
    if ECR_IMAGE_URI:
        return ECR_IMAGE_URI

    account_id = make_boto_session().client("sts").get_caller_identity()["Account"]
    return f"{account_id}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPOSITORY_NAME}:{ECR_IMAGE_TAG}"


def validate_mode(mode: str) -> None:
    if mode not in VALID_DEPLOYMENT_MODES:
        raise ValueError(f"DEPLOYMENT_MODE must be one of {sorted(VALID_DEPLOYMENT_MODES)}.")


def validate_archives() -> None:
    missing = [str(config["archive"]) for config in MODELS.values() if not config["archive"].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model archives: {missing}")


def print_configuration(mode: str, bucket: str, role: str, image_uri: str) -> None:
    print("Deployment configuration")
    print(f"AWS region: {REGION}")
    print(f"SageMaker SDK version: {sagemaker.__version__}")
    print(f"Selected deployment mode: {mode}")
    print(f"S3 bucket: {bucket}")
    print(f"S3 prefix: {S3_PREFIX}")
    print(f"Execution role: {role}")
    print(f"Custom inference image: {image_uri}")
    print(f"Readiness endpoint: {READINESS_ENDPOINT_NAME}")
    print(f"VO2 endpoint: {VO2_ENDPOINT_NAME}")
    print(f"Serverless memory MB: {SERVERLESS_MEMORY_MB}")
    print(f"Serverless maximum concurrency: {SERVERLESS_MAX_CONCURRENCY}")
    print(f"Real-time fallback instance type: {REALTIME_INSTANCE_TYPE}")
    print(f"Real-time initial instance count: {REALTIME_INITIAL_INSTANCE_COUNT}")
    print("Archive paths:")
    for key, config in MODELS.items():
        print(f"  {key}: {config['archive']}")


def upload_archives(sm_session: sagemaker.Session, bucket: str) -> dict[str, str]:
    uploaded: dict[str, str] = {}
    for key, config in MODELS.items():
        archive = config["archive"]
        uploaded[key] = sm_session.upload_data(
            path=str(archive),
            bucket=bucket,
            key_prefix=f"{S3_PREFIX}/{key}",
        )
        print(f"Uploaded {key}: {uploaded[key]}")
    return uploaded


def delete_model_if_exists(sm_client: Any, model_name: str) -> None:
    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"Deleted existing model resource: {model_name}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in {"ValidationException", "ResourceNotFound"}:
            raise


def create_model(
    sm_client: Any,
    config: dict[str, Any],
    model_data: str,
    role: str,
    image_uri: str,
) -> None:
    model_name = config["model_name"]
    delete_model_if_exists(sm_client, model_name)
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
            "Environment": config["container_env"],
        },
    )
    print(f"Created model resource: {model_name}")


def create_endpoint_config(sm_client: Any, config: dict[str, Any], mode: str) -> str:
    endpoint_config_name = f"{config['endpoint_name']}-{int(time.time())}"
    variant: dict[str, Any] = {
        "VariantName": "AllTraffic",
        "ModelName": config["model_name"],
    }
    if mode == "serverless":
        variant["ServerlessConfig"] = {
            "MemorySizeInMB": SERVERLESS_MEMORY_MB,
            "MaxConcurrency": SERVERLESS_MAX_CONCURRENCY,
        }
    else:
        variant["InitialInstanceCount"] = REALTIME_INITIAL_INSTANCE_COUNT
        variant["InstanceType"] = REALTIME_INSTANCE_TYPE

    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[variant],
    )
    print(f"Created endpoint config: {endpoint_config_name}")
    return endpoint_config_name


def endpoint_exists(sm_client: Any, endpoint_name: str) -> bool:
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"ValidationException", "ResourceNotFound"}:
            return False
        raise


def deploy_endpoint(sm_client: Any, endpoint_name: str, endpoint_config_name: str) -> None:
    if endpoint_exists(sm_client, endpoint_name):
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        print(f"Updated endpoint: {endpoint_name}")
    else:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        print(f"Created endpoint: {endpoint_name}")


def wait_for_endpoint(sm_client: Any, endpoint_name: str) -> str:
    started = time.time()
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"{endpoint_name} status: {status}")
        if status == "InService":
            return status
        if status == "Failed":
            reason = response.get("FailureReason", "No failure reason returned.")
            print(f"{endpoint_name} failed: {reason}")
            return status
        if time.time() - started > ENDPOINT_MAX_WAIT_SECONDS:
            raise TimeoutError(f"Timed out waiting for {endpoint_name} to become InService.")
        time.sleep(ENDPOINT_WAIT_SECONDS)


def deploy(mode: str, bucket_override: str | None, role_arn: str | None, image_uri_arg: str | None) -> None:
    validate_mode(mode)
    validate_archives()

    boto_session = make_boto_session()
    sm_client = boto_session.client("sagemaker")
    sm_session = make_sagemaker_session()
    bucket = resolve_bucket(sm_session, bucket_override)
    role = resolve_role(role_arn)
    image_uri = resolve_image_uri(image_uri_arg)

    print_configuration(mode, bucket, role, image_uri)
    model_data = upload_archives(sm_session, bucket)

    for key, config in MODELS.items():
        endpoint_name = config["endpoint_name"]
        print(f"Deploying {key} endpoint: {endpoint_name}")
        create_model(sm_client, config, model_data[key], role, image_uri)
        endpoint_config_name = create_endpoint_config(sm_client, config, mode)
        deploy_endpoint(sm_client, endpoint_name, endpoint_config_name)
        status = wait_for_endpoint(sm_client, endpoint_name)
        if status != "InService":
            raise RuntimeError(f"{endpoint_name} did not deploy successfully.")
        print(f"Confirmed InService: {endpoint_name}")


def cleanup() -> None:
    sm_client = make_boto_session().client("sagemaker")
    for config in MODELS.values():
        endpoint_name = config["endpoint_name"]
        model_name = config["model_name"]
        try:
            endpoint = sm_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_config_name = endpoint["EndpointConfigName"]
        except ClientError:
            endpoint_config_name = endpoint_name

        delete_operations = [
            (sm_client.delete_endpoint, {"EndpointName": endpoint_name}),
            (sm_client.delete_endpoint_config, {"EndpointConfigName": endpoint_config_name}),
            (sm_client.delete_model, {"ModelName": model_name}),
        ]
        for action, kwargs in delete_operations:
            try:
                action(**kwargs)
                print(f"Deleted {kwargs}")
            except ClientError as exc:
                print(f"Skipped {kwargs}: {exc.response['Error']['Message']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Delete endpoints, endpoint configs, and model resources.")
    parser.add_argument("--mode", choices=sorted(VALID_DEPLOYMENT_MODES), default=DEPLOYMENT_MODE)
    parser.add_argument("--s3-bucket", default=None, help="Override S3_BUCKET. Defaults to SageMaker's default bucket.")
    parser.add_argument("--role-arn", default=None, help="Optional execution role ARN fallback outside SageMaker Studio.")
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Custom ECR image URI. Defaults to account-local ECR repo/tag or SAGEMAKER_INFERENCE_IMAGE_URI.",
    )
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
    else:
        deploy(args.mode, args.s3_bucket, args.role_arn, args.image_uri)


if __name__ == "__main__":
    main()
