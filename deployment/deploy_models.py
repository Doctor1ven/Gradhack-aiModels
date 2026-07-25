"""Deploy packaged models to SageMaker Serverless or real-time endpoints.

Run this from SageMaker Studio/JupyterLab. Serverless Inference is the default
for hackathon testing; real-time endpoints remain available as a fallback.
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
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel


DEPLOYMENT_MODE = "serverless"
VALID_DEPLOYMENT_MODES = {"serverless", "realtime"}

SERVERLESS_MEMORY_MB = 4096
SERVERLESS_MAX_CONCURRENCY = 5

REALTIME_INSTANCE_TYPE = "ml.m5.large"
REALTIME_INITIAL_INSTANCE_COUNT = 1

REGION = "eu-west-1"
S3_BUCKET = ""
S3_PREFIX = "gradhack-health-coach/models"

READINESS_ENDPOINT_NAME = "recovery-readiness-endpoint"
VO2_ENDPOINT_NAME = "vo2-forecast-endpoint"

FRAMEWORK_VERSION = "1.2-1"
PY_VERSION = "py3"

ENDPOINT_WAIT_SECONDS = 60
ENDPOINT_MAX_WAIT_SECONDS = 3600

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "readiness": {
        "archive": BASE_DIR / "readiness" / "model.tar.gz",
        "code_dir": BASE_DIR / "readiness" / "code",
        "endpoint_name": READINESS_ENDPOINT_NAME,
        "model_name": "recovery-readiness-model",
    },
    "vo2": {
        "archive": BASE_DIR / "vo2" / "model.tar.gz",
        "code_dir": BASE_DIR / "vo2" / "code",
        "endpoint_name": VO2_ENDPOINT_NAME,
        "model_name": "vo2-forecast-model",
    },
}


def make_session() -> sagemaker.Session:
    boto_session = boto3.Session(region_name=REGION)
    return sagemaker.Session(boto_session=boto_session)


def resolve_role(role_arn: str | None) -> str:
    if role_arn:
        return role_arn
    env_role = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
    if env_role:
        return env_role
    return sagemaker.get_execution_role()


def resolve_bucket(sm_session: sagemaker.Session, configured_bucket: str | None) -> str:
    return configured_bucket or S3_BUCKET or sm_session.default_bucket()


def validate_mode(mode: str) -> None:
    if mode not in VALID_DEPLOYMENT_MODES:
        raise ValueError(f"DEPLOYMENT_MODE must be one of {sorted(VALID_DEPLOYMENT_MODES)}.")


def validate_archives() -> None:
    missing = [str(config["archive"]) for config in MODELS.values() if not config["archive"].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model archives: {missing}")


def print_configuration(mode: str, bucket: str, role: str) -> None:
    print("Deployment configuration")
    print(f"AWS region: {REGION}")
    print(f"SageMaker SDK version: {sagemaker.__version__}")
    print(f"Selected deployment mode: {mode}")
    print(f"S3 bucket: {bucket}")
    print(f"S3 prefix: {S3_PREFIX}")
    print(f"Execution role: {role}")
    print(f"Readiness endpoint: {READINESS_ENDPOINT_NAME}")
    print(f"VO2 endpoint: {VO2_ENDPOINT_NAME}")
    print(f"Serverless memory MB: {SERVERLESS_MEMORY_MB}")
    print(f"Serverless maximum concurrency: {SERVERLESS_MAX_CONCURRENCY}")
    print(f"Real-time fallback instance type: {REALTIME_INSTANCE_TYPE}")
    print(f"Real-time initial instance count: {REALTIME_INITIAL_INSTANCE_COUNT}")
    print(f"Selected framework version: {FRAMEWORK_VERSION}")
    print(f"Selected Python version: {PY_VERSION}")
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


def make_model(config: dict[str, Any], model_data: str, role: str, sm_session: sagemaker.Session) -> SKLearnModel:
    return SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir=str(config["code_dir"]),
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sm_session,
        name=config["model_name"],
    )


def deploy_model(model: SKLearnModel, endpoint_name: str, mode: str) -> None:
    if mode == "serverless":
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=SERVERLESS_MEMORY_MB,
            max_concurrency=SERVERLESS_MAX_CONCURRENCY,
        )
        model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
        )
    else:
        model.deploy(
            initial_instance_count=REALTIME_INITIAL_INSTANCE_COUNT,
            instance_type=REALTIME_INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )


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


def deploy(mode: str, bucket_override: str | None, role_arn: str | None) -> None:
    validate_mode(mode)
    validate_archives()

    sm_session = make_session()
    bucket = resolve_bucket(sm_session, bucket_override)
    role = resolve_role(role_arn)
    print_configuration(mode, bucket, role)
    model_data = upload_archives(sm_session, bucket)
    sm_client = boto3.Session(region_name=REGION).client("sagemaker")

    for key, config in MODELS.items():
        endpoint_name = config["endpoint_name"]
        print(f"Deploying {key} endpoint: {endpoint_name}")
        model = make_model(config, model_data[key], role, sm_session)
        deploy_model(model, endpoint_name, mode)
        status = wait_for_endpoint(sm_client, endpoint_name)
        if status != "InService":
            raise RuntimeError(f"{endpoint_name} did not deploy successfully.")
        print(f"Confirmed InService: {endpoint_name}")


def cleanup() -> None:
    sm_client = boto3.Session(region_name=REGION).client("sagemaker")
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
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
    else:
        deploy(args.mode, args.s3_bucket, args.role_arn)


if __name__ == "__main__":
    main()
