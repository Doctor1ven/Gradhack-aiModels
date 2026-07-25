"""Invoke the two deployed SageMaker endpoints with one local test row each."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
import pandas as pd


REGION = "eu-west-1"
READINESS_ENDPOINT_NAME = "recovery-readiness-endpoint"
VO2_ENDPOINT_NAME = "vo2-forecast-endpoint"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
READINESS_TEST_PATH = PROJECT_ROOT / "longitudinal_data" / "test.csv"
VO2_TEST_PATH = PROJECT_ROOT / "vo2_data" / "test.csv"

CONTENT_TYPE = "application/json"
MAX_INVOKE_ATTEMPTS = 8
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
ENDPOINT_WAIT_SECONDS = 30
ENDPOINT_MAX_WAIT_SECONDS = 1800

RETRYABLE_ERROR_CODES = {
    "ModelNotReadyException",
    "ServiceUnavailable",
    "InternalFailure",
    "InternalServerException",
    "ThrottlingException",
    "Throttling",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
}


def wait_for_endpoint(endpoint_name: str) -> None:
    sm_client = boto3.Session(region_name=REGION).client("sagemaker")
    started = time.time()
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"{endpoint_name} status: {status}")
        if status == "InService":
            return
        if status == "Failed":
            reason = response.get("FailureReason", "No failure reason returned.")
            raise RuntimeError(f"{endpoint_name} failed: {reason}")
        if time.time() - started > ENDPOINT_MAX_WAIT_SECONDS:
            raise TimeoutError(f"Timed out waiting for {endpoint_name} to become InService.")
        time.sleep(ENDPOINT_WAIT_SECONDS)


def is_retryable_error(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = error.get("Code", "")
    message = error.get("Message", "")
    return code in RETRYABLE_ERROR_CODES or any(
        text in message.lower()
        for text in ["modelnotready", "service unavailable", "throttl", "internal failure"]
    )


def invoke(endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = boto3.Session(region_name=REGION).client("sagemaker-runtime")
    body = json.dumps(payload)
    for attempt in range(1, MAX_INVOKE_ATTEMPTS + 1):
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=CONTENT_TYPE,
                Accept=CONTENT_TYPE,
                Body=body,
            )
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as exc:
            if not is_retryable_error(exc) or attempt == MAX_INVOKE_ATTEMPTS:
                raise
            sleep_seconds = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0, 1)
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            print(f"{endpoint_name} retryable invoke error {code}; retrying in {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to invoke {endpoint_name}.")


def main() -> None:
    wait_for_endpoint(READINESS_ENDPOINT_NAME)
    wait_for_endpoint(VO2_ENDPOINT_NAME)

    readiness_row = pd.read_csv(READINESS_TEST_PATH).iloc[0].drop(labels=["future_readiness_label"]).to_dict()
    vo2_row = pd.read_csv(VO2_TEST_PATH).iloc[0].drop(labels=["future_vo2_4_weeks"]).to_dict()

    readiness_response = invoke(READINESS_ENDPOINT_NAME, {"instances": [readiness_row]})
    vo2_response = invoke(VO2_ENDPOINT_NAME, {"instances": [vo2_row]})

    print("Recovery readiness response:")
    print(json.dumps(readiness_response, indent=2))
    print("\nVO2 forecast response:")
    print(json.dumps(vo2_response, indent=2))


if __name__ == "__main__":
    main()
