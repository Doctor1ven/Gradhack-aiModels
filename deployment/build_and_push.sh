#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-eu-west-1}"
REPOSITORY_NAME="${1:-gradhack-health-coach-inference}"
IMAGE_TAG="${2:-py312}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

aws ecr describe-repositories \
  --repository-names "${REPOSITORY_NAME}" \
  --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
    --repository-name "${REPOSITORY_NAME}" \
    --region "${REGION}" >/dev/null

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# SageMaker requires a single-platform Docker v2 manifest. Without these
# flags, buildx produces an OCI image index with a provenance attestation,
# which SageMaker rejects at CreateModel with "Unsupported manifest media
# type".
docker build --platform linux/amd64 --provenance=false --sbom=false \
  --output type=docker -t "${REPOSITORY_NAME}:${IMAGE_TAG}" .
docker tag "${REPOSITORY_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "${IMAGE_URI}"
