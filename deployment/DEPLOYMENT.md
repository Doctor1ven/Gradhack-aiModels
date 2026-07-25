# SageMaker Custom Container Deployment

This deployment uses one reusable custom SageMaker inference container for both
models. It does not retrain, rename, overwrite, or delete the trained `.joblib`
files.

The custom image runs Python 3.12 so the endpoint can install and run these
exact model dependencies:

```text
scikit-learn==1.9.0
xgboost==3.3.0
pandas==3.0.5
joblib==1.5.3
numpy
```

## Files

- `Dockerfile`
- `container-requirements.txt`
- `app/server.py`
- `app/__init__.py`
- `build_and_push.sh`
- `readiness/model.tar.gz`
- `vo2/model.tar.gz`
- `deploy_models.py`
- `test_endpoints.py`
- `cleanup_endpoints.py`

Each model archive contains this layout at the archive root:

```text
model_file.joblib
code/inference.py
code/requirements.txt
```

The container is generic. At startup it imports `/opt/ml/model/code/inference.py`
from the extracted model archive, calls that module's `model_fn()`, then routes
SageMaker requests through the existing `input_fn()`, `predict_fn()`, and
`output_fn()`. This preserves the current request and response schemas.

## Endpoints

The endpoint names are unchanged:

```text
recovery-readiness-endpoint
vo2-forecast-endpoint
```

The readiness endpoint returns:

```json
{
  "readiness_label": 1,
  "readiness": "MAINTAIN",
  "confidence": 0.52,
  "probabilities": {
    "REDUCE": 0.08,
    "MAINTAIN": 0.52,
    "PROGRESS": 0.39
  }
}
```

The VO2 endpoint returns:

```json
{
  "current_vo2": 37.9,
  "predicted_vo2_4_weeks": 38.7,
  "predicted_change": 0.8
}
```

## Configuration

Important defaults in `deploy_models.py`:

```python
DEPLOYMENT_MODE = "serverless"
REGION = "eu-west-1"
S3_PREFIX = "gradhack-health-coach/models"
ECR_REPOSITORY_NAME = "gradhack-health-coach-inference"
ECR_IMAGE_TAG = "py312"
SERVERLESS_MEMORY_MB = 4096
SERVERLESS_MAX_CONCURRENCY = 5
REALTIME_INSTANCE_TYPE = "ml.m5.large"
REALTIME_INITIAL_INSTANCE_COUNT = 1
```

`deploy_models.py` does not hardcode an AWS account ID. It resolves the account
with STS and derives the default image URI:

```text
<account-id>.dkr.ecr.eu-west-1.amazonaws.com/gradhack-health-coach-inference:py312
```

You can override the image at deployment time:

```bash
python deploy_models.py --image-uri <full-ecr-image-uri>
```

## Prerequisites

Run these commands from a SageMaker Studio/JupyterLab terminal with Docker
available.

Install client dependencies:

```bash
pip install boto3 sagemaker pandas
```

Confirm AWS identity and region:

```bash
aws sts get-caller-identity
aws configure get region
```

Use `eu-west-1` for all commands:

```bash
export AWS_REGION=eu-west-1
```

## Build And Push Image

From the `deployment` folder:

```bash
cd deployment
```

Create the ECR repository:

```bash
aws ecr create-repository \
  --repository-name gradhack-health-coach-inference \
  --region eu-west-1
```

If the repository already exists, this command returns an error. That is safe;
continue with login/build/push.

Set account and image variables without hardcoding the account ID:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com/gradhack-health-coach-inference:py312"
```

Log Docker into ECR:

```bash
aws ecr get-login-password --region eu-west-1 \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com"
```

Build the image:

```bash
docker build --platform linux/amd64 -t gradhack-health-coach-inference:py312 .
```

Tag the image:

```bash
docker tag gradhack-health-coach-inference:py312 "${IMAGE_URI}"
```

Push the image:

```bash
docker push "${IMAGE_URI}"
```

Shortcut command:

```bash
bash build_and_push.sh
```

The shortcut creates the repository if needed, logs in, builds, pushes, and
prints the final image URI.

## Deploy

Serverless is the default:

```bash
python deploy_models.py
```

This uploads:

```text
s3://<bucket>/gradhack-health-coach/models/readiness/model.tar.gz
s3://<bucket>/gradhack-health-coach/models/vo2/model.tar.gz
```

Then it creates:

- SageMaker model resource `recovery-readiness-model`
- SageMaker model resource `vo2-forecast-model`
- endpoint config for `recovery-readiness-endpoint`
- endpoint config for `vo2-forecast-endpoint`
- endpoint `recovery-readiness-endpoint`
- endpoint `vo2-forecast-endpoint`

If serverless custom containers are not compatible in your account or region,
use the real-time fallback:

```bash
python deploy_models.py --mode realtime
```

To deploy with an explicit image URI:

```bash
python deploy_models.py --image-uri "${IMAGE_URI}"
```

To pass an explicit SageMaker execution role outside Studio:

```bash
python deploy_models.py --role-arn arn:aws:iam::<account-id>:role/<role-name>
```

## Test

After both endpoints are `InService`:

```bash
python test_endpoints.py
```

The tester waits for both endpoints, sends one row from
`../longitudinal_data/test.csv` to the readiness endpoint, sends one row from
`../vo2_data/test.csv` to the VO2 endpoint, and prints both JSON responses.

## Logs

Endpoint logs are in CloudWatch:

```text
/aws/sagemaker/Endpoints/recovery-readiness-endpoint
/aws/sagemaker/Endpoints/vo2-forecast-endpoint
```

Use these logs for container startup failures, model loading failures, request
parsing errors, and prediction errors.

## Cleanup

Delete deployed SageMaker resources:

```bash
python cleanup_endpoints.py
```

Equivalent command:

```bash
python deploy_models.py --cleanup
```

Cleanup deletes both endpoints, their active endpoint configs, and both
SageMaker model resources. It does not delete the ECR image, ECR repository,
S3 model artifacts, or S3 bucket.
