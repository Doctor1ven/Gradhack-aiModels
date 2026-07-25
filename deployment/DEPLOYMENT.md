# SageMaker Serverless Deployment Package

This folder packages two existing trained models for Amazon SageMaker AI. It
does not retrain, rename, overwrite, or delete the trained model files.

Serverless Inference is the default because it is a good fit for hackathon
testing: there is no always-on instance to manage, traffic can be intermittent,
and costs are easier to limit while endpoints are idle. Standard real-time
endpoints remain available as a fallback.

## Files

- `readiness/model/recovery_readiness_longitudinal_pipeline.joblib`
- `readiness/code/inference.py`
- `readiness/code/requirements.txt`
- `readiness/model.tar.gz`
- `vo2/model/vo2_forecast_pipeline.joblib`
- `vo2/code/inference.py`
- `vo2/code/requirements.txt`
- `vo2/model.tar.gz`
- `deploy_models.py`
- `test_endpoints.py`
- `cleanup_endpoints.py`
- `DEPLOYMENT.md`

Each archive contains this SageMaker layout at the archive root:

- `model_file.joblib`
- `code/inference.py`
- `code/requirements.txt`

The endpoint containers install XGBoost and the other model dependencies from
each archive's `code/requirements.txt`.

## Serverless vs Real-Time

Serverless endpoints start compute only when requests arrive. They are simpler
for low-volume demos, but the first request after idle time can have a cold
start while SageMaker provisions capacity and loads the model.

Real-time endpoints keep instances running. They usually avoid cold starts, but
the instance accrues charges while it exists. Use real-time mode if serverless
startup or package compatibility fails and you need the `ml.m5.large` fallback.

## Configuration

Open `deployment/deploy_models.py`. The important defaults are:

```python
DEPLOYMENT_MODE = "serverless"
SERVERLESS_MEMORY_MB = 4096
SERVERLESS_MAX_CONCURRENCY = 5
REALTIME_INSTANCE_TYPE = "ml.m5.large"
REALTIME_INITIAL_INSTANCE_COUNT = 1
REGION = "eu-west-1"
S3_PREFIX = "gradhack-health-coach/models"
```

To switch to real-time fallback, either edit:

```python
DEPLOYMENT_MODE = "realtime"
```

or run:

```bash
python deploy_models.py --mode realtime
```

To tune serverless capacity, change `SERVERLESS_MEMORY_MB` and
`SERVERLESS_MAX_CONCURRENCY`.

## S3 Bucket

The deploy script uses the SageMaker default S3 bucket unless `S3_BUCKET` is
set in `deploy_models.py` or `--s3-bucket` is passed on the command line.

If you create your own bucket, create it in `eu-west-1`. The script uploads:

```text
s3://<bucket>/gradhack-health-coach/models/readiness/model.tar.gz
s3://<bucket>/gradhack-health-coach/models/vo2/model.tar.gz
```

The script does not hardcode an AWS account ID or IAM role ARN. Inside
SageMaker Studio it uses `sagemaker.get_execution_role()`. Outside Studio, pass
`--role-arn` or set `SAGEMAKER_EXECUTION_ROLE_ARN`.

## SageMaker Studio Setup

Upload the project folder, including `deployment`, into SageMaker Studio
JupyterLab. Open a terminal in the `deployment` folder and install client
dependencies:

```bash
pip install boto3 sagemaker pandas
```

## Deploy

From the `deployment` folder in SageMaker Studio JupyterLab:

```bash
python deploy_models.py
```

The script prints the region, SageMaker SDK version, deployment mode, S3 bucket,
S3 prefix, role, endpoint names, serverless settings, real-time fallback
settings, archive paths, framework version, and Python version before it
deploys.

It then uploads both archives, creates separate endpoints, and waits until each
endpoint is `InService`. If an endpoint reaches `Failed`, the script prints the
failure reason.

## Test

After deployment:

```bash
python test_endpoints.py
```

The tester waits for both endpoints to be `InService`, sends one row from
`../longitudinal_data/test.csv` to `recovery-readiness-endpoint`, sends one row
from `../vo2_data/test.csv` to `vo2-forecast-endpoint`, and prints both JSON
responses. It retries only cold-start/startup style SageMaker invocation
errors, such as `ModelNotReadyException`, service unavailable errors, internal
failures, and throttling. It does not retry malformed requests or
missing-feature errors.

## Logs

Endpoint logs are in Amazon CloudWatch:

```text
/aws/sagemaker/Endpoints/recovery-readiness-endpoint
/aws/sagemaker/Endpoints/vo2-forecast-endpoint
```

Use these logs for dependency installation errors, model loading failures,
request parsing errors, and prediction failures.

## Cleanup

Real-time and serverless endpoints can incur charges. Delete resources after
testing:

```bash
python cleanup_endpoints.py
```

Equivalent command:

```bash
python deploy_models.py --cleanup
```

Cleanup deletes both endpoints, both endpoint configurations, and both
SageMaker model resources. It does not delete the S3 bucket.

## Compatibility Warning

The local models were saved with pinned package versions in each
`code/requirements.txt`, including XGBoost. Managed SageMaker scikit-learn
containers may still have package compatibility limits. If container startup
fails, inspect CloudWatch logs first. If the managed image cannot install the
required versions, either use real-time fallback with:

```bash
python deploy_models.py --mode realtime
```

or build a custom inference image with the pinned dependencies.
