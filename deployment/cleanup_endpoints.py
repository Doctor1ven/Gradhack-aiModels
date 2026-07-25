"""Delete SageMaker endpoints, endpoint configs, and model resources."""

from __future__ import annotations

from deploy_models import cleanup


if __name__ == "__main__":
    cleanup()
