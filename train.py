import argparse
import os
import sys

from xray.exception import XRayException
from xray.pipeline.train_pipeline import TrainPipeline


def load_env_file(path=".env"):
    """Load simple KEY=VALUE entries without logging credential values."""
    if not os.path.isfile(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                # The project-local .env is explicit configuration and should
                # override stale credentials inherited from the shell.
                os.environ[key] = value


def start_training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise XRayException(e, sys)


if __name__ == "__main__":
    load_env_file()
    parser = argparse.ArgumentParser(description="Train the lung X-ray classifier")
    parser.add_argument(
        "--data-dir",
        help="Folder containing train/NORMAL, train/PNEUMONIA, "
        "test/NORMAL, and test/PNEUMONIA",
    )
    args = parser.parse_args()
    if args.data_dir:
        os.environ["XRAY_DATA_DIR"] = os.path.abspath(args.data_dir)
    start_training()
