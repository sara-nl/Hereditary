#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Builds an NVIDIA FLARE job that wraps this Flower app (FlowerJob API).
import argparse
import shutil
from pathlib import Path

from nvflare.app_opt.flower.flower_job import FlowerJob

REPO_ROOT = Path(__file__).resolve().parent
MIN_CLIENTS = 3
STAGING_DIR = REPO_ROOT / "staging" / "flower_payload"



def build_job(export_root: Path, flower_app_path: str, job_name: str) -> Path:
    job = FlowerJob(
        name=job_name,
        flower_app_path="local/custom/" + flower_app_path,
        min_clients=MIN_CLIENTS,
        allow_runtime_dependency_installation=True
    )

    export_root.mkdir(parents=True, exist_ok=True)
    job.export_job(str(export_root))
    print(f"Exported job to {export_root / job_name}")
    return export_root / job_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NVFlare + Flower job folder.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "nvflare_workspace" / "jobs",
        help="Directory that will contain the job folder (default: ./nvflare_workspace/jobs)",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="fs26-demo-no-byoc",
        help="Job name (default: fs26-demo-no-byoc)",
    )
    parser.add_argument(
        "--flower-app-path",
        type=str,
        required=True,
        help="Predeployed Flower app path (required)",
    )
    args = parser.parse_args()

    build_job(args.export_dir, args.flower_app_path, args.job_name)


if __name__ == "__main__":
    main()
