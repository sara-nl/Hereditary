#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Builds an NVIDIA FLARE job that wraps this Flower app (FlowerJob API).
import argparse
import shutil
from pathlib import Path

from nvflare.app_opt.flower.flower_job import FlowerJob
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver

REPO_ROOT = Path(__file__).resolve().parent
MIN_CLIENTS = 3
STAGING_DIR = REPO_ROOT / "staging" / "flower_payload"


def prepare_flower_content_dir(source_dir: Path) -> str:
    """Minimal tree for FLARE custom/: pyproject.toml + package directory."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    
    EXCLUDE_DIRS = {
        ".venv", "data", "__pycache__", ".vscode", ".git",
        "nvflare_workspace", "dist", "staging", "custom4client",
    }
    # Copy everything from the source directory to the staging directory
    for item in source_dir.iterdir():
        if item.is_dir():
            if item.name not in EXCLUDE_DIRS:
                shutil.copytree(item, STAGING_DIR / item.name)
        else:
            shutil.copy2(item, STAGING_DIR / item.name)
            
    return str(STAGING_DIR)


def build_job(source_dir: Path, job_name: str, export_root: Path, stream_metrics: bool = True) -> Path:
    # Warn if this doesn't look like a Flower job
    if not (source_dir / "pyproject.toml").exists():
        print(f"Warning: '{source_dir}' does not contain pyproject.toml. "
              f"Currently, job.py uses the FlowerJob API. This may not work for native NVFlare jobs "
              f"unless you add the appropriate NVFlare Job API builder logic.")

    flower_content = prepare_flower_content_dir(source_dir)
    job = FlowerJob(
        name=job_name,
        flower_content=flower_content,
        min_clients=MIN_CLIENTS,
    )
    if stream_metrics:
        job.to_server(TBAnalyticsReceiver(tb_folder="tb_events"))

    export_root.mkdir(parents=True, exist_ok=True)
    job.export_job(str(export_root))
    job_dir = export_root / job_name
    print(f"Exported job to {job_dir}")

    # Proactive cleanup: remove staging folder after job packaging is done
    staging_root = STAGING_DIR.parent
    if staging_root.exists():
        shutil.rmtree(staging_root)

    return job_dir


JOB_NAME = "cifar10_flower"  # The exported NVFlare job name
JOB_SOURCE = REPO_ROOT  # pyproject.toml + flwr_pt_tb/ live directly here


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NVFlare + Flower job folder.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "nvflare_workspace" / "jobs",
        help="Directory that will contain the job folder (default: ./nvflare_workspace/jobs)",
    )
    parser.add_argument(
        "--no-stream-metrics",
        action="store_true",
        help="Disable server-side TensorBoard receiver (client NVFlare SummaryWriter will no-op).",
    )
    args = parser.parse_args()

    if not JOB_SOURCE.exists():
        raise FileNotFoundError(f"Job code directory not found: {JOB_SOURCE}")

    build_job(JOB_SOURCE, JOB_NAME, args.export_dir, stream_metrics=not args.no_stream_metrics)


if __name__ == "__main__":
    main()
