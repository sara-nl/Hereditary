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
JOB_NAME = "cifar10_flower"
MIN_CLIENTS = 3
STAGING_DIR = REPO_ROOT / "staging" / "flower_payload"


def prepare_flower_content_dir() -> str:
    """Minimal tree for FLARE custom/: pyproject.toml + flwr_pt_tb package (no .venv/data)."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "pyproject.toml", STAGING_DIR / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "flwr_pt_tb", STAGING_DIR / "flwr_pt_tb")
    return str(STAGING_DIR)


def build_job(export_root: Path, stream_metrics: bool = True) -> Path:
    flower_content = prepare_flower_content_dir()
    job = FlowerJob(
        name=JOB_NAME,
        flower_content=flower_content,
        min_clients=MIN_CLIENTS,
    )
    if stream_metrics:
        job.to_server(TBAnalyticsReceiver(tb_folder="tb_events"))

    export_root.mkdir(parents=True, exist_ok=True)
    job.export_job(str(export_root))
    job_dir = export_root / JOB_NAME
    print(f"Exported job to {job_dir}")
    return job_dir


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

    build_job(args.export_dir, stream_metrics=not args.no_stream_metrics)


if __name__ == "__main__":
    main()
