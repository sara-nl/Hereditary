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


def prepare_flower_content_dir(flower_dir: str) -> str:
    """Minimal tree for FLARE custom/: pyproject.toml + flower package (no .venv/data)."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / flower_dir / "pyproject.toml", STAGING_DIR / "pyproject.toml")
    shutil.copy2(REPO_ROOT / flower_dir / "LICENSE", STAGING_DIR / "LICENSE")
    shutil.copytree(REPO_ROOT / flower_dir / "demo", STAGING_DIR / "demo")
    return str(STAGING_DIR)


def build_job(export_root: Path, flower_dir: str) -> Path:
    flower_content = prepare_flower_content_dir(flower_dir)
    job = FlowerJob(
        name=flower_dir,
        flower_content=flower_content,
        min_clients=MIN_CLIENTS,
        allow_runtime_dependency_installation=True
    )

    export_root.mkdir(parents=True, exist_ok=True)
    job.export_job(str(export_root))
    job_dir = export_root / flower_dir
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
        "--flower-dir",
        type=str,
        default="fs26-demo",
        choices=["fs26-demo", "fs26-demo-old"],
        help="Flower directory to use (default: fs26-demo)",
    )
    args = parser.parse_args()

    build_job(args.export_dir, args.flower_dir)


if __name__ == "__main__":
    main()
