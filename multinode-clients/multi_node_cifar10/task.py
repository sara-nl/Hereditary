"""Utility functions for federated CIFAR-10 training.

Handles saving/loading model checkpoints and launching multi-node
distributed training jobs via sbatch.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_global_weights(state_dict: dict, path: str) -> None:
    """Save a state_dict as a CIFAR-compatible checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict}, path)
    logger.info("Saved global weights to %s (%d keys)", path, len(state_dict))


def load_updated_weights(path: str) -> dict:
    """Load model weights from either a safetensors file or a CIFAR checkpoint.pt."""
    if path.endswith(".safetensors"):
        state_dict = safe_load(path)
    else:
        full_ckpt = torch.load(path, map_location="cpu")
        state_dict = full_ckpt["model_state_dict"]

    logger.info("Loaded updated weights from %s (%d keys)", path, len(state_dict))

    keys_to_remove = [k for k in state_dict.keys() if k.endswith("_extra_state")]
    for k in keys_to_remove:
        del state_dict[k]

    if keys_to_remove:
        logger.info(f"Filtered {len(keys_to_remove)} '_extra_state' keys to prevent server aggregation corruption.")

    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            state_dict[k] = v.to(dtype=torch.float32)

    return state_dict


def prepare_output_dir(
    output_dir: str,
    config_json: str,
    esm_nv_py: str,
    state_dict: dict | None = None,
) -> None:
    """Create a HuggingFace-compatible model directory with config + custom code.

    This directory will be used as ``output_dir`` so that
    ``AutoConfig.from_pretrained(output_dir)`` can load the NVEsm architecture.
    When ``state_dict`` is provided, it is also saved as ``model.safetensors``
    so that ``AutoModelForMaskedLM.from_pretrained(output_dir)`` can
    initialise the model with federated weights.

    Args:
        output_dir: Directory to create / populate.
        config_json: Contents of ``config.json``.
        esm_nv_py: Contents of ``esm_nv.py``.
        state_dict: Optional model state_dict to save as ``model.safetensors``.
    """
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        f.write(config_json)

    esm_nv_path = os.path.join(output_dir, "esm_nv.py")
    with open(esm_nv_path, "w") as f:
        f.write(esm_nv_py)

    if state_dict is not None:
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        safe_save(state_dict, safetensors_path)
        logger.info(
            "Saved model weights to %s (%d keys)", safetensors_path, len(state_dict)
        )

    logger.info("Prepared output directory at %s", output_dir)


def launch_dummy_training(
    *,
    global_weights_path: str,
    output_dir: str,
) -> str:
    """Launch a local dummy script that saves out a safetensors file for testing."""
    os.makedirs(output_dir, exist_ok=True)
    final_model_path = os.path.join(output_dir, "dummy_model.safetensors")

    script_path = os.path.join(os.path.dirname(__file__), "dummy_training.py")

    cmd = [
        "python3", script_path,
        "--resume_from", global_weights_path,
        "--output_path", final_model_path,
    ]
    logger.info("Launching local dummy training:\n  %s", " ".join(cmd))

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info("Dummy stdout:\n%s", result.stdout[-500:] if result.stdout else "")
    if result.stderr:
        logger.warning("Dummy stderr:\n%s", result.stderr[-500:])

    return final_model_path


def launch_cifar_training(
    *,
    output_dir: str,
    resume_path: str,
    epochs: int = 3,
    lr: float = 0.001,
    batch_size: int = 128,
    max_batches_per_epoch: int = 32,
    data_root: str | None = None,
    save_freq: int = 5,
) -> str:
    """Submit cifar10/slurm_train.sh via sbatch and wait for completion.

    Returns the path to the final checkpoint.pt produced by the run.
    """
    slurm_script_src = os.path.join(
        os.path.dirname(__file__), "..", "cifar10", "slurm_train_h100.sh"
    )
    slurm_script_src = os.path.abspath(slurm_script_src)

    if not os.path.isfile(slurm_script_src):
        raise FileNotFoundError(f"SLURM script not found at {slurm_script_src}")

    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="cifar_sbatch_")
    tmp_script = os.path.join(tmp_dir, "slurm_train.sh")
    shutil.copy2(slurm_script_src, tmp_script)

    try:
        env = os.environ.copy()
        # Prevent Flower/client python environment from leaking into the SLURM job
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        
        export_vars = [
            "ALL",
            f"OUTPUT_DIR={output_dir}",
            f"EPOCHS={epochs}",
            f"LR={lr}",
            f"BATCH_SIZE={batch_size}",
            f"MAX_BATCHES={max_batches_per_epoch}",
            f"SAVE_FREQ={save_freq}"
        ]
        if data_root:
            export_vars.append(f"DATA_ROOT={data_root}")
        if resume_path:
            export_vars.append(f"RESUME_PATH={resume_path}")
            
        export_str = ",".join(export_vars)

        result = subprocess.run(
            [
                "sbatch", 
                "--chdir", output_dir, 
                "--output", os.path.join(output_dir, "slurm-%j.out"),
                f"--export={export_str}", 
                tmp_script
            ],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            logger.error("sbatch failed:\n%s", result.stderr)
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split()[-1]
        logger.info("Submitted CIFAR training job: %s", job_id)
        logger.info("Output for job %s will be saved to: %s", job_id, os.path.join(output_dir, f"slurm-{job_id}.out"))

        while True:
            sq = subprocess.run(
                ["squeue", "-j", job_id, "--noheader"],
                capture_output=True, text=True,
            )
            if sq.stdout.strip():
                time.sleep(10)
                continue
            break

        logger.info("CIFAR training job %s completed.", job_id)

        log_file = os.path.join(output_dir, f"slurm-{job_id}.out")
        if os.path.isfile(log_file):
            with open(log_file, "r") as f:
                tail = f.read()[-2000:]
            logger.info("SLURM output (last 2000 chars):\n%s", tail)
            if "Traceback" in tail or "Error" in tail:
                logger.warning("Potential error detected in SLURM output log.")

        final_ckpt_path = os.path.join(output_dir, "checkpoint.pt")
        if not os.path.exists(final_ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at {final_ckpt_path} after training."
            )

        return final_ckpt_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
