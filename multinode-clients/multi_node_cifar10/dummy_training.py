"""dummy_training.py: A local dummy script simulating an ESM2 training run.

Loads the provided PyTorch global model, simulates processing time or
manipulation, and saves the result as a Safetensors file.

When ``--resume_from`` is omitted or empty the script starts with a single
placeholder tensor so that the Flower server receives a non-empty ArrayRecord
and federated averaging can proceed normally.  The real initialisation happens
inside the CIFAR training scripts on the client nodes.
"""

import argparse
import logging
import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from",
        default="",
        help="Input global weights path (optional; omit for random-init round 1)",
    )
    parser.add_argument("--output_path", required=True, help="Output safetensors path")
    args = parser.parse_args()

    if args.resume_from:
        logger.info("Dummy Run: Loading initial weights from %s", args.resume_from)
        if args.resume_from.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.resume_from, device="cpu")
        else:
            state_dict = torch.load(
                args.resume_from, map_location="cpu", weights_only=True
            )
        # Convert state_dict values to pure tensors without grad attached just in
        # case, and mimic a minor training pass (left unchanged to simulate
        # "safetensors mapping").
        for k, v in dict(state_dict).items():
            state_dict[k] = v.contiguous()
    else:
        logger.info(
            "Dummy Run: No checkpoint provided – using placeholder tensor for round 1."
        )
        # Provide a minimal non-empty state_dict so ArrayRecord is valid.
        state_dict = {"_placeholder": torch.zeros(1)}

    logger.info("Dummy Run: Saving updated weights as safetensors to %s", args.output_path)
    save_file(state_dict, args.output_path)


if __name__ == "__main__":
    main()
