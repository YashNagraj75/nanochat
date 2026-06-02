import os
import json
import torch
from datetime import datetime
import torch.distributed as dist

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_base_dir():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def save_training_metadata(
    output_dir,
    metadata,
    markdown_filename="metadata.md",
):
    """
    Save training metadata to JSON, tensor, and markdown formats.

    Args:
        output_dir (str): Directory where metadata files will be saved
        metadata (dict): Dictionary containing all metadata. Can include:
            - timestamp (str, optional): ISO format timestamp
            - train_time_seconds (float, optional): Training time
            - vocab_size (int, optional): Vocabulary size
            - num_special_tokens (int, optional): Number of special tokens
            - args (dict, optional): Arguments passed to the script
            - metrics (dict, optional): Training metrics
            - Any other custom fields
        markdown_filename (str): Name of the markdown file to save (default: "metadata.md")

    Returns:
        dict: The metadata dictionary that was saved
    """
    # Add timestamp if not present
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.now().isoformat()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    metadata_json_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON to {metadata_json_path}")

    # Save as PyTorch tensor (only numeric values for the tensor)
    numeric_values = []
    numeric_keys = []
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
            numeric_keys.append(key)

    if numeric_values:
        metadata_tensor = torch.tensor(
            numeric_values, dtype=torch.float32, device="cpu"
        )
        metadata_tensor_path = os.path.join(output_dir, "metadata.pt")
        torch.save(metadata_tensor, metadata_tensor_path)
        print(f"Saved metadata tensor to {metadata_tensor_path}")

    # Save as markdown file
    markdown_path = os.path.join(output_dir, markdown_filename)
    with open(markdown_path, "w") as f:
        f.write("# Training Metadata\n\n")

        if "timestamp" in metadata:
            f.write(f"**Generated**: {metadata['timestamp']}\n\n")

        # Organize sections based on common keys
        if any(
            k in metadata
            for k in ["train_time_seconds", "vocab_size", "num_special_tokens"]
        ):
            f.write("## Training Details\n\n")
            if "train_time_seconds" in metadata:
                f.write(
                    f"- **Training Time**: {metadata['train_time_seconds']:.2f} seconds\n"
                )
            if "vocab_size" in metadata:
                f.write(f"- **Vocabulary Size**: {metadata['vocab_size']:,}\n")
            if "num_special_tokens" in metadata:
                f.write(f"- **Special Tokens**: {metadata['num_special_tokens']}\n")
            if "vocab_size" in metadata and "num_special_tokens" in metadata:
                f.write(
                    f"- **Regular Tokens**: {metadata['vocab_size'] - metadata['num_special_tokens']:,}\n"
                )
            f.write("\n")

        if "args" in metadata and metadata["args"]:
            f.write("## Training Arguments\n\n")
            for key, value in metadata["args"].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")

        if "metrics" in metadata and metadata["metrics"]:
            f.write("## Training Metrics\n\n")
            for key, value in metadata["metrics"].items():
                if isinstance(value, float):
                    f.write(f"- **{key}**: {value:.6f}\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")

        # Add any remaining custom fields
        custom_fields = {
            k: v
            for k, v in metadata.items()
            if k
            not in [
                "timestamp",
                "train_time_seconds",
                "vocab_size",
                "num_special_tokens",
                "args",
                "metrics",
            ]
        }
        if custom_fields:
            f.write("## Additional Information\n\n")
            for key, value in custom_fields.items():
                if isinstance(value, dict):
                    f.write(f"- **{key}**:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  - {subkey}: {subvalue}\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")

    print(f"Saved metadata markdown to {markdown_path}")

    return metadata


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_COMPUTE_DTYPE")
    if env is not None:
        env = env.lower()
        if env in _DTYPE_MAP:
            return env
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability >= (8, 0):
                return (
                    torch.bfloat16,
                    f"Auto-detected: CUDA SM: {capability[0]}{capability[1]} (bf16 supported)",
                )
            return (
                torch.float32,
                f"Auto-detected: CUDA {capability[0]}{capability[1]}, (bfloat16 not supported)",
            )
        return (
            torch.float32,
            f"Auto-detected (CPU/MPS): {capability[0]}{capability[1]}",
        )


COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()
print(COMPUTE_DTYPE, COMPUTE_DTYPE_REASON)


def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
