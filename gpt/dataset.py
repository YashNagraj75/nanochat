import os
from time import time
import requests
from multiprocessing import Pool
import pyarrow.parquet as pq
from gpt.common import get_base_dir
import argparse

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
index_to_filename = lambda index: (
    f"shard_{index:05d}.parquet"
)  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")


def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir

    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    return parquet_files


def parquet_iter_batched(
    split, start=0, step=1
):  # start and step are good skip parameters, when we need to skip to world size in DDP
    assert split in ["train", "val"], "Split should be either 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for file in parquet_paths:
        pf = pq.ParquetFile(file)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# Imporrtant function for downloading a single file from hf, honestly could use streaming dataset
def download_single_file(index):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename}, already exists.")
        return True

    url = f"{BASE_URL}/{filename}"
    print("Downloading:", url)

    # Download with retires
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Write to temp file first
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.rename(temp_path, filepath)
            print(f"Downloaded the file: {filepath}")
            return True

        except (requests.RequestException, IOError) as e:
            print(
                f"Attempting to download the file: {filepath} for {attempt}/{max_attempts} because of {e}"
            )
            for path in [filepath, temp_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        pass

            if attempt < max_attempts - 1:
                wait_time = 2**attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(
                    f"Failed to download the file: {filepath} after {max_attempts} attempts."
                )
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of train shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = (
        MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    )
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard

    # Download the shards
    print(
        f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers..."
    )
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")

