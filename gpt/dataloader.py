import torch
import pyarrow.parquet as pq
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt.common import get_dist_info
from gpt.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Iterator for text from parquet files.

    Handles DDP sharding  and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, (
        "No parquet files found in data directory, did you run dataset.py ?"
    )
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = (
        resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    )
    resume_epoch = resume_state_dict["epoch"] if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file else on ddp rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = resume_rg_idx + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)

                rg_idx = ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1
