import os
import sys
import argparse
from time import time
import torch

# Add parent directory to path so gpt module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.dataset import parquet_iter_batched
from gpt.common import get_base_dir, save_training_metadata
from gpt.tokenizer import RustBPE_Tokenizer


parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max-chars",
    type=int,
    default=2_000_000_000,
    help="Maximum characters to train on (default: 2B)",
)
parser.add_argument(
    "--doc-cap",
    type=int,
    default=10_000,
    help="Maximum characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Vocabulary size (default: 32768 = 2^15)",
)
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")


def text_iter():
    nchars = 0
    for batch in parquet_iter_batched(split="train"):
        for doc in batch:
            if len(doc) > args.doc_cap:
                doc_text = doc[: args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars >= args.max_chars:
                return


text_iterator = text_iter()

t0 = time.time()
tokenizer = RustBPE_Tokenizer.train_from_iterator(
    text_iterator, vocab_size=args.vocab_size
)
t1 = time.time()
print(f"Trained BPE tokenizer in {t1 - t0:.2f} seconds")

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# Simple test
text = "Hello I am just here to test the tokenization"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(text)
assert decoded == text

# I want to calculated bits per byte for thr vocab as it will allow us to measure the loss
# invariant of the tokenizer acrhitecture and vocab size. Then we calculate the bits per byte
# for the validation set which I care about

vocab_size = len(tokenizer.get_vocab())
special_set = set(tokenizer.get_special_tokens())
token_strings = [
    tokenizer.decode(token_id)
    for token_id in range(vocab_size)
    if token_id not in special_set
]

token_bytes = []
for token_id in range(vocab_size):
    token_string = token_strings[token_id]
    ids = len(token_string.encode("utf-8"))
    token_bytes.append(ids)

token_bytes = torch.tensor(token_bytes, dtype=torch.float32, device="cpu")

# Prepare metadata dictionary
metadata = {
    "train_time_seconds": t1 - t0,
    "vocab_size": vocab_size,
    "num_special_tokens": len(special_set),
    "args": vars(args),
}

# Save token_bytes tensor with metadata
token_bytes_data = {
    "tensor": token_bytes,
    "metadata": metadata,
}
torch.save(token_bytes_data, os.path.join(tokenizer_dir, "token_bytes.pt"))
print(
    f"Saved token byte counts with metadata to {os.path.join(tokenizer_dir, 'token_bytes.pt')}"
)

# Save metadata separately using the utility function

# Save training metadata
save_training_metadata(
    output_dir=tokenizer_dir,
    metadata=metadata,
    markdown_filename="tokenizer.md",
)
