import os
import sys
import argparse
from time import time

# Add parent directory to path so gpt module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt.dataset import parquet_iter_batched
from gpt.common import get_base_dir
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

# I want to
