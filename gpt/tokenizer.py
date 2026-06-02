lpimport torch
from anyio.functools import lru_cache
import rustbpe
import tiktoken
import os, pickle
import copy

# Special Tokens used in the dataset, copied from karpathy/nanochat
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]

# Again Copy
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# Tokenizer based on rustbpe and tiktoken for inference
class RustBPE_Tokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = (
            vocab_size - len(SPECIAL_TOKENS)
        )  # Getting vacab_size to verify the tokenizer has learnt the encoding which is vocab size length
        assert vocab_size_no_special >= 256, (
            f"Vocab size must be at least 256 to accommodate all byte values. {vocab_size_no_special}"
        )  # So the tokenizer here encodes hexadecimal byte values as tokens, and we need at least 256 tokens to represent all possible byte values (0-255). If the vocab size is less than 256, it won't be able to encode all byte values, which could lead to issues during tokenization.
        tokenizer.train_from_iterator(
            text_iterator, vocab_size=vocab_size_no_special, pattern=SPLIT_PATTERN
        )

        # Get the encodings for inference
        pattern = tokenizer.get_pattern()
        print(f"Tokenizer trained with pattern: {pattern}")
        mergable_ranks_list = tokenizer.get_mergeable_ranks()
        print(f"Tokenizer trained with {len(mergable_ranks_list)} merges.")
        mergable_ranks = {bytes(k): v for k, v in mergable_ranks_list}
        print(mergable_ranks)
        # Adding the special tokens to the tokenizer
        offset = len(mergable_ranks)
        special_tokens = {name: i + offset for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens,
        )  # using tiktoken once our encoder is trained to get ecodings which will be sent to the transformer model for inference
        return cls(enc, bos_token="<|bos|>")

    @classmethod
    def load_from_file(
        cls, tokenizer_dir
    ):  # Function to load the tokenizer from a file,
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, bos_token="<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(
            enc, bos_token="<|endoftext|>"
        )  # This is industry standard, so most tokenizers use <|endoftext|> as the special token to denote the end of a sequence.

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token

    # Encode function to convert text into token ids, for both single string and list of strings, usefull for batch processing
    def encode(self, text, prepend=None, append=None):
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            token_ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                token_ids.insert(0, prepend_id)
            if append is not None:
                token_ids.append(append_id)
        if isinstance(text, list):
            token_ids = self.enc.encode_ordinary_batch(text)
            if prepend is not None:
                token_ids = [
                    [prepend_id] + ids for ids in token_ids
                ]  # I think its slow ?
            if append is not None:
                token_ids = [ids + [append_id] for ids in token_ids]
        return token_ids

    def __call__(self, *args, **kwds):
        return self.encode(*args, **kwds)

    def decode(self, token_ids):
        return self.enc.decode(token_ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Tokenizer saved at {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize the converstion which is called as doc or document.
        Returns ids and mask.
        So as the model has no memory of the chat history we will have to tokenize the full chat until that point everytime we
        send a prompt to the model, so we will have to render the full conversation into token ids and mask everytime.
        """
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Sometimes first convo is system prompt, just merge it to the first user message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(
                conversation
            )  # To avoid modifying the original conversation
            messages = conversation["messages"]
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = conversation["messages"][1:]
        else:
            messages = conversation["messages"]

        # Now getting all the special tokens needed to insert into the convo
        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<|user_start|>"),
            self.encode_special("<|user_end|>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = (
            self.encode_special("<|python_start|>"),
            self.encode_special("<|python_end|>"),
        )
        output_start, output_end = (
            self.encode_special("<|output_start|>"),
            self.encode_special("<|output_end|>"),
        )

        add_tokens(bos, 0)

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            must_be_from = "user" if i % 2 == 0 else "assistant"
            print(f"The message must be from {must_be_from}, but got {role}")

            if role == "user":
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)

            elif role == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # String parts just add them
                            add_tokens(value_ids, 1)
                        if part["type"] == "python":
                            # Python tool call -> add the tokens between <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)

                        if part["type"] == "python_output":
                            # python output add it between <|output_start|> <|output_end|> but its unsupervised as the output is coming from python, no reasoning will be done on the output
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)

                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(
                        f"Unknown content type for assistant message: {type(content)}"
                    )
                add_tokens(assistant_end, 0)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


def get_tokenizer(base_dir):
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    return RustBPE_Tokenizer.load_from_file(tokenizer_path)


def get_tokenizer_bytes(device="cpu", base_dir=None):
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
