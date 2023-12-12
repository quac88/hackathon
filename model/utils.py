import torch
from transformers import AutoTokenizer

class TokenizerWrapper(object):
    def __init__(self, model_name: str, seq_len: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_len = seq_len

        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]

    def __call__(self, text):
        return self._tokenize(text)

# Usage
tokenizer_wrapper = TokenizerWrapper("gpt2", 512)  # Example: using GPT-2 tokenizer
