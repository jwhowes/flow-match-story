import torch
import numpy as np
import torch.nn.functional as F

from datasets import load_dataset
from random import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class StoryDataset(Dataset):
    def __init__(
            self, tokenizer_path="google-bert/bert-base-uncased", split="train", max_length=512,
            p_uncond=0.1, p_autoreg=0.7
    ):
        assert p_uncond + p_autoreg <= 1.0
        self.p_uncond = p_uncond
        self.p_autoreg = p_autoreg
        self.p_random = 1.0 - (p_uncond + p_autoreg)

        self.ds = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.tokenizer.model_max_length = max_length

        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length
        self.pad_token = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.ds[idx]["text"], truncation=True, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        L = tokens.shape[0]

        r = random()
        clean_mask = torch.zeros(L)
        if r >= self.p_uncond:
            r -= self.p_uncond
            num_clean = np.random.randint(0, L)
            if r < self.p_autoreg:
                clean_mask[:num_clean] = 1.0
            else:
                clean_mask[torch.randperm(L)[:num_clean]] = 1.0

        return (
            F.pad(tokens, (0, self.max_length - L), value=self.pad_token),
            F.pad(clean_mask, (0, self.max_length - L), value=0.0)
        )
