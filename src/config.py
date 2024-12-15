import yaml
import os
import torch

from typing import Optional, Dict
from . import accelerator
from transformers import AutoTokenizer


class SubConfig:
    def __init__(self, config: Optional[Dict] = None):
        if config is not None:
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)


class DatasetConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.batch_size = 64
        self.tokenizer_path = "google-bert/bert-base-uncased"
        self.max_length = 512

        self.p_uncond = 0.1
        self.p_autoreg = 0.7

        super().__init__(config)

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id


class ModelConfig(SubConfig):
    def __init__(self, config: Optional[Dict] = None):
        self.d_model = 768
        self.d_t = 768
        self.n_layers = 12
        self.n_heads = 12

        self.attn_dropout = 0.0
        self.ffn_dropout = 0.0

        self.sigma_min = 1e-4

        super().__init__(config)

        self.sigma_min = float(self.sigma_min)


class Config(SubConfig):
    def __init__(self, config_path: str):
        self.lr = 3e-4
        self.weight_decay = 0.01

        self.num_warmup_steps = 500
        self.num_epochs = 1

        self.log_interval = 100

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super().__init__(config)
        self.lr = float(self.lr)

        self.dataset = DatasetConfig(config["dataset"] if "dataset" in config else None)
        self.model = ModelConfig(config["model"] if "model" in config else None)

        self.exp_name = os.path.splitext(os.path.basename(config_path))[0]
        self.save_dir = os.path.join("experiments", self.exp_name)

    def save(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        with open(os.path.join(self.save_dir, "config.yaml"), "w+") as f:
            yaml.dump(self, f)

    def save_checkpoint(self, model, epoch):
        torch.save(
            accelerator.get_state_dict(model),
            os.path.join(self.save_dir, f"checkpoint_{epoch:02}.pt")
        )
