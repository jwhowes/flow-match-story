import torch
import os

from argparse import ArgumentParser
from transformers import AutoTokenizer

from src.config import Config
from src.model import FlowMatchTransformer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file", type=str)
    parser.add_argument("out_file", type=str)

    parser.add_argument("--config", type=str, default="configs/story-writer.yaml")
    parser.add_argument("--checkpoint", type=int, default=1)

    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--step", type=str, default="euler")

    args = parser.parse_args()

    config = Config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)

    model = FlowMatchTransformer(
        length=config.dataset.max_length,
        pad_token=config.dataset.pad_token,
        vocab_size=config.dataset.vocab_size,
        d_model=config.model.d_model,
        d_t=config.model.d_t,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        attn_dropout=config.model.attn_dropout,
        ffn_dropout=config.model.ffn_dropout,
        sigma_min=config.model.sigma_min
    )
    ckpt = torch.load(
        os.path.join(config.save_dir, f"checkpoint_{args.checkpoint:02}.pt"),
        weights_only=True, map_location="cpu"
    )
    model.load_state_dict(ckpt)
    del ckpt

    with open(args.in_file, "r") as f:
        prompt = f.read()

    tokens = tokenizer(prompt, truncation=True, return_tensors="pt")["input_ids"][0, :-1]

    pred_tokens = model.sample(
        tokens, num_steps=args.num_steps, step=args.step, guidance_scale=args.guidance_scale
    )
    pred_text = tokenizer.decode(pred_tokens[1:])

    with open(args.out_file, "w+") as f:
        f.write(pred_text)
