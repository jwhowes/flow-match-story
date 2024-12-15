import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src import accelerator
from src.config import Config
from src.model import FlowMatchTransformer
from src.data import StoryDataset


def train(
        model: FlowMatchTransformer,
        dataloader: DataLoader,
        config: Config
):
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    model.train()
    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1} / {config.num_epochs}")
        for i, (tokens, clean_mask) in enumerate(dataloader):
            opt.zero_grad()

            loss = model(tokens, clean_mask)
            accelerator.backward(loss)

            opt.step()
            lr_scheduler.step()

            if i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

        config.save_checkpoint(model, epoch)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config)

    dataset = StoryDataset(
        tokenizer_path=config.dataset.tokenizer_path,
        split="train",
        max_length=config.dataset.max_length,
        p_uncond=config.dataset.p_uncond,
        p_autoreg=config.dataset.p_autoreg
    )

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

    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    config.save()

    train(model, dataloader, config)
