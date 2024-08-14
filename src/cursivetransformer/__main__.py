import os
import json
import sys
import time

import torch
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from .configs import AppConfig, ModelConfig
from .data import create_datasets, InfiniteDataLoader
from .model import Transformer
from .utils import evaluate, parse_args, save_samples, setup_logger

logger = setup_logger()

def main(config: AppConfig):
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if accelerator.is_main_process:
        os.makedirs(config.work_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(
        raw_data_path=config.raw_data_path,
        augment=config.augment,
        max_seq_length=config.max_seq_length,
        num_words=config.num_words,
    )
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_stroke_seq_length()
    context_block_size = train_dataset.get_text_seq_length()
    context_vocab_size = train_dataset.get_char_vocab_size()
    
    if accelerator.is_main_process:
        print(f"Dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        context_block_size=context_block_size,
        context_vocab_size=context_vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_embd2=config.n_embd2,
        ablate_cross_attention=config.ablate_cross_attention,
        n_ctx_head=config.n_head,
    )
    model = Transformer(model_config)
    
    if accelerator.is_main_process:
        print(f"Model #params: {sum(p.numel() for p in model.parameters())}")
    
    if config.resume or config.sample_only:
        model.load_state_dict(torch.load(os.path.join(config.work_dir, "model.pt"), map_location=device))
    
    if config.sample_only:
        print("Sample only mode is not implemented")
        return

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    scheduler = None
    if config.lr_scheduler == "steplr":
        scheduler = StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_decay
        )

    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Prepare everything with accelerator
    model, optimizer, batch_loader = accelerator.prepare(model, optimizer, batch_loader)

    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=config,
        )

    # training loop
    best_loss = float('inf')
    step = 0
    while True:
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, C, Y = batch

        with accelerator.accumulate(model):
            logits, loss = model(X, C, Y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        if accelerator.is_main_process:
            wandb.log({"train_loss_step": loss.item(), "step": step})

        t1 = time.time()

        if accelerator.is_main_process and step % 100 == 0:
            print(
                f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | lr {scheduler.get_last_lr()[0] if scheduler else config.learning_rate:.6f}"
            )

        if step > 0 and step % 1000 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10, device=device)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10, device=device)
            
            if accelerator.is_main_process:
                wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step})
                print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")
                
                if test_loss < best_loss:
                    out_path = os.path.join(config.work_dir, "model.pt")
                    print(f"Test loss {test_loss:.4f} is the best so far, saving model to {out_path}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), out_path)
                    best_loss = test_loss

        if accelerator.is_main_process and step > 0 and step % 1000 == 0:
            save_samples(model, test_dataset, num=6, do_sample=True, device=device)
            save_samples(model, test_dataset, num=6, do_sample=False, device=device)
            save_samples(model, train_dataset, num=3, do_sample=True, device=device)
            save_samples(model, train_dataset, num=3, do_sample=False, device=device)

        step += 1
        if config.max_steps >= 0 and step >= config.max_steps:
            break

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    config = AppConfig(**config)
    main(config)