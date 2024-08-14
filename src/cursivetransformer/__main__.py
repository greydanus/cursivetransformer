import os
import sys
import time

import torch
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from .config import AppConfig, ModelConfig
from .data import InfiniteDataLoader, create_datasets
from .model import Transformer
from .utils import evaluate, parse_args, save_samples, setup_logger

logger = setup_logger()


def main(config: AppConfig):

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.makedirs(config.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(
        augment=config.augment,
        max_seq_length=config.max_seq_length,
        num_words=config.num_words,
    )
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_stroke_seq_length()
    context_block_size = train_dataset.get_text_seq_length()
    context_vocab_size = train_dataset.get_char_vocab_size()
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
    model.to(config.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")
    if (
        config.resume or config.sample_only
    ):  # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(config.work_dir, "model.pt")))
    if config.sample_only:
        # save_samples(num=50)
        print("This functionality is temporarily commented out")
        sys.exit()

    # init optimizer and batch loader
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")

    scheduler = None
    if config.lr_scheduler == "steplr":
        scheduler = StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_decay
        )

    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=args,
    )

    wandb.config.update(
        {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "ablate_cross_attention": args.ablate_cross_attention,
        }
    )

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, C, Y = batch

        # feed into the model
        logits, loss = model(X, C, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        wandb.log({"train_loss_step": loss.item(), "step": step})

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 100 == 0:
            print(
                f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | lr {scheduler.get_last_lr()[0]:.6f}"
            )

        # evaluate the model
        if step > 0 and step % 1000 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step})
            print(
                f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}"
            )
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(
                    f"Test loss {test_loss:.4f} is the best so far, saving model to {out_path}"
                )
                torch.save(model.state_dict(), out_path)
                # wandb.save(out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 1000 == 0:
            save_samples(model, test_dataset, num=6, do_sample=True)
            save_samples(model, test_dataset, num=6, do_sample=False)
            save_samples(model, train_dataset, num=3, do_sample=True)
            save_samples(model, train_dataset, num=3, do_sample=False)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = AppConfig(**vars(args))
    main(config)
