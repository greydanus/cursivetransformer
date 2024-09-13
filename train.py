########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, argparse, getpass
from typing import Optional
from dataclasses import dataclass
from types import SimpleNamespace

import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_checkpoint, save_checkpoint, get_all_args
from sample import save_samples
from data import InfiniteDataLoader, create_datasets


@torch.inference_mode()
def evaluate(model, dataset, batch_size=15, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, C, Y = batch
        logits, loss = model(X, C, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


########## ARGS, LOGGING, AND TRAIN LOOP ##########



if __name__ == '__main__':

    args = get_all_args()

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not args.sample_only:
        wandb_init_args = {"project": args.wandb_project, "entity": args.wandb_entity, "config": args}
        if args.load_from_run_id:
            wandb_init_args["id"] = args.load_from_run_id
            wandb_init_args["resume"] = "must"
        else:
            wandb_init_args["name"] = args.wandb_run_name
        wandb.init(**wandb_init_args)

    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    train_dataset, test_dataset = create_datasets(args)  # init datasets
    args.vocab_size = train_dataset.get_vocab_size()
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.context_vocab_size = train_dataset.get_char_vocab_size()
    print(f"Dataset determined that: {args.vocab_size=}, {args.block_size=}")

    model, optimizer, scheduler, step, best_loss = get_checkpoint(args)

    if args.sample_only:
        save_samples(model, test_dataset, num=6, do_sample=True)
        save_samples(model, test_dataset, num=6, do_sample=False)
        sys.exit()

    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    wandb.watch(model, log="all", log_freq=args.log_every, log_graph=False)  # model saving stuff


    ########## ARGS, LOGGING, AND TRAIN LOOP ##########

    # training loop
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, C, Y = batch

        # feed into the model
        logits, loss = model(X, C, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True) ; loss.backward()
        optimizer.step() ; scheduler.step()
        wandb.log({"train_loss_step": loss.item(), "step": step})

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % args.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | lr {scheduler.get_last_lr()[0]:.6f}")

        # evaluate the model
        if step > 0 and step % args.log_every == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step })
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")

            if best_loss is None or test_loss < best_loss:  # save the model to W&B if it has improved
                best_loss = test_loss
                print(f"Test loss {test_loss:.4f} is the best so far, saving checkpoint to {args.local_checkpoint_path}")
                save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss)
                artifact = wandb.Artifact('best_checkpoint', type='model')
                artifact.add_file(args.local_checkpoint_path)
                wandb.log_artifact(artifact)


        # sample from the model
        if step > 0 and step % args.log_every == 0:
            save_samples(model, test_dataset, num=6, do_sample=True)
            save_samples(model, test_dataset, num=6, do_sample=False)
            save_samples(model, train_dataset, num=3, do_sample=True)
            save_samples(model, train_dataset, num=3, do_sample=False)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    wandb.finish()

