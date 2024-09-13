########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, argparse, getpass
from typing import Optional
from dataclasses import dataclass
import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_checkpoint
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

    parser = argparse.ArgumentParser(description='Train a cursivetransformer model')
    parser.add_argument('--max_steps', type=int, default=110000, help='How many steps to train for')
    parser.add_argument('--print_every', type=int, default=100, help='Print log info after how many steps')
    parser.add_argument('--log_every', type=int, default=2500, help='Sample model after how many steps')
    parser.add_argument('--lr_decay', type=float, default=0.333, help='How much to decay the learning rate')
    parser.add_argument('--step_lr_every', type=int, default=33000, help='How often to decay the learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='This is meant to be trained on a GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--n_layer', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--n_embd', type=int, default=64, help='Number of embedding dimensions in self attention')
    parser.add_argument('--n_embd2', type=int, default=64, help='Number of embedding dimensions in cross attention')
    parser.add_argument('--n_ctx_head', type=int, default=4, help='Number of attention heads in Transformer block')

    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--train_size', type=int, default=497000, help='Number of train examples')
    parser.add_argument('--test_size', type=int, default=3000, help='Number of test examples')
    parser.add_argument('--num_words', type=int, default=4, help='Number of words')
    parser.add_argument('--max_seq_length', type=int, default=1000, help='Maximum sequence length (tokens)')
    parser.add_argument('--augment', action='store_true', default=True, help='Perform augmentations')
    parser.add_argument('--ablate_cross_attention', action='store_true', default=False, help='Ablate the cross attention')
    parser.add_argument('--downsample_mean', type=float, default=0.65, help='Mean amount to downsample stroke points (0.65=65%)')
    parser.add_argument('--downsample_width', type=float, default=0.1, help='Width of the uniform distribution (0.1=10%)')
    parser.add_argument('--add_digits', action='store_true', default=True, help='Add digit words to the word bank')
    parser.add_argument('--alphabet', type=str, default=" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ",
                            help='All the characters that this model will be able to draw')
    parser.add_argument('--dataset_name', type=str, default='bigbank', help='Set this to your wandb username or team name')

    parser.add_argument('--wandb_project', type=str, default='synthbank_experiments', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='sam-greydanus', help='Set this to your wandb username or team name')
    parser.add_argument('--wandb_run_name', type=str, default='unnamed_run', help='W&B run name')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='Weights & Biases API Key')

    parser.add_argument('--load_from_run_id', type=str, default=None, help='Load from a specific W&B run ID')
    parser.add_argument('--sample_only', action='store_true', default=False, help='Only sample from the model')
    parser.add_argument('--local_checkpoint_path', type=str, default='best_checkpoint.pt', help='Path to local model file')

    args = parser.parse_args()

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

