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
from torch.optim.lr_scheduler import StepLR

from model import Transformer
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

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    context_block_size: int = None
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    context_vocab_size: int = None # size of the context vocabulary (ASCII characters)
    context_length: int = None # maximum length of the context sequence
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    n_ctx_head: int = 4 # number of heads for cross-attention
    ablate_cross_attention: bool = False


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
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads in Transformer block')

    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--train_size', type=int, default=497000, help='Number of train examples')
    parser.add_argument('--test_size', type=int, default=3000, help='Number of test examples')
    parser.add_argument('--num_words', type=int, default=4, help='Number of words')
    parser.add_argument('--max_seq_length', type=int, default=1000, help='Maximum sequence length (tokens)')
    parser.add_argument('--augment', action='store_true', default=True, help='Perform augmentations')
    parser.add_argument('--ablate_cross_attention', action='store_true', default=False, help='Ablate the cross attention')
    parser.add_argument('--add_digits', action='store_true', default=True, help='Add digit words to the word bank')
    parser.add_argument('--alphabet', type=str, default=" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ",
                            help='All the characters that this model will be able to draw')
    parser.add_argument('--dataset_name', type=str, default='zachbigbank', help='Set this to your wandb username or team name')

    parser.add_argument('--wandb_project', type=str, default='cursivetransformer', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='zwimpee', help='Set this to your wandb username or team name')
    parser.add_argument('--wandb_run_name', type=str, default='test_20240911_zachbigbank_v1', help='W&B run name')
    parser.add_argument('--wandb_api_key', type=str, default='e56bbe426145e5983e72a933299daca195ebb6a7', help='Weights & Biases API Key')

    parser.add_argument('--resume_from_run_id', type=str, default=None, help='Resume from a specific W&B run ID')
    parser.add_argument('--sample_only', action='store_true', default=False, help='Only sample from the model')
    parser.add_argument('--local_model_path', type=str, default='best_model.pt', help='Path to local model file')

    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not args.sample_only:
        wandb_init_args = {"project": args.wandb_project, "entity": args.wandb_entity, "config": args}
        if args.resume_from_run_id:
            wandb_init_args["id"] = args.resume_from_run_id
            wandb_init_args["resume"] = "must"
        else:
            wandb_init_args["name"] = args.wandb_run_name
        wandb.init(**wandb_init_args)

    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    # init datasets
    train_dataset, test_dataset = create_datasets(args)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_stroke_seq_length()
    context_block_size = train_dataset.get_text_seq_length()
    context_vocab_size = train_dataset.get_char_vocab_size()
    print(f"Dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size, context_block_size=context_block_size,
                         context_vocab_size=context_vocab_size, n_layer=args.n_layer, n_head=args.n_head,
                         n_embd=args.n_embd, n_embd2=args.n_embd2, ablate_cross_attention=args.ablate_cross_attention,
                         n_ctx_head=args.n_head)
    model = Transformer(config)
    model.to(args.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume_from_run_id or args.sample_only:
        if os.path.exists(args.local_model_path):
            model.load_state_dict(torch.load(args.local_model_path, weights_only=True))
            print(f"Loaded model from {args.local_model_path}")
        else:
            print("Downloading model from W&B")
            api = wandb.Api()
            artifact = api.artifact(f'{args.wandb_entity}/{args.wandb_project}/{args.resume_from_run_id or args.wandb_run_name}:model:latest')
            model_dir = artifact.download()
            model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", weights_only=True))
            torch.save(model.state_dict(), args.local_model_path)
    if args.sample_only:
        save_samples(model, test_dataset, num=6, do_sample=True)
        save_samples(model, test_dataset, num=6, do_sample=False)
        sys.exit()

    # init optimizer and batch loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=args.step_lr_every, gamma=args.lr_decay)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    wandb.config.update({
        "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "ablate_cross_attention": args.ablate_cross_attention,
    })

    # model saving stuff
    wandb.watch(model, log="all", log_freq=args.log_every, log_graph=False)



    ########## ARGS, LOGGING, AND TRAIN LOOP ##########

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, C, Y = batch
        
        # Add this sanity check
        if X.max() >= vocab_size or Y.max() >= vocab_size:
            print(f"Warning: Token indices out of range. X max: {X.max()}, Y max: {Y.max()}, Vocab size: {vocab_size}")
            X = torch.clamp(X, 0, vocab_size - 1)
            Y = torch.clamp(Y, 0, vocab_size - 1)

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
                print(f"Test loss {test_loss:.4f} is the best so far, saving model to {args.local_model_path}")
                torch.save(model.state_dict(), args.local_model_path)
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(args.local_model_path)
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