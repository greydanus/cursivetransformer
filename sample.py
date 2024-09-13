########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, getpass
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Transformer, get_checkpoint, get_all_args
from data import create_datasets, offsets_to_strokes


def plot_strokes(stroke, title, fig=None, ax=None):
    """Plot a single stroke"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 2), dpi=150)

    # Separate strokes based on pen lifts
    strokes = []
    current_stroke = []
    for point in stroke:
        if point[2] == 1:  # Pen is down
            current_stroke.append(point)
        else:  # Pen is up
            if current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
    if current_stroke:
        strokes.append(current_stroke)

    # Plot each stroke
    for stroke in strokes:
        x, y = zip(*[(p[0], 1 - p[1]) for p in stroke])  # Invert y-axis
        ax.plot(x, y, 'b-', linewidth=1.3)

    ax.set_aspect('equal') ; ax.set_title(title)
    if fig is None: plt.show()
    return fig, ax


@torch.no_grad()
def generate(model, idx, context, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    steps = max(0, max_new_tokens-idx.size(1))
    for i in range(steps):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond, context)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def save_samples(model, dataset, num=2, model_device='cpu', warmup_steps=100, do_sample=False, log_wandb=True):
    """ samples from the model and plots the decoded strokes """
    model_device = next(model.parameters()).device

    stroke_seq, context = [], []
    for i in range(num):
      x, c, y = dataset[i]
      stroke_seq.append(x) ; context.append(c)

    X_init = torch.stack(stroke_seq).to(model_device)[:,:warmup_steps]
    context = torch.stack(context).long().to(model_device)
    top_k = None
    steps = dataset.get_stroke_seq_length() - 1  # -1 because we already start with the first token

    X_samp = generate(model, X_init, context, steps, top_k=top_k, do_sample=do_sample).to('cpu')

    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i].detach().cpu().numpy()
        offset_samp = dataset.decode_stroke(row)
        point_samp = offsets_to_strokes(offset_samp)
        decoded_ascii = dataset.decode_text(context[i])

        # Plot the stroke
        fig, ax = plot_strokes(point_samp, f'Sample {i+1}: "{decoded_ascii}"') #plt.axis('off')
        tag = 'sample' if do_sample else 'topk'
        fig.savefig(f"{dataset.name}_{tag}_{i+1}.png")
        if log_wandb:
            wandb.log({f"{dataset.name}_{tag}_{i+1}": wandb.Image(f"{dataset.name}_{tag}_{i+1}.png")})
        plt.close(fig)
        print(f"Saved {dataset.name}_{tag}_{i+1}.png")

    print('-'*80)


########## ARGS, LOGGING, AND TRAIN LOOP ##########


if __name__ == '__main__':

    args = get_all_args()
    args.sample_only = True

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    train_dataset, test_dataset = create_datasets(args)  # init datasets
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.vocab_size = train_dataset.get_vocab_size()
    args.context_vocab_size = train_dataset.get_char_vocab_size()
    print(f"Dataset determined that: {args.vocab_size=}, {args.block_size=}")

    model, optimizer, scheduler, step, best_loss = get_checkpoint(args)

    save_samples(model, test_dataset, num=6, do_sample=True, log_wandb=False)
    save_samples(model, test_dataset, num=6, do_sample=False, log_wandb=False)
    sys.exit()

