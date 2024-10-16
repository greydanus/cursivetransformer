# Sam Greydanus | 2024

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
def generate(model, idx, context, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, is_hooked=False):
    print(f"Starting generation with max_new_tokens: {max_new_tokens}, is_hooked: {is_hooked}")
    if is_hooked:
        block_size = model.cfg.n_ctx
    else:
        block_size = model.cfg.block_size 
    print(f"Block size: {block_size}")
    steps = max(0, max_new_tokens-idx.size(1))
    print(f"Generating for {steps} steps")
    attention_patterns = []
    
    for i in range(steps):
        if i % 100 == 0:
            print(f"Generation step: {i}/{steps}")
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        try:
            if is_hooked:
                logits, new_cache = model.run_with_cache(
                    idx_cond, 
                    context, 
                    return_type='logits',
                    names_filter=lambda name: name.endswith('.hook_pattern')
                )
                # Store only the attention patterns
                attention_patterns.append({k: v.detach().cpu() for k, v in new_cache.items()})
            else:
                logits, _ = model(idx_cond, context)
        except RuntimeError as e:
            print(f"Error at step {i}: {e}")
            return idx, attention_patterns

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)

    print("Generation completed successfully")
    return idx, attention_patterns


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

    X_samp, _ = generate(model, X_init, context, steps, top_k=top_k, do_sample=do_sample)
    X_samp = X_samp.to('cpu')   

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

def generate_n_words(model, dataset, text, model_device='cpu', do_sample=False,
                     top_k=None, temperature=1.0, num_steps=1250, n_words=4, is_hooked=False):
    print(f"Generating {n_words} words with text: '{text}'")
    print(f"Model device: {model_device}, Is hooked: {is_hooked}")

    SEED_TOKENS = torch.tensor(
        [289,   0, 255,   8, 266,  18, 262,  14, 262,  14, 262,  14, 261,   9,
        260,  15, 258,  13, 248,   8, 378,  13, 352,   9, 337,  11, 337,  13,
        337,  10, 337,  13, 337,  11, 345,   8, 342,  12, 378,   3, 258,  13,
        258,  13, 261,   9, 265,  15, 276,  13, 297,   7, 337,  11, 345,   8,
        367,  15, 374,  13, 373,   9, 289,  13, 311,  11, 330,   8, 301,  13,
        267,  11, 262,  14, 268,  14, 278,  10, 332,  11, 337,  11, 289,   0,
        262,  14, 270,   9, 276,  13, 295,   9, 337,   8, 337,  11, 311,  11,
        289,   9, 270,  15, 265,  15, 267,  11, 280,  12, 316,   9, 333,  13,
        351,  14, 352,   9, 317,   7, 322,  12, 330,   8, 337,  11, 345,   8,
        357,  14, 367,  15, 232,   8, 258,  13, 270,  18, 267,  11, 267,  14,
        267,  11, 265,  15, 265,  15, 261,   9, 276,  13, 337,  11, 337,   8,
        337,   4, 262,   9, 262,  14, 273,  14, 274,  11, 319,  11, 337,  11,
        346,   8, 321,   8, 285,  13, 289, 100, 289, 157], dtype=torch.int64)
    SEED_CHARS = 'knzn'

    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
    warmup_steps = len(SEED_TOKENS)
    ascii_context = f'{SEED_CHARS} {text}'

    def trunc_or_pad_words(text):
        n = len(text.split(' '))
        if n > n_words:
            print(f"Expected {n_words+1} words, got {n}; truncating")
            return ' '.join(text.split(' ')[:n_words])
        elif n < n_words:
            print(f"Expected {n_words+1} words, got {n}; padding with 'hello'")
            return text + ' hello'*(n_words-n)
        return text
    
    text = trunc_or_pad_words(text)
    print(f"Processed text: '{text}'")

    context = dataset.encode_text(ascii_context).unsqueeze(0)
    print(f"Context shape: {context.shape}")
    context = context.to(model_device)
    X_init = SEED_TOKENS.unsqueeze(0).to(model_device)
    print(f"X_init shape: {X_init.shape}")

    steps = num_steps - X_init.size(1)
    print(f"Generating for {steps} steps")

    try:
        X_samp, attention_patterns = generate(model, X_init, context, steps, temperature=temperature,
                                              top_k=top_k, do_sample=do_sample, is_hooked=is_hooked)
        print(f"Generation successful, X_samp shape: {X_samp.shape}")
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        return None, None, None

    X_samp = X_samp.to('cpu')

    stroke_seq = X_samp[0].detach().cpu().numpy()[len(SEED_TOKENS):]
    offset_samp = dataset.decode_stroke(stroke_seq)
    point_samp = offsets_to_strokes(offset_samp)

    print("Generation completed successfully")
    return offset_samp, point_samp, attention_patterns
########## ARGS, LOGGING, AND TRAIN LOOP ##########


if __name__ == '__main__':

    args = get_all_args()
    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    train_dataset, test_dataset = create_datasets(args)  # init datasets
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.vocab_size = train_dataset.get_vocab_size()
    args.context_vocab_size = train_dataset.get_char_vocab_size()
    print(f"Dataset determined that: {args.vocab_size=}, {args.block_size=}")

    model, optimizer, scheduler, step, best_loss = get_checkpoint(args, sample_only=True)

    save_samples(model, test_dataset, num=6, do_sample=True, log_wandb=False)
    save_samples(model, test_dataset, num=6, do_sample=False, log_wandb=False)
    sys.exit()
