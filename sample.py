# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, getpass, textwrap
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
from data import create_datasets, offsets_to_strokes, word_offsets_to_points


def plot_strokes(stroke, title, fig=None, ax=None, figsize=(12, 2), dpi=150):
    """Plot a single stroke"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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
        point_samp = word_offsets_to_points(offset_samp)
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


def generate_helper_fn(model, dataset, text, num_steps=1250, do_sample=False,
                         top_k=None, temperature=1.0, n_words=4, seed_ix=0, verbose=False):
    '''Assumes we're using tokenization of git commit afc2425f5bf92c14a9db62da44e8cf2995e7bf8d'''
    SEED_TOKENS = [torch.tensor(
        [341,   0, 232,  13, 445,  13, 232,  13, 432,  12, 390,  13, 391,
        13, 350,   9, 335,  13, 347,  13, 372,  13, 396,  13, 424,  12,
       439,  13, 232,  16, 341, 116, 454, 454], dtype=torch.int64),
        torch.tensor(
        [341,   0, 232,  11, 444,  12, 232,  11, 417,  14, 397,  13, 392,
        13, 331,  16, 341,  11, 368,  13, 393,  18, 421,  12, 438,  12,
       232,  15, 341, 116, 454, 454], dtype=torch.int64)][seed_ix]
    SEED_CHARS = '5'

    model_device = next(model.parameters()).device
    warmup_steps = len(SEED_TOKENS)

    def trunc_or_pad_words(text):
      n = len(text.split(' '))
      if n > n_words:
        if verbose: print(f"Expected {n_words+1} words, got {n}; truncating")
        return ' '.join(text.split(' ')[:n_words])
      elif n < n_words:
        if verbose: print(f"Expected {n_words+1} words, got {n}; padding with 'hello'")
        return text + ' hello'*(n_words-n)
      return text
    text = trunc_or_pad_words(text)
    ascii_context = f'{SEED_CHARS} {text}'

    context = dataset.encode_text(ascii_context).unsqueeze(0)
    context = context.to(model_device)
    X_init = SEED_TOKENS.unsqueeze(0).to(model_device)

    steps = num_steps - X_init.size(1)
    X_samp = generate(model, X_init, context, steps, temperature=temperature,
                      top_k=top_k, do_sample=do_sample).to('cpu')

    stroke_seq = X_samp[0].detach().cpu().numpy()[len(SEED_TOKENS):]
    offset_samp = dataset.decode_stroke(stroke_seq)
    point_samp = word_offsets_to_points(offset_samp)

    return offset_samp, point_samp


def generate_paragraph(model, dataset, text, n_at_a_time=3, **kwargs):
    word_list = text.split(' ')
    word_list_offsets = []
    print('Generating...')
    for i in range(0, len(word_list), n_at_a_time):
        words_to_generate = word_list[i:i+n_at_a_time]
        text_chunk = ' '.join(words_to_generate)
        offset_samp, _ = generate_helper_fn(model, dataset, text=text_chunk, **kwargs)
        word_list_offsets += offset_samp[:len(words_to_generate)]
        print('   ', text_chunk)
    return word_list_offsets


def word_offsets_to_points(word_offsets, space_width=0.17, line_width=12.0, line_height=0.75, 
                          min_x=0, max_y=5.0):  # Add bounds parameters
    word_points = []
    last_point = None
    current_x = current_y = 0
    
    for offsets in word_offsets:
        points = offsets_to_strokes(offsets)
        if last_point is not None:
            points = points + last_point[np.newaxis, :]
            # Check if word exceeds line width and wrap if needed
            if len(points) > 0 and current_x + (points[-1][0] - points[0][0]) > line_width:
                current_x = min_x  # Reset to minimum x bound
                current_y = min(current_y + line_height, max_y)  # Bound maximum y
                points = points + np.array([current_x - points[0][0], current_y - points[0][1], 0])
        
        if len(points) > 0:
            # Update last point and add space for next word
            last_point = points[-1].copy()
            last_point[0] = (current_x := max(min_x, min(last_point[0] + space_width, line_width)))
            last_point[1] = min(current_y, max_y)
            
        word_points.append(points)
    
    return np.vstack(word_points)


def plot_paragraph(word_list_offsets, text, figsize=(12, 4*2), dpi=200, **kwargs):
    point_samp = word_offsets_to_points(word_list_offsets, **kwargs)
    fig, ax = plot_strokes(point_samp, '', figsize=figsize, dpi=dpi)
    ax.set_title('\n'.join(textwrap.wrap(text, width=83)), loc='left', fontsize=13)


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

