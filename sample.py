# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, getpass, textwrap, copy
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


@dataclass
class GenerationParams:
    """Arguments for handwriting generation/sampling"""
    temperature: float = 1.0
    top_k: bool = None
    do_sample: bool = False
    num_steps: int = 1050
    warmup_steps: int = 50
    n_at_a_time: int = 2
    n_words: int = 4
    space_width: float = 0.16
    line_width: float = 8.0
    line_height: float = 0.55
    letter_height: float = 0.35
    warmup_sample_ix: int = None
    verbose: bool = True
    seed: int = 42


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


def save_samples(model, dataset, num=2, model_device='cpu', warmup_steps=50, do_sample=False, log_wandb=True):
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


def generate_helper_fn(model, dataset, word_list, params):
    model_device = next(model.parameters()).device
    warmup_sample_ix = params.warmup_sample_ix if params.warmup_sample_ix else torch.randint(len(dataset), (1,)).item()
    if params.verbose: print(f' (warmup_sample_ix={warmup_sample_ix})')

    seed_x, seed_c, _ = dataset[warmup_sample_ix]  # Get seed tokens and text from dataset
    word_tokens = dataset.split_by_word_tokens(seed_x)  # Get just first word tokens
    first_word_tokens = torch.tensor(word_tokens[0])
    first_word_tokens = torch.cat([first_word_tokens, torch.tensor([dataset.WORD_TOKEN])])  # Add word token
    warmup_steps = len(first_word_tokens)
    
    # Get just the first word from the context
    seed_text = dataset.decode_text(seed_c)
    first_word = seed_text.split()[0]
    
    def trunc_or_pad_words(word_list):
        n = len(word_list) ; n_words = params.n_words
        if n > n_words:
            # if params.verbose: print(f"Expected {n_words} words, got {n}; truncating")
            return word_list[:n_words-1]
        elif n < n_words:
            # if params.verbose: print(f"Expected {n_words} words, got {n}; padding with placeholder words")
            return word_list + ['Hkggcvr!', 'TOLAPYPI', '9074', '0.', 'efhgb.'][:max(0, n_words-n-1)]
        return word_list

    word_list = trunc_or_pad_words(word_list)
    text = ' '.join(word_list)
    ascii_context = f'{first_word} {text}'

    context = dataset.encode_text(ascii_context).unsqueeze(0)
    context = context.to(model_device)
    X_init = first_word_tokens.unsqueeze(0).to(model_device)

    steps = params.num_steps - X_init.size(1)
    X_samp = generate(model, X_init, context, steps, temperature=params.temperature,
                      top_k=params.top_k, do_sample=params.do_sample).to('cpu')

    stroke_seq = X_samp[0].detach().cpu().numpy()[warmup_steps:]
    offset_samp = dataset.decode_stroke(stroke_seq)

    return offset_samp


def generate_paragraph(model, dataset, text, params, word_list_offsets=None, regenerate_ixs=None):
    torch.manual_seed(params.seed)  # system inits
    torch.cuda.manual_seed_all(params.seed)

    word_list = text.strip(' ').split(' ')
    if word_list_offsets is None:
        word_list_offsets = []
        if params.verbose: print('Generating...')
        for i in range(0, len(word_list), params.n_at_a_time):
            word_list_subset = word_list[i:i+params.n_at_a_time]
            if params.verbose: print('   ', ' '.join(word_list_subset), end='')
            offset_sample = generate_helper_fn(model, dataset, word_list_subset, params)
            word_list_offsets += offset_sample[:len(word_list_subset)]
    else:
        # Regenerate specific words if requested
        if regenerate_ixs:
            if params.verbose: print('Regenerating words at indices:', regenerate_ixs)
            for i in regenerate_ixs:
                if i >= len(word_list):
                    continue
                if params.verbose: print('   ', word_list[i], end='')
                offset_sample = generate_helper_fn(model, dataset, [word_list[i]], params)
                word_list_offsets[i] = offset_sample[0]

    return word_list_offsets


def word_offsets_to_points(word_offsets, params, word_list=None):  # Add bounds parameters
    word_points = []
    last_point = None
    current_x = current_y = 0

    starts_at_bottom = "enaitoshrdx.vpukbgfcymzwlqjS,GJ"
    starts_at_top = "8049637OTA5N)EHR\"\'(BCQLMWYUF!DXVKP"  # starts_elsewhere = "1I2Z?"

    sentence_points = []
    for i, offsets in enumerate(word_offsets):

      points = offsets_to_strokes(copy.deepcopy(offsets))

      if word_list:
        word = word_list[i]
        if word[0] in starts_at_bottom:
          points[:,1] -= points[0,1]  # # print('Was at the bottom')
        elif word[0] in starts_at_top:
          points[:,1] -= points[0,1] + 0.18 #pass #

      if current_x > params.line_width:
        current_x = 0
        current_y += params.line_height

      if points is not None and points.shape[0] > 0:
        points[:,0] = points[:,0] + current_x
        points[:,1] = np.clip(points[:,1], -params.letter_height, params.letter_height) + current_y
        current_x = points[-1, 0] + params.space_width
        sentence_points.append(points)

    return sentence_points


def add_word_indices(ax, sentence_points):  # index numbers above each word start position
    for i, points in enumerate(sentence_points):
        start_x, start_y = points[0, 0], points[0, 1]
        ax.text(start_x-0.12, .95-start_y, str(i), fontsize=8, ha='left', va='bottom')


def plot_paragraph(word_list_offsets, text, figsize=(12, 4*2), dpi=200,
                   params=None, show_indices=False, include_title=False):
    params = params if params else GenerationParams()
    sentence_points = word_offsets_to_points(word_list_offsets, params, word_list=text.split())
    point_samp = np.vstack(sentence_points)
    fig, ax = plot_strokes(point_samp, '', figsize=figsize, dpi=dpi)

    if show_indices:
        add_word_indices(ax, sentence_points)
    if include_title:
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