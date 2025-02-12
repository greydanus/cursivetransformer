# Sam Greydanus | 2024

import os, sys, time, getpass, textwrap, copy
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import Transformer, get_checkpoint, get_all_args
from data import create_datasets, offsets_to_strokes

@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_k: Optional[int] = None
    do_sample: bool = False
    max_tokens: int = 1250
    words_per_batch: int = 3
    space_width: float = 0.14
    line_width: float = 10.0
    line_height: float = 0.40
    letter_height: float = 0.35

@torch.no_grad()
def generate_tokens(model, initial_tokens, context, max_new_tokens, params: GenerationParams) -> torch.Tensor:
    """Core token generation logic"""
    block_size = model.get_block_size()
    steps = max(0, max_new_tokens - initial_tokens.size(1))
    idx = initial_tokens
    
    for _ in range(steps):
        # Crop sequence if too long
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Get next token prediction
        logits, _ = model(idx_cond, context)
        logits = logits[:, -1, :] / params.temperature
        
        if params.top_k:
            v, _ = torch.topk(logits, params.top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) if params.do_sample else \
                  torch.topk(probs, k=1, dim=-1)[1]
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def prepare_seed_tokens(dataset, seed_stroke_tokens):
    """Get initial tokens for generation"""
    word_tokens = dataset.split_by_word_tokens(seed_stroke_tokens)
    first_word = torch.tensor(word_tokens[0])
    return torch.cat([first_word, torch.tensor([dataset.WORD_TOKEN])])

def prepare_context(dataset, words, seed_char_tokens):
    """Prepare context for generation"""
    first_word = dataset.decode_text(seed_char_tokens)
    text = f"{first_word} {' '.join(words)}"
    return dataset.encode_text(text).unsqueeze(0)

def generate_sequence(model, dataset, text: str, params: GenerationParams, seed_ix: Optional[int] = None):
    """Generate handwriting for a sequence of words"""
    device = next(model.parameters()).device
    words = text.strip().split()
    word_offsets = []
    
    for i in range(0, len(words), params.words_per_batch):
        batch_words = words[i:i + params.words_per_batch]
        
        # Get seed tokens
        if seed_ix is None:
            seed_ix = torch.randint(len(dataset), (1,)).item()
        seed_x, seed_c, _ = dataset[seed_ix]
        first_word_tokens = prepare_seed_tokens(dataset, seed_x)
        
        # Prepare context
        context = prepare_context(dataset, batch_words, seed_c)
        
        # Generate
        generated = generate_tokens(
            model,
            initial_tokens=first_word_tokens.unsqueeze(0).to(device),
            context=context.to(device),
            max_new_tokens=params.max_tokens,
            params=params
        )
        
        # Decode and store results
        stroke_seq = generated[0].cpu().numpy()[len(first_word_tokens):]
        offset_sample = dataset.decode_stroke(stroke_seq)
        word_offsets.extend(offset_sample[:len(batch_words)])
        
    return word_offsets

def word_offsets_to_points(word_offsets: List[np.ndarray], params: GenerationParams) -> np.ndarray:
    """Convert word offsets to absolute coordinates"""
    sentence_points = []
    current_x = current_y = 0

    for offsets in word_offsets:
        points = offsets_to_strokes(copy.deepcopy(offsets))
        
        if current_x > params.line_width:
            current_x = 0
            current_y += params.line_height

        if points is not None and points.shape[0] > 0:
            points[:,0] = points[:,0] + current_x
            points[:,1] = np.clip(points[:,1], -params.letter_height, params.letter_height) + current_y
            current_x = points[-1, 0] + params.space_width
            sentence_points.append(points)

    return np.vstack(sentence_points)

def plot_strokes(stroke, title='', figsize=(12, 2), dpi=150):
    """Plot a single stroke sequence"""
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

    ax.set_aspect('equal')
    ax.set_title(title)
    return fig, ax

def plot_paragraph(word_list_offsets: List[np.ndarray], text: str, params: GenerationParams):
    """Plot a paragraph of text"""
    points = word_offsets_to_points(word_list_offsets, params)
    fig, ax = plot_strokes(points, '')
    ax.set_title('\n'.join(textwrap.wrap(text, width=83)), loc='left', fontsize=13)
    return fig, ax

def save_samples(model, dataset, params: GenerationParams, num_samples: int = 2, 
                warmup_steps: int = 50, log_wandb: bool = True):
    """Generate and save samples"""
    device = next(model.parameters()).device
    stroke_seq, context = [], []
    
    for i in range(num_samples):
        x, c, _ = dataset[i]
        stroke_seq.append(x)
        context.append(c)

    X_init = torch.stack(stroke_seq).to(device)[:,:warmup_steps]
    context = torch.stack(context).long().to(device)
    steps = dataset.get_stroke_seq_length() - 1

    X_samp = generate_tokens(model, X_init, context, steps, params).to('cpu')

    for i in range(X_samp.size(0)):
        row = X_samp[i].detach().cpu().numpy()
        offset_samp = dataset.decode_stroke(row)
        point_samp = word_offsets_to_points([offset_samp], params)
        decoded_ascii = dataset.decode_text(context[i])

        fig, _ = plot_strokes(point_samp, f'Sample {i+1}: "{decoded_ascii}"')
        tag = 'sample' if params.do_sample else 'topk'
        filename = f"{dataset.name}_{tag}_{i+1}.png"
        fig.savefig(filename)
        
        if log_wandb:
            import wandb
            wandb.log({f"{dataset.name}_{tag}_{i+1}": wandb.Image(filename)})
        plt.close(fig)

def main():
    args = get_all_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Initialize datasets and model
    train_dataset, test_dataset = create_datasets(args)
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.vocab_size = train_dataset.get_vocab_size()
    args.context_vocab_size = train_dataset.get_char_vocab_size()
    
    model, optimizer, scheduler, step, best_loss = get_checkpoint(args, sample_only=True)

    # Generate samples with different parameters
    params = GenerationParams(do_sample=True)
    save_samples(model, test_dataset, params, num_samples=6, log_wandb=False)
    
    params.do_sample = False
    save_samples(model, test_dataset, params, num_samples=6, log_wandb=False)

if __name__ == '__main__':
    main()