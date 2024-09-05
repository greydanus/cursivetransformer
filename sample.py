########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, argparse, getpass
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import Transformer, save_checkpoint
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

    parser = argparse.ArgumentParser(description='Generate a word bank')
    parser.add_argument('--device', type=str, default='cuda', help='This is meant to be trained on a GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--n_layer', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--n_embd', type=int, default=64, help='Number of embedding dimensions in self attention')
    parser.add_argument('--n_embd2', type=int, default=64, help='Number of embedding dimensions in cross attention')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads in Transformer block')

    # parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_size', type=int, default=9000, help='Number of train examples')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test examples')
    parser.add_argument('--num_words', type=int, default=4, help='Number of words')
    parser.add_argument('--max_seq_length', type=int, default=1000, help='Maximum sequence length (tokens)')
    parser.add_argument('--augment', action='store_true', default=True, help='Perform augmentations')
    parser.add_argument('--ablate_cross_attention', action='store_true', default=False, help='Ablate the cross attention')
    parser.add_argument('--add_digits', action='store_true', default=True, help='Add digit words to the word bank')
    parser.add_argument('--alphabet', type=str, default=" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ",
                            help='All the characters that this model will be able to draw')
    parser.add_argument('--dataset_name', type=str, default='bigbank', help='Set this to your wandb username or team name')

    parser.add_argument('--wandb_project', type=str, default='synthbank_experiments', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='sam-greydanus', help='Set this to your wandb username or team name')
    parser.add_argument('--wandb_run_name', type=str, default='unnamed_run', help='W&B run name')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='Weights & Biases API Key')

    parser.add_argument('--local_checkpoint_path', type=str, default='best_checkpoint.pt', help='Path to local model file')
    parser.add_argument('--load_from_run_id', type=str, default=None, help='Resume from a specific W&B run ID')

    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

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

    if os.path.exists(args.local_checkpoint_path):
        checkpoint = torch.load(args.local_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.local_checkpoint_path}")
    else:
        print("Downloading checkpoint from W&B")
        api = wandb.Api()
        artifact = api.artifact(f'{args.wandb_entity}/{args.wandb_project}/{args.resume_from_run_id or args.wandb_run_name}:model:latest')
        model_dir = artifact.download()
        checkpoint = torch.load(f"{model_dir}/{args.local_checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        save_checkpoint(model, args.local_checkpoint_path)

    if os.path.exists(args.local_model_path):
        model.load_state_dict(torch.load(args.local_model_path, weights_only=True))
        print(f"Loaded model from {args.local_model_path}")
    elif args.load_from_run_id:
        print("Downloading model from W&B")
        api = wandb.Api()
        artifact = api.artifact(f'{args.wandb_entity}/{args.wandb_project}/{args.load_from_run_id}:model:latest')
        model_dir = artifact.download()
        model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", weights_only=True))
        torch.save(model.state_dict(), args.local_model_path)
    else:
        print("No local model or W&B run ID provided. Exiting.")
        sys.exit()

    save_samples(model, test_dataset, num=6, do_sample=True, log_wandb=False)
    save_samples(model, test_dataset, num=6, do_sample=False, log_wandb=False)
    sys.exit()