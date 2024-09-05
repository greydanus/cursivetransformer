########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, math, io, copy, json, pickle, glob, functools, zipfile, argparse, getpass
import numpy as np
from scipy.ndimage import rotate
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass
from math import comb

import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


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


def save_samples(model, dataset, num=2, model_device='cpu', warmup_steps=100, do_sample=False):
    """ samples from the model and plots the decoded strokes """
    model_device = list(model.parameters())[0].device # hacky

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
    parser.add_argument('--dataset_name', type=str, default='bigbank', help='Set this to your wandb username or team name')

    parser.add_argument('--wandb_project', type=str, default='synthbank_experiments', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='sam-greydanus', help='Set this to your wandb username or team name')
    parser.add_argument('--wandb_run_name', type=str, default='unnamed_run', help='W&B run name')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='Weights & Biases API Key')

    parser.add_argument('--resume', action='store_true', default=False, help='Load model from checkpoint')
    parser.add_argument('--local_model_path', type=str, default='best_model.pt', help='Path to local model file')

    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not args.sample_only:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=args)

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
    if args.resume or args.sample_only:
        if os.path.exists(args.local_model_path):
            model.load_state_dict(torch.load(args.local_model_path))
            print(f"Loaded model from {args.local_model_path}")
        else:
            print("Downloading model from W&B")
            artifact = wandb.use_artifact(f'{args.wandb_entity}/{args.wandb_project}/best_model:latest')
            model_dir = artifact.download()
            model.load_state_dict(torch.load(f"{model_dir}/best_model.pt"))
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
    wandb.watch(model, log="all", log_freq=args.sample_every, log_graph=False)



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
        if step > 0 and step % args.sample_every == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step })
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")

            if best_loss is None or test_loss < best_loss:  # save the model to W&B if it has improved
                best_loss = test_loss
                torch.save(model.state_dict(), args.local_model_path)
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(args.local_model_path)
                wandb.log_artifact(artifact)


        # sample from the model
        if step > 0 and step % args.sample_every == 0:
            save_samples(model, test_dataset, num=6, do_sample=True)
            save_samples(model, test_dataset, num=6, do_sample=False)
            save_samples(model, train_dataset, num=3, do_sample=True)
            save_samples(model, train_dataset, num=3, do_sample=False)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    wandb.finish()