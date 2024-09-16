########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import math, os, sys, argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F

import wandb


########## ALL ARGUMENTS ##########

def get_all_args(use_argparse=True):
    args_config = {
        'max_steps': (110000, int, 'How many steps to train for'),
        'print_every': (100, int, 'Print log info after how many steps'),
        'log_every': (2500, int, 'Sample model after how many steps'),
        'lr_decay': (0.333, float, 'How much to decay the learning rate'),
        'step_lr_every': (33000, int, 'How often to decay the learning rate'),
        'device': ('cuda', str, 'This is meant to be trained on a GPU'),
        'seed': (42, int, 'Random seed for reproducibility'),
        'n_layer': (4, int, 'Number of Transformer layers'),
        'n_embd': (64, int, 'Number of embedding dimensions in self attention'),
        'n_embd2': (64, int, 'Number of embedding dimensions in cross attention'),
        'n_ctx_head': (4, int, 'Number of attention heads in Transformer block'),
        'learning_rate': (1e-2, float, 'Learning rate'),
        'weight_decay': (1e-4, float, 'Weight decay'),
        'batch_size': (32, int, 'Batch size'),
        'train_size': (497000, int, 'Number of train examples'),
        'test_size': (3000, int, 'Number of test examples'),
        'num_words': (4, int, 'Number of words'),
        'max_seq_length': (1000, int, 'Maximum sequence length (tokens)'),
        'augment': (True, 'store_true', 'Perform augmentations'),
        'ablate_cross_attention': (False, 'store_true', 'Ablate the cross attention'),
        'downsample_mean': (0.65, float, 'Mean amount to downsample stroke points (0.65=65%)'),
        'downsample_width': (0.1, float, 'Width of the uniform distribution (0.1=10%)'),
        'add_digits': (True, 'store_true', 'Add digit words to the word bank'),
        'alphabet': (" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ", str, 
                        'All the characters that this model will be able to draw'),
        'dataset_name': ('zachbigbank', str, 'Set this to your wandb username or team name'),
        'wandb_project': ('bigbank_experiments', str, 'W&B project name'),
        'wandb_entity': ('zwimpee', str, 'Set this to your wandb username or team name'),
        'wandb_run_name': ('unnamed_run', str, 'W&B run name'),
        'wandb_api_key': (None, str, 'Weights & Biases API Key'),
        'load_from_run_id': (None, str, 'Load from a specific W&B run ID'),
        'sample_only': (False, 'store_true', 'Only sample from the model'),
        'local_checkpoint_path': ('best_checkpoint.pt', str, 'Path to local model file'),
    }

    if use_argparse:
        parser = argparse.ArgumentParser(description='Train a cursivetransformer model')
        for arg, (default, arg_type, help_text) in args_config.items():
            if arg_type == 'store_true':
                parser.add_argument(f'--{arg}', action=arg_type, default=default, help=help_text)
            else:
                parser.add_argument(f'--{arg}', type=arg_type, default=default, help=help_text)
        return parser.parse_args()
    else:
        return SimpleNamespace(**{k: v[0] for k, v in args_config.items()})



########## MODEL I/O ##########

def get_checkpoint(args):
    model = Transformer(args)
    model.to(args.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")

    if not args.sample_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_every, gamma=args.lr_decay)
    else:
        optimizer = None
        scheduler = None

    step = 0
    best_loss = None

    if args.load_from_run_id or args.sample_only:
        if os.path.exists(args.local_checkpoint_path):
            checkpoint = torch.load(args.local_checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from local path: {args.local_checkpoint_path}")
            if not args.sample_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                step = checkpoint['step']
                best_loss = checkpoint['best_loss']
        elif args.load_from_run_id:
            artifact = get_latest_checkpoint_artifact(args)
            artifact_dir = artifact.download()
            checkpoint = torch.load(os.path.join(artifact_dir, "best_checkpoint.pt"), weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if not args.sample_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                step = checkpoint['step'] + 1
                best_loss = checkpoint['best_loss']
            
            save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss)
        else:
            print("No local model or W&B run ID provided. Exiting.")
            sys.exit()

    return model, optimizer, scheduler, step, best_loss



def get_latest_checkpoint_artifact(args, verbose=True):
    run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{args.load_from_run_id}")

    if verbose:
        print(f"Finding latest checkpoint for W&B run id {args.load_from_run_id}")
    latest_artifact = None
    get_version = lambda artifact: -1 if artifact is None else int(artifact.name.split(':v')[-1])
    for artifact in run.logged_artifacts():
        if verbose:
            print(f"  {artifact.type}:{artifact.name}")
        if artifact.type == 'model' and (get_version(artifact) > get_version(latest_artifact)):
            latest_artifact = artifact
    if verbose:
        print(f"Selected:  {latest_artifact.type}:{latest_artifact.name}")
    return latest_artifact


def save_checkpoint(model, path, optimizer=None, scheduler=None, step=None, best_loss=None):
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if step is not None:
        checkpoint['step'] = step
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    torch.save(checkpoint, path)


########## MAIN MODEL DEFINITION ##########


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_ctx_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd2 % config.n_ctx_head == 0
        # query projections for all heads
        self.c_attn_q = nn.Linear(config.n_embd2, config.n_embd2)
        # key, value projections for all heads
        self.c_attn_kv = nn.Linear(config.n_embd2, 2 * config.n_embd2)
        # output projection
        self.c_proj = nn.Linear(config.n_embd2, config.n_embd2)
        self.n_ctx_head = config.n_ctx_head
        self.n_embd2 = config.n_embd2

    def forward(self, x, context):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd2)
        _, T_ctx, _ = context.size()

        # calculate query for all heads in batch and move head forward to be the batch dim
        q = self.c_attn_q(x).view(B, T, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2) # (B, nh, T, hs)

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.c_attn_kv(context).split(self.n_embd2, dim=2)
        k = k.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2) # (B, nh, T_ctx, hs)
        v = v.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2) # (B, nh, T_ctx, hs)

        # cross-attention; (B, nh, T, hs) x (B, nh, hs, T_ctx) -> (B, nh, T, T_ctx)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T_ctx) x (B, nh, T_ctx, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd2)
        self.cross_attn = CrossAttention(config) # NEW
        self.ln_3 = nn.LayerNorm(config.n_embd) # NEW
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x, context):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), context)
        x = x + self.mlpf(self.ln_3(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            wce = nn.Embedding(config.context_vocab_size, config.n_embd2), # NEW
            wcpe = nn.Embedding(config.context_block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of Transformer parameters: {:.0f}".format(n_params,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, context, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # forward the GPT model itself
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        context_t = context.size(-1)
        context_pos = torch.arange(0, context_t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        context_emb = self.transformer.wce(context) # context embeddings of shape (b, t_ctx, n_embd2)
        context_pos_emb = self.transformer.wcpe(context_pos)
        c = context_emb + context_pos_emb

        if self.config.ablate_cross_attention:
          c = torch.zeros_like(c)

        for block in self.transformer.h:
            x = block(x, c)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss