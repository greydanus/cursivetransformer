import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        # query projections for all heads
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        # key, value projections for all heads
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_ctx_head = config.n_ctx_head
        self.n_embd = config.n_embd

    def forward(self, x, context):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        _, T_ctx, _ = context.size()

        # calculate query for all heads in batch and move head forward to be the batch dim
        q = (
            self.c_attn_q(x)
            .view(B, T, self.n_ctx_head, C // self.n_ctx_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k, v = self.c_attn_kv(context).split(self.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(
            1, 2
        )  # (B, nh, T_ctx, hs)
        v = v.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(
            1, 2
        )  # (B, nh, T_ctx, hs)

        # cross-attention; (B, nh, T, hs) x (B, nh, hs, T_ctx) -> (B, nh, T, T_ctx)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T_ctx) x (B, nh, T_ctx, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)  # NEW
        self.ln_3 = nn.LayerNorm(config.n_embd)  # NEW
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x, context):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), context)
        x = x + self.mlpf(self.ln_3(x))
        return x


class Transformer(nn.Module):
    """Transformer Language Model, exactly as seen in GPT-2"""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                wce=nn.Embedding(config.context_vocab_size, config.n_embd),  # NEW
                wcpe=nn.Embedding(config.context_block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(
            "Number of Transformer parameters: {:.0f}".format(
                n_params,
            )
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, context, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # forward the GPT model itself
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        context_t = context.size(-1)
        context_pos = torch.arange(
            0, context_t, dtype=torch.long, device=device
        ).unsqueeze(
            0
        )  # shape (1, t)
        context_emb = self.transformer.wce(
            context
        )  # context embeddings of shape (b, t_ctx, n_embd)
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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss
