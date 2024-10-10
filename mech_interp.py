import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Tuple

import matplotlib.pyplot as plt

from typing import Dict, Optional, Union

from .model import get_latest_checkpoint_artifact

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities.attention import simple_attn_linear
from transformer_lens.components import (
    Attention,
    GroupedQueryAttention,
    LayerNorm,
    LayerNormPre,
    RMSNorm,
    RMSNormPre,
    Embed,
    Unembed
)
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.utilities.addmm import batch_addmm


class HookedCursiveTransformerConfig(HookedTransformerConfig):
    def __init__(self, **kwargs):
        # Extract custom arguments
        self.d_model_c = kwargs.pop('d_model_c', None)
        self.context_block_size = kwargs.pop('context_block_size', None)
        self.context_vocab_size = kwargs.pop('context_vocab_size', None)
        # self.use_cross_attention = kwargs.pop('use_cross_attention', True)

        # W&B specific parameters
        self.wandb_entity = kwargs.pop('wandb_entity', None)
        self.wandb_project = kwargs.pop('wandb_project', None)
        self.load_from_run_id = kwargs.pop('load_from_run_id', None)

        # Now, call the superclass constructor with the remaining kwargs
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class HookedCursiveTransformer(HookedTransformer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # Embedding layers
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.embed_c = nn.Embedding(cfg.context_vocab_size, cfg.d_model_c)
        self.pos_embed_c = nn.Embedding(cfg.context_block_size, cfg.d_model_c)

        # Projection layer if d_model_c != d_model
        if cfg.d_model_c != cfg.d_model:
            self.context_proj = nn.Linear(cfg.d_model_c, cfg.d_model)
        else:
            self.context_proj = nn.Identity()

        # Blocks
        self.blocks = nn.ModuleList([TransformerBlock(cfg, block_index=_) for _ in range(cfg.n_layers)])

        # Final layers
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=True)

        # Hook points
        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.hook_embed_c = HookPoint()
        self.hook_pos_embed_c = HookPoint()

        self.setup()

    # - [ ] TODO: DEBUG `per_token_loss`, currently still returning a scalar
    def forward(self, tokens, context, targets=None,return_type="logits", per_token_loss=False):
        B, T = tokens.shape
        B_c, T_c = context.shape

        token_embed = self.hook_embed(self.embed(tokens))
        pos_embed = self.hook_pos_embed(self.pos_embed(torch.arange(T, device=tokens.device)))
        x = token_embed + pos_embed

        context_embed = self.hook_embed_c(self.embed_c(context))
        context_pos_embed = self.hook_pos_embed_c(self.pos_embed_c(torch.arange(T_c, device=context.device)))
        c = context_embed + context_pos_embed
        c = self.context_proj(c)
        for block in self.blocks:
            x = block(x, c)

        x = self.ln_final(x)
        logits = self.unembed(x)

        if return_type == "logits":
            return logits
        elif return_type == "loss":
            if per_token_loss:
                loss = self.loss_fn(logits, targets, per_token=True)
                return loss
            else:
                loss = self.loss_fn(logits, targets)
                return loss.mean()
        elif return_type == "both":
            if per_token_loss:  
                loss = self.loss_fn(logits, targets, per_token=True)
                return logits, loss
            else:
                loss = self.loss_fn(logits, targets)
                return logits, loss.mean()
        else:
            raise ValueError(f"Invalid return_type {return_type}")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cfg,
        tokenizer=None,
        **from_pretrained_kwargs
    ):
        """
        Load a pretrained CursiveTransformer model into the HookedCursiveTransformer format.

        Args:
            model_name (str): The name or path of the pretrained model.
            cfg: The configuration object for the model.
            tokenizer: The tokenizer to use (optional).
            **from_pretrained_kwargs: Additional keyword arguments.

        Returns:
            HookedCursiveTransformer: The loaded model.
        """
        print(f"Loading pretrained model {model_name}")

        # Initialize the HookedCursiveTransformer with the given config
        model = cls(cfg)

        # Load the state dict from the wandb artifact
        state_dict = cls.load_state_dict_from_wandb(cfg)

        # Convert the state dict to match HookedCursiveTransformer format
        converted_state_dict = cls.convert_cursivetransformer_weights(state_dict, cfg)

        # Load the converted state dict into the model
        model.load_state_dict(converted_state_dict, strict=False)

        if tokenizer is not None:
            model.tokenizer = tokenizer

        print(f"Successfully loaded pretrained model {model_name}")
        return model

    @staticmethod
    def load_state_dict_from_wandb(args):
        artifact = get_latest_checkpoint_artifact(args)
        artifact_dir = artifact.download()
        checkpoint = torch.load(os.path.join(artifact_dir, "best_checkpoint.pt"), weights_only=True)
        return checkpoint['model_state_dict']

    @staticmethod
    def convert_cursivetransformer_weights(state_dict, cfg):
        """Convert CursiveTransformer weights to HookedCursiveTransformer format."""
        new_state_dict = {}

        # Embeddings
        new_state_dict["embed.W_E"] = state_dict["transformer.wte.weight"]
        new_state_dict["pos_embed.W_pos"] = state_dict["transformer.wpe.weight"]
        new_state_dict["embed_c.W_E"] = state_dict["transformer.wce.weight"]
        new_state_dict["pos_embed_c.W_pos"] = state_dict["transformer.wcpe.weight"]

        for l in range(cfg.n_layers):
            # Layer Norms
            new_state_dict[f'blocks.{l}.ln1.w'] = state_dict[f'transformer.h.{l}.ln_1.weight']
            new_state_dict[f'blocks.{l}.ln1.b'] = state_dict[f'transformer.h.{l}.ln_1.bias']
            new_state_dict[f'blocks.{l}.ln2.w'] = state_dict[f'transformer.h.{l}.ln_2.weight']
            new_state_dict[f'blocks.{l}.ln2.b'] = state_dict[f'transformer.h.{l}.ln_2.bias']
            new_state_dict[f'blocks.{l}.ln3.w'] = state_dict[f'transformer.h.{l}.ln_3.weight']
            new_state_dict[f'blocks.{l}.ln3.b'] = state_dict[f'transformer.h.{l}.ln_3.bias']

            # Self-Attention
            W_qkv = state_dict[f'transformer.h.{l}.attn.c_attn.weight']
            b_qkv = state_dict[f'transformer.h.{l}.attn.c_attn.bias']
            W_q, W_k, W_v = W_qkv.t().chunk(3, dim=1)
            b_q, b_k, b_v = b_qkv.chunk(3, dim=0)

            new_state_dict[f'blocks.{l}.attn.W_Q'] = W_q.t().reshape(cfg.n_heads, cfg.d_model, cfg.d_head)
            new_state_dict[f'blocks.{l}.attn.W_K'] = W_k.t().reshape(cfg.n_heads, cfg.d_model, cfg.d_head)
            new_state_dict[f'blocks.{l}.attn.W_V'] = W_v.t().reshape(cfg.n_heads, cfg.d_model, cfg.d_head)
            new_state_dict[f'blocks.{l}.attn.b_Q'] = b_q.reshape(cfg.n_heads, cfg.d_head)
            new_state_dict[f'blocks.{l}.attn.b_K'] = b_k.reshape(cfg.n_heads, cfg.d_head)
            new_state_dict[f'blocks.{l}.attn.b_V'] = b_v.reshape(cfg.n_heads, cfg.d_head)

            W_o = state_dict[f'transformer.h.{l}.attn.c_proj.weight']
            new_state_dict[f'blocks.{l}.attn.W_O'] = W_o.t().reshape(cfg.n_heads, cfg.d_head, cfg.d_model)
            new_state_dict[f'blocks.{l}.attn.b_O'] = state_dict[f'transformer.h.{l}.attn.c_proj.bias']

            # Cross-Attention
            W_q = state_dict[f'transformer.h.{l}.cross_attn.c_attn_q.weight']
            b_q = state_dict[f'transformer.h.{l}.cross_attn.c_attn_q.bias']
            new_state_dict[f'blocks.{l}.cross_attn.W_Q'] = W_q.t().reshape(cfg.n_heads, cfg.d_model, cfg.d_head)
            new_state_dict[f'blocks.{l}.cross_attn.b_Q'] = b_q.reshape(cfg.n_heads, cfg.d_head)

            W_kv = state_dict[f'transformer.h.{l}.cross_attn.c_attn_kv.weight']
            b_kv = state_dict[f'transformer.h.{l}.cross_attn.c_attn_kv.bias']
            W_k, W_v = W_kv.t().chunk(2, dim=1)
            b_k, b_v = b_kv.chunk(2, dim=0)

            new_state_dict[f'blocks.{l}.cross_attn.W_K'] = W_k.t().reshape(cfg.n_heads, cfg.d_model_c, cfg.d_head)
            new_state_dict[f'blocks.{l}.cross_attn.W_V'] = W_v.t().reshape(cfg.n_heads, cfg.d_model_c, cfg.d_head)
            new_state_dict[f'blocks.{l}.cross_attn.b_K'] = b_k.reshape(cfg.n_heads, cfg.d_head)
            new_state_dict[f'blocks.{l}.cross_attn.b_V'] = b_v.reshape(cfg.n_heads, cfg.d_head)

            W_o = state_dict[f'transformer.h.{l}.cross_attn.c_proj.weight']
            new_state_dict[f'blocks.{l}.cross_attn.W_O'] = W_o.t().reshape(cfg.n_heads, cfg.d_head, cfg.d_model)
            new_state_dict[f'blocks.{l}.cross_attn.b_O'] = state_dict[f'transformer.h.{l}.cross_attn.c_proj.bias']

            # MLP
            new_state_dict[f'blocks.{l}.mlp.W_in'] = state_dict[f'transformer.h.{l}.mlp.c_fc.weight'].t()
            new_state_dict[f'blocks.{l}.mlp.b_in'] = state_dict[f'transformer.h.{l}.mlp.c_fc.bias']
            new_state_dict[f'blocks.{l}.mlp.W_out'] = state_dict[f'transformer.h.{l}.mlp.c_proj.weight'].t()
            new_state_dict[f'blocks.{l}.mlp.b_out'] = state_dict[f'transformer.h.{l}.mlp.c_proj.bias']

        # Final layer norm and unembedding
        new_state_dict["ln_final.w"] = state_dict["transformer.ln_f.weight"]
        new_state_dict["ln_final.b"] = state_dict["transformer.ln_f.bias"]
        new_state_dict["unembed.W_U"] = state_dict["lm_head.weight"].t()
        new_state_dict["unembed.b_U"] = state_dict.get("lm_head.bias", torch.zeros(cfg.d_vocab))

        return new_state_dict

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__()
        self.cfg = HookedCursiveTransformerConfig.unwrap(cfg)

        self.cfg_c = copy.deepcopy(cfg)
        self.cfg_c.attention_dir = 'bidirectional'

        # Determine normalization type
        if self.cfg.normalization_type == "LN":
            normalization_layer = LayerNorm
        elif self.cfg.normalization_type == "LNPre":
            normalization_layer = LayerNormPre
        elif self.cfg.normalization_type == "RMS":
            normalization_layer = RMSNorm
        elif self.cfg.normalization_type == "RMSPre":
            normalization_layer = RMSNormPre
        elif self.cfg.normalization_type is None:
            normalization_layer = lambda cfg: nn.Identity()
        else:
            raise ValueError(f"Invalid normalization_type: {self.cfg.normalization_type}")

        # Initialize layers
        self.ln1 = normalization_layer(cfg)
        self.ln2 = normalization_layer(cfg)
        self.ln3 = normalization_layer(cfg)

        attention_class = Attention
        cross_attention_class = CrossAttention
        self.attn = attention_class(self.cfg, "global", block_index)
        self.cross_attn = cross_attention_class(self.cfg_c, "global", block_index)
        self.mlp = MLPFactory.create_mlp(self.cfg)

        # Hook points
        self.hook_attn_in = HookPoint()
        self.hook_cross_attn_in = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_cross_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid1 = HookPoint()
        self.hook_resid_mid2 = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        context: Float[torch.Tensor, "batch context_len d_model_c"],
        attention_mask: Optional[Int[torch.Tensor, "batch 1 seq_len seq_len"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        resid_pre = self.hook_resid_pre(resid_pre)

        # Self-attention
        attn_in = self.hook_attn_in(resid_pre)
        normalized_attn_in = self.ln1(attn_in)
        attn_out = self.attn(
            query_input=normalized_attn_in,
            key_input=normalized_attn_in,
            value_input=normalized_attn_in,
            attention_mask=attention_mask,
        )
        attn_out = self.hook_attn_out(attn_out)
        resid_mid1 = self.hook_resid_mid1(resid_pre + attn_out)

        # Cross-attention
        cross_attn_in = self.hook_cross_attn_in(resid_mid1)
        normalized_cross_attn_in = self.ln2(cross_attn_in)
        cross_attn_out = self.cross_attn(
            query_input=normalized_cross_attn_in,
            key_input=context,
            value_input=context,
            attention_mask=None,  # No mask for cross-attention
        )
        cross_attn_out = self.hook_cross_attn_out(cross_attn_out)
        resid_mid2 = self.hook_resid_mid2(resid_mid1 + cross_attn_out)

        # MLP
        mlp_in = self.hook_mlp_in(resid_mid2)
        normalized_mlp_in = self.ln3(mlp_in)
        mlp_out = self.mlp(normalized_mlp_in)
        mlp_out = self.hook_mlp_out(mlp_out)

        resid_post = self.hook_resid_post(resid_mid2 + mlp_out)
        return resid_post

class CrossAttention(Attention):
    def __init__(self, cfg, attn_type='global', layer_id=None):
        super().__init__(cfg, attn_type, layer_id)
        self.cfg = cfg

        # Initialize W_K and W_V for cross-attention with d_model_c
        self.W_K = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model_c, self.cfg.d_head, dtype=self.cfg.dtype
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                self.cfg.n_heads, self.cfg.d_model_c, self.cfg.d_head, dtype=self.cfg.dtype
            )
        )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        # W_Q and W_O are inherited from AbstractAttention

        # Override the attention direction
        # self.cfg.attention_dir = 'bidirectional'

    def calculate_qkv_matrices(self, query_input, key_input, value_input):
        # query_input: [batch, pos, d_model]
        # key_input and value_input: [batch, kv_pos, d_model_c]

        # Use W_Q and b_Q from AbstractAttention for queries
        q = self.hook_q(simple_attn_linear(query_input, self.W_Q, self.b_Q))
        # Use custom W_K and b_K for keys with d_model_c
        k = self.hook_k(simple_attn_linear(key_input, self.W_K, self.b_K))
        v = self.hook_v(simple_attn_linear(value_input, self.W_V, self.b_V))
        return q, k, v


class MLPFactory:
    @staticmethod
    def create_mlp(cfg):
        return MLP(cfg)

class MLP(CanBeUsedAsMLP):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)
        self.select_activation_function()

        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.d_mlp, dtype=self.cfg.dtype))
        self.b_in = nn.Parameter(torch.zeros(self.d_mlp, dtype=self.cfg.dtype))

        self.W_out = nn.Parameter(torch.empty(self.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = self.hook_pre(batch_addmm(self.b_in, self.W_in, x))  # [batch, pos, d_mlp]

        if (
            self.cfg.is_layer_norm_activation()
            and self.hook_mid is not None
            and self.ln is not None
        ):
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        return batch_addmm(self.b_out, self.W_out, post_act)


def convert_cursivetransformer_model_config(args):
    cfg_dict = {
        # Standard parameters
        "d_model": args.n_embd,
        "n_layers": args.n_layer,
        "d_mlp": args.n_embd * 4,
        "d_head": args.n_embd // args.n_ctx_head,
        "n_heads": args.n_ctx_head,
        "n_ctx": args.max_seq_length,
        "d_vocab": args.vocab_size,
        "tokenizer_name": None,
        "act_fn": "gelu_new",
        "attn_only": False,
        "final_rms": False,
        "attention_dir": "causal",
        "original_architecture": "cursivetransformer",
        "normalization_type": "LN",
        "init_weights": False,
        "device": args.device,
        # Additional parameters for cross-attention
        "d_model_c": args.n_embd2,
        "context_block_size": args.context_block_size,
        "context_vocab_size": args.context_vocab_size,
        # "use_cross_attention": True,
        # W&B specific parameters
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "load_from_run_id": args.load_from_run_id,
    }
    return HookedCursiveTransformerConfig.from_dict(cfg_dict)

# cfg = convert_cursivetransformer_model_config(args)
# model = HookedCursiveTransformer.from_pretrained("cursivetransformer", cfg)

def visualize_attention(model, x, c, layer_range=None, head_range=None, attn_type='self'):
    with torch.no_grad():
        _, cache = model.run_with_cache(x, c, return_type="both")

    if layer_range is None:
        layer_range = range(model.cfg.n_layers)
    if head_range is None:
        head_range = range(model.cfg.n_heads)

    n_layers = len(layer_range)
    n_heads = len(head_range)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4*n_heads, 4*n_layers), squeeze=False)

    for i, layer in enumerate(layer_range):
        for j, head in enumerate(head_range):
            if attn_type == 'self':
                attn_patterns = cache[f'blocks.{layer}.attn.hook_pattern']
            elif attn_type == 'cross':
                attn_patterns = cache[f'blocks.{layer}.cross_attn.hook_pattern']
            else:
                raise ValueError("attn_type must be 'self' or 'cross'")

            attn = attn_patterns[0, head].cpu().numpy()
            im = axes[i, j].imshow(attn, cmap='viridis', aspect='auto', interpolation=None)
            axes[i, j].set_title(f'Layer {layer}, Head {head}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def generate_repeated_stroke_tokens(
    model,
    seq_len: int,
    n_repeats: int,
    batch_size: int = 1
) -> Int[torch.Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated stroke tokens, alternating between θ and r tokens.

    Args:
        model: The model instance.
        seq_len: Number of (θ, r) pairs in the initial sequence.
        n_repeats: Number of times to repeat the sequence.
        batch_size: Batch size.

    Returns:
        rep_tokens: Tensor of shape [batch_size, n_repeats * 2 * seq_len]
    """
    device = model.cfg.device
    feature_sizes = test_dataset.feature_sizes  # [size_r_bins, size_theta_bins]
    cumulative_sizes = test_dataset.cumulative_sizes  # cumulative indices for token types

    # Get valid indices for θ and r tokens
    theta_token_indices = torch.arange(
        cumulative_sizes[1],
        cumulative_sizes[2],
        device=device
    )
    r_token_indices = torch.arange(
        cumulative_sizes[0],
        cumulative_sizes[1],
        device=device
    )

    # Generate random θ and r tokens
    random_theta_tokens = theta_token_indices[
        torch.randint(
            low=0,
            high=feature_sizes[1],
            size=(batch_size, seq_len),
            device=device
        )
    ]
    random_r_tokens = r_token_indices[
        torch.randint(
            low=0,
            high=feature_sizes[0],
            size=(batch_size, seq_len),
            device=device
        )
    ]

    # Alternate between θ and r tokens
    stroke_tokens_half = torch.zeros(batch_size, seq_len * 2, dtype=torch.long, device=device)
    stroke_tokens_half[:, 0::2] = random_theta_tokens
    stroke_tokens_half[:, 1::2] = random_r_tokens

    # Repeat the sequence
    rep_tokens = stroke_tokens_half.repeat(1, n_repeats)

    return rep_tokens

def generate_random_ascii_context(
    model,
    batch_size: int = 1
) -> Int[torch.Tensor, "batch_size context_seq_len"]:
    """
    Generates a random ASCII context sequence.

    Args:
        model: The model instance.
        batch_size: Batch size.

    Returns:
        context_tokens: Tensor of shape [batch_size, context_seq_len]
    """
    device = model.cfg.device
    context_seq_len = model.cfg.context_block_size
    context_vocab_size = model.cfg.context_vocab_size

    context_tokens = torch.randint(
        low=0,
        high=context_vocab_size - 1,  # Exclude PAD token
        size=(batch_size, context_seq_len),
        dtype=torch.long,
        device=device
    )

    return context_tokens

def run_and_cache_model_repeated_tokens(
    model,
    rep_tokens: torch.Tensor,
    context_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, ActivationCache]:
    """
    Runs the model on repeated tokens and caches activations.

    Args:
        model: The model instance.
        rep_tokens: Input stroke tokens of shape [batch_size, seq_len]
        context_tokens: Input context tokens of shape [batch_size, context_seq_len]

    Returns:
        logits: Model output logits.
        cache: Activation cache.
    """
    # Shift inputs to create targets
    inputs = rep_tokens[:, :-1]
    targets = rep_tokens[:, 1:]

    # Run model with cache
    logits, cache = model.run_with_cache(
        tokens=inputs,
        context=context_tokens,
        targets=targets,
        return_type="both"
    )

    return logits, targets, cache

# - [ ] TODO: use test_dataset.cumulative_sizes to get token index ranges for r and theta
def sanity_check_token_pairs(rep_tokens: torch.Tensor):
    """
    Performs a sanity check to ensure that token pairs are correctly formed.

    Args:
        rep_tokens: Input stroke tokens of shape [batch_size, seq_len]
    """
    batch_size, seq_len = rep_tokens.shape
    assert seq_len % 2 == 0, "Sequence length should be even to form (θ, r) pairs."

    token_pairs = rep_tokens.view(batch_size, seq_len // 2, 2)  # Shape: [batch_size, seq_len_pairs, 2]
    for pair in token_pairs[0]:
        theta, r = pair.tolist()
        # Add any specific checks for theta and r validity if applicable
        # For example, ensure that theta and r are within expected ranges
        # Here, we'll simply print a few pairs for manual inspection
    print("Sample token pairs:", token_pairs[0][:5].tolist())

def verify_attention_summation(cache: ActivationCache, layer: int, head: int, attn_type: str = 'self'):
    """
    Verifies that attention weights sum to 1 across key positions for each query.

    Args:
        cache: Activation cache.
        layer: Layer index.
        head: Head index.
        attn_type: 'self' or 'cross' to specify attention type.
    """
    with torch.no_grad():
        if attn_type == 'self':
            attn = cache[f'blocks.{layer}.attn.hook_pattern'][0, head]
        elif attn_type == 'cross':
            attn = cache[f'blocks.{layer}.cross_attn.hook_pattern'][0, head]
        else:
            raise ValueError("attn_type must be 'self' or 'cross'")

    attn_sum = attn.sum(dim=-1)
    if not torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5):
        print(f"Attention weights do not sum to 1 for Layer {layer}, Head {head} ({attn_type}-Attention).")
    else:
        print(f"Attention weights verified for Layer {layer}, Head {head} ({attn_type}-Attention).")

def compute_induction_scores(
    rep_tokens: torch.Tensor,
    cache: ActivationCache,
    model: HookedCursiveTransformer
) -> torch.Tensor:
    """
    Computes induction scores for all attention heads.

    Args:
        rep_tokens: Input tokens of shape [batch_size, seq_len]
        cache: Activation cache containing attention patterns.
        model: The transformer model.

    Returns:
        induction_scores: Tensor of shape [num_layers, num_heads]
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    induction_scores = torch.zeros(num_layers, num_heads, device=model.cfg.device)

    batch_size, seq_len = rep_tokens.shape

    tokens = rep_tokens[:, :-1]  # Exclude last token
    targets = rep_tokens[:, 1:]  # Exclude first token

    for layer in range(num_layers):
        attention = cache["pattern", layer]  # Shape: [batch_size, num_heads, seq_len - 1, seq_len - 1]

        for head in range(num_heads):
            # Extract attention weights for this head
            attn_weights = attention[:, head]  # Shape: [batch_size, seq_len - 1, seq_len - 1]

            # Initialize induction score for this head
            score_sum = 0.0

            for b in range(batch_size):
                for t in range(1, seq_len - 1):
                    # Current token and previous tokens
                    current_token = tokens[b, t]
                    previous_tokens = tokens[b, :t]

                    # Find positions where previous_tokens == current_token
                    matching_positions = (previous_tokens == current_token).nonzero(as_tuple=True)[0]

                    # For each matching position, check if the next token matches the target
                    for pos in matching_positions:
                        if targets[b, pos] == targets[b, t]:
                            # Accumulate attention weight
                            score_sum += attn_weights[b, t, pos].item()

            # Normalize the score
            induction_scores[layer, head] = score_sum / (batch_size * (seq_len - 2))

    return induction_scores

def compute_cross_attention_induction_scores(
    model,
    context_tokens: torch.Tensor,
    cache: ActivationCache
) -> torch.Tensor:
    """
    Computes induction-like scores for cross-attention heads.

    Args:
        model: The model instance.
        context_tokens: Context tokens of shape [batch_size, context_seq_len]
        cache: Activation cache.

    Returns:
        cross_induction_scores: Tensor of shape [num_layers, num_heads]
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    cross_induction_scores = torch.zeros(num_layers, num_heads, device=model.cfg.device)

    batch_size, context_seq_len = context_tokens.shape

    for layer in range(num_layers):
        attn_patterns = cache["pattern", layer, "cross_attn"]  # Need to access cross-attention patterns
        for head in range(num_heads):
            attn = attn_patterns[0, head]  # Shape: [stroke_seq_len, context_seq_len]
            # For this example, we might need more specific analysis based on the use case
            # Placeholder for cross-attention induction score computation
            cross_induction_scores[layer, head] = attn.mean().item()
    return cross_induction_scores

def plot_induction_scores(induction_scores: torch.Tensor):
    """
    Plots a heatmap of induction scores with categorical annotations.

    Args:
        induction_scores: Tensor of shape [num_layers, num_heads]
    """
    plt.figure(figsize=(12, 6))
    induction_scores_np = induction_scores.cpu().numpy()

    # Define thresholds
    high_threshold = 0.05
    medium_threshold = 0.02

    # Create a mask for categories
    categories = np.empty_like(induction_scores_np, dtype=str)
    categories[induction_scores_np > high_threshold] = 'High'
    categories[(induction_scores_np <= high_threshold) & (induction_scores_np > medium_threshold)] = 'Moderate'
    categories[induction_scores_np <= medium_threshold] = 'Low'

    # Plot heatmap
    sns.heatmap(
        induction_scores_np,
        annot=induction_scores_np,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[f"H{h}" for h in range(induction_scores.shape[1])],
        yticklabels=[f"L{l}" for l in range(induction_scores.shape[0])],
        cbar_kws={'label': 'Induction Score'}
    )

    # Overlay categories
    for i in range(induction_scores_np.shape[0]):
        for j in range(induction_scores_np.shape[1]):
            plt.text(j + 0.5, i + 0.5, categories[i, j],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='red' if categories[i, j] == 'High' else
                            'orange' if categories[i, j] == 'Moderate' else
                            'black',
                     fontsize=8)

    plt.title("Induction Scores per Head")
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.show()

def plot_head_attention_pattern(
    cache: ActivationCache,
    layer: int,
    head: int,
    seq_len: int,
    attn_type: str = 'self'
):
    """
    Plots the attention pattern of a specific head with enhanced clarity.

    Args:
        cache: Activation cache.
        layer: Layer index.
        head: Head index.
        seq_len: Total sequence length.
        attn_type: 'self' or 'cross' to specify attention type.
    """
    with torch.no_grad():
        if attn_type == 'self':
            attn = cache[f'blocks.{layer}.attn.hook_pattern'][0, head].cpu().numpy()
        elif attn_type == 'cross':
            attn = cache[f'blocks.{layer}.cross_attn.hook_pattern'][0, head].cpu().numpy()
        else:
            raise ValueError("attn_type must be 'self' or 'cross'")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        cmap='viridis',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(f"Attention Pattern - Layer {layer}, Head {head} ({attn_type}-Attention)")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.tight_layout()
    plt.show()

def create_induction_summary(induction_scores: torch.Tensor, threshold_high=0.05, threshold_medium=0.02):
    """
    Creates a summary table of induction scores with categories.

    Args:
        induction_scores: Tensor of shape [num_layers, num_heads]
        threshold_high: Threshold for high induction score
        threshold_medium: Threshold for moderate induction score

    Returns:
        df_summary: Pandas DataFrame with Layer, Head, Score, and Category
    """
    num_layers, num_heads = induction_scores.shape
    data = []
    for layer in range(num_layers):
        for head in range(num_heads):
            score = induction_scores[layer, head].item()
            if score > threshold_high:
                category = 'High'
            elif score > threshold_medium:
                category = 'Moderate'
            else:
                category = 'Low'
            data.append({'Layer': layer, 'Head': head, 'Score': score, 'Category': category})
    df_summary = pd.DataFrame(data)
    return df_summary

def ablate_heads(
    model: HookedCursiveTransformer,
    head_list: List[Tuple[int, int]],
    rep_tokens: torch.Tensor,
    context_tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    from transformer_lens.hook_points import HookPoint
    from functools import partial

    def zero_out_head_output(value, hook, head_idx):
        # value shape: [batch_size, seq_len, n_heads, d_head]
        value[:, :, head_idx, :] = 0.0
        return value

    # Set up hooks
    ablation_hooks = []
    for layer_idx, head_idx, _ in head_list:
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        hook_fn = partial(zero_out_head_output, head_idx=head_idx)
        ablation_hooks.append((hook_name, hook_fn))

    # Run the model with ablation hooks
    inputs = rep_tokens[:, :-1]
    logits = model.run_with_hooks(
        inputs,
        context=context_tokens,
        return_type="logits",
        fwd_hooks=ablation_hooks
    )

    return logits

def compute_loss_on_induction_positions(
    logits,
    targets: torch.Tensor,
    induction_positions: List[torch.Tensor]
):
    """
    Computes cross-entropy loss only on the specified positions.

    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size], or a tuple where the first element is logits.
        targets: Target tokens of shape [batch_size, seq_len]
        induction_positions: List of tensors indicating positions to include in loss

    Returns:
        loss: Scalar tensor representing the loss on the specified positions
    """
    import torch.nn.functional as F

    # If logits is a tuple, extract the first element
    if isinstance(logits, tuple):
        logits = logits[0]

    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)        # Shape: [batch_size * seq_len, vocab_size]
    targets_flat = targets.reshape(-1)                  # Shape: [batch_size * seq_len]

    # Create mask for induction positions
    mask = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=logits.device)
    for b in range(batch_size):
        indices = induction_positions[b] + b * seq_len

        # Ensure indices are within bounds
        if torch.any(indices >= batch_size * seq_len):
            raise ValueError(f"Indices out of bounds for batch {b}: {indices}")

        mask[indices] = True

    # Apply the mask
    logits_masked = logits_flat[mask]                   # Select logits at induction positions
    targets_masked = targets_flat[mask]                 # Select targets at induction positions

    # Compute loss
    if logits_masked.numel() == 0:
        print("No induction positions found. Returning zero loss.")
        return torch.tensor(0.0, device=logits.device, requires_grad=False)

    loss = F.cross_entropy(logits_masked, targets_masked)
    return loss

def get_induction_positions(
    rep_tokens: torch.Tensor,
    pattern_length: int,
    n_repeats: int
) -> List[torch.Tensor]:
    """
    Computes the positions in the sequence where induction occurs.

    Args:
        rep_tokens: Input tokens of shape [batch_size, seq_len]
        pattern_length: Length of the repeating pattern (number of (θ, r) pairs)
        n_repeats: Number of times the pattern repeats

    Returns:
        induction_positions: List of tensors, one per batch, containing positions of induction tokens
    """
    batch_size, seq_len = rep_tokens.shape
    induction_positions = []

    total_patterns = n_repeats
    pattern_seq_length = pattern_length * 2  # Times 2 because each pattern token is two tokens long (θ and r)
    expected_seq_len = total_patterns * pattern_seq_length

    # Check if the sequence length matches the expected length
    if seq_len != expected_seq_len:
        raise ValueError(f"Mismatch between seq_len ({seq_len}) and expected_seq_len ({expected_seq_len})")

    for b in range(batch_size):
        positions = []

        # Induction occurs starting from the second pattern
        for i in range(1, n_repeats):
            # Positions where the model should recall from previous patterns
            start = i * pattern_seq_length
            end = start + pattern_seq_length

            # Induction positions are the positions within the current pattern where the model should predict the next token based on the pattern
            induction_positions_in_pattern = torch.arange(start, end, device=rep_tokens.device)

            positions.append(induction_positions_in_pattern)

        # Concatenate positions for all repeats in the batch
        if positions:
            positions = torch.cat(positions)
        else:
            positions = torch.tensor([], dtype=torch.long, device=rep_tokens.device)

        induction_positions.append(positions)

    return induction_positions

def activation_patching(
    model: HookedCursiveTransformer,
    src_tokens: torch.Tensor,
    dst_tokens: torch.Tensor,
    layer: int,
    head: int,
    context_tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Patches activations from src_tokens into dst_tokens at a specific layer and head.

    Args:
        model: The transformer model.
        src_tokens: Source input tokens.
        dst_tokens: Destination input tokens.
        layer: Layer index.
        head: Head index.
        context_tokens: Optional context tokens.

    Returns:
        logits: Model logits with patched activations.
    """
    from transformer_lens.hook_points import HookPoint

    def replace_head_activation(dst_value, hook: HookPoint):
        # Run the model on src_tokens to get the source head activation
        src_cache = {}
        def save_src_activation(value, hook):
            src_cache["value"] = value.detach()
            return value

        # Run src_tokens to get the source activation
        model.run_with_hooks(
            src_tokens,
            context=context_tokens,
            fwd_hooks=[(hook.name, save_src_activation)]
        )

        # Replace the destination activation with source activation for the specified head
        dst_value[:, :, head, :] = src_cache["value"][:, :, head, :]
        return dst_value

    # Run the model with activation patching
    inputs = dst_tokens[:, :-1]
    logits = model.run_with_hooks(
        inputs,
        context=context_tokens,
        return_type="logits",
        fwd_hooks=[(f"v{layer}", replace_head_activation)]
    )

    return logits
    