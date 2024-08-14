from dataclasses import dataclass

from .utils import get_time_string


@dataclass
class AppConfig:
    # system/input/output
    work_dir: str = "out"
    resume: bool = False
    sample_only: bool = False
    num_workers: int = 1
    max_steps: int = 100000
    lr_decay: float = 1.0
    seed: int = 3407

    # sampling
    top_k: int = -1

    # model configuration
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    ablate_cross_attention: bool = False
    augment: bool = True
    max_seq_length: int = 1250

    # optimization
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4

    # wandb parameters
    wandb_project: str = "cursivetransformer"
    wandb_entity: str = "zwimpee"
    wandb_run_name: str = f"{get_time_string()}_cursivetransformer"


@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences of integers
    context_block_size: int = None
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    context_vocab_size: int = None  # size of the context vocabulary (ASCII characters)
    context_length: int = None  # maximum length of the context sequence
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    n_ctx_head: int = 4  # number of heads for cross-attention
    ablate_cross_attention: bool = False
