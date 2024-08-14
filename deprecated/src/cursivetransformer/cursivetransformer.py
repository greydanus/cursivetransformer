import argparse
import copy
import io
import json
import math
import os
import pdb
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import comb
from typing import Any, AnyStr, Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from google.colab import files
from scipy.ndimage import rotate
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from wandb.errors import CommError

# Try attaching to GPU
DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Using:", DEVICE)


def get_time_string(fmt="%m%d_%H%M"):
    return datetime.now().strftime(fmt)


@dataclass
class ExperimentConfig:
    experiment_type: str = "pretraining"
    wandb_project: str = "cursivetransformer"
    wandb_entity: str = "cursivetransformer"
    wandb_run_name: str = field(init=False)
    work_dir: str = "out"
    resume: bool = False
    sample_only: bool = False
    num_workers: int = 1
    max_steps: int = 50000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42069
    top_k: int = -1
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    augment: bool = True
    max_seq_length: int = 1500
    batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    cross_attention_types: List[str] = field(
        default_factory=lambda: ["standard", "causal"]
    )
    cross_attention_type: str = "standard"

    def __post_init__(self):
        if self.experiment_type == "cross_attention_ablation":
            self.wandb_run_name = f"{get_time_string()}_{self.experiment_type}_{self.cross_attention_type}"
        else:
            self.wandb_run_name = f"{get_time_string()}_{self.experiment_type}"

    def __json__(self):
        return {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in asdict(self).items()
            if not k.startswith("_")
        }

    def update(self, new_config: Dict[str, Any]) -> "ExperimentConfig":
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class DataConfig:
    block_size: int = field(init=False)
    context_block_size: int = field(init=False)
    vocab_size: int = field(init=False)
    context_vocab_size: int = field(init=False)
    context_length: int = field(init=False)
    train_dataset: Any = field(init=False)
    test_dataset: Any = field(init=False)

    def __post_init__(self):
        self.train_dataset, self.test_dataset = create_datasets(
            augment=True, max_seq_length=1500
        )
        self.vocab_size = self.train_dataset.get_vocab_size()
        self.block_size = self.train_dataset.get_stroke_seq_length()
        self.context_block_size = self.train_dataset.get_text_seq_length()
        self.context_vocab_size = self.train_dataset.get_char_vocab_size()
        self.context_length = self.train_dataset.get_text_seq_length()


@dataclass
class ModelConfig:
    block_size: int = None
    context_block_size: int = None
    vocab_size: int = None
    context_vocab_size: int = None
    context_length: int = None
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    n_ctx_head: int = 4
    cross_attention_type: str = "standard"  # Options: "standard", "causal"


def get_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()


def get_data_config() -> DataConfig:
    return DataConfig()


def get_latest_checkpoint(exp_config):
    api = wandb.Api()
    runs = api.runs(
        f"{exp_config.wandb_entity}/{exp_config.wandb_project}",
        {
            "$and": [
                {"config.experiment_type": exp_config.experiment_type},
                {"config.experiment_params": exp_config},
                {"state": {"$in": ["running", "finished"]}},
            ]
        },
    )

    if not runs:
        return None, None

    latest_run = max(runs, key=lambda run: run.created_at)
    try:
        artifacts = latest_run.logged_artifacts()
        checkpoints = [
            artifact for artifact in artifacts if artifact.type == "model-checkpoint"
        ]
        if not checkpoints:
            return latest_run, None
        latest_checkpoint = max(checkpoints, key=lambda c: c.version)
        return latest_run, latest_checkpoint
    except CommError:
        return latest_run, None


def run_experiment(exp_config: ExperimentConfig, data_config: DataConfig):
    # Check for existing run and checkpoint
    existing_run, checkpoint = get_latest_checkpoint(exp_config)

    if existing_run and existing_run.state == "finished":
        print(
            f"Experiment {exp_config.experiment_type} with params {exp_config} has already been completed."
        )
        return None, existing_run.summary.get("best_loss", float("inf"))

    # Set up the experiment based on the experiment type
    if exp_config.experiment_type == "pretraining":
        config = ModelConfig(
            vocab_size=data_config.vocab_size,
            block_size=data_config.block_size,
            context_block_size=data_config.context_block_size,
            context_vocab_size=data_config.context_vocab_size,
            n_layer=exp_config.n_layer,
            n_head=exp_config.n_head,
            n_embd=exp_config.n_embd,
            n_embd2=exp_config.n_embd2,
        )
    elif exp_config.experiment_type == "cross_attention_ablation":
        config = ModelConfig(
            vocab_size=data_config.vocab_size,
            block_size=data_config.block_size,
            context_block_size=data_config.context_block_size,
            context_vocab_size=data_config.context_vocab_size,
            n_layer=exp_config.n_layer,
            n_head=exp_config.n_head,
            n_embd=exp_config.n_embd,
            n_embd2=exp_config.n_embd2,
            cross_attention_type=exp_config.cross_attention_type,
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp_config.experiment_type}")

    model = Transformer(config).to(exp_config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_config.learning_rate,
        weight_decay=exp_config.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    start_step = 0
    best_loss = float("inf")

    if existing_run and exp_config.resume:
        run = wandb.init(id=existing_run.id, resume="must")
        print(f"Resuming run {run.name}")
        if checkpoint:
            checkpoint_dir = checkpoint.download()
            checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pt")
            checkpoint_data = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            start_step = checkpoint_data["step"]
            best_loss = checkpoint_data["best_loss"]
            print(f"Loaded checkpoint from step {start_step}")
    else:
        run = wandb.init(
            project=exp_config.wandb_project,
            entity=exp_config.wandb_entity,
            name=exp_config.wandb_run_name,
            config={
                **exp_config.__dict__,
                "experiment_type": exp_config.experiment_type,
            },
            group=exp_config.experiment_type,
        )

    batch_loader = InfiniteDataLoader(
        data_config.train_dataset,
        batch_size=exp_config.batch_size,
        pin_memory=True,
        num_workers=exp_config.num_workers,
    )

    for step in range(start_step, exp_config.max_steps):
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(exp_config.device) for t in batch]
        X, C, Y = batch

        logits, loss = model(X, C, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if exp_config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 100 == 0:
            print(
                f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms"
            )

        wandb.log(
            {
                "train_loss_step": loss.item(),
                "step": step,
                "step_time_ms": (t1 - t0) * 1000,
            }
        )

        if step > 0 and step % 2000 == 0:
            train_loss = evaluate(
                model,
                exp_config,
                data_config.train_dataset,
                batch_size=100,
                max_batches=10,
            )
            test_loss = evaluate(
                model,
                exp_config,
                data_config.test_dataset,
                batch_size=100,
                max_batches=10,
            )
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step})
            print(
                f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}"
            )

            if test_loss < best_loss:
                best_loss = test_loss
                checkpoint_path = f"best_model_{exp_config.wandb_run_name}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "best_loss": best_loss,
                    },
                    checkpoint_path,
                )
                artifact = wandb.Artifact(
                    f"model-checkpoint-{step}", type="model-checkpoint"
                )
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)
                print(f"New best model saved with test loss: {best_loss:.4f}")

            save_samples(model, data_config.test_dataset, num=3, do_sample=True)
            save_samples(model, data_config.test_dataset, num=3, do_sample=False)

    wandb.finish()
    return model, best_loss


@torch.inference_mode()
def evaluate(model, exp_config, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(exp_config.device) for t in batch]
        X, C, Y = batch
        logits, loss = model(X, C, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # reset model back to training mode
    return mean_loss


@torch.no_grad()
def generate(
    model, idx, context, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
):
    block_size = model.get_block_size()
    steps = max(0, max_new_tokens - idx.size(1))
    for i in range(steps):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond, context)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def save_samples(
    model, dataset, num=2, model_device="cpu", warmup_steps=100, do_sample=False
):
    model_device = list(model.parameters())[0].device  # hacky

    stroke_seq, context = [], []
    for i in range(num):
        x, c, y = dataset[i]
        stroke_seq.append(x)
        context.append(c)

    X_init = torch.stack(stroke_seq).to(model_device)[:, :warmup_steps]
    context = torch.stack(context).long().to(model_device)
    top_k = None
    steps = (
        dataset.get_stroke_seq_length() - 1
    )  # -1 because we already start with the first token

    X_samp = generate(
        model, X_init, context, steps, top_k=top_k, do_sample=do_sample
    ).to("cpu")

    for i in range(X_samp.size(0)):
        row = X_samp[i].detach().cpu().numpy()
        offset_samp = dataset.decode_stroke(row)
        point_samp = offsets_to_strokes(offset_samp)
        decoded_ascii = dataset.decode_text(context[i])

        fig, ax = plot_strokes(point_samp, f"Sample {i+1}: '{decoded_ascii}'")
        tag = "sample" if do_sample else "topk"
        fig.savefig(f"{dataset.name}_{tag}_{i+1}.png")
        wandb.log(
            {
                f"{dataset.name}_{tag}_{i+1}": wandb.Image(
                    f"{dataset.name}_{tag}_{i+1}.png"
                )
            }
        )
        plt.close(fig)
        print(f"Saved {dataset.name}_{tag}_{i+1}.png")

    print("-" * 80)


def plot_strokes(stroke, title, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))

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

    for stroke in strokes:
        x, y = zip(*[(p[0], 1 - p[1]) for p in stroke])  # Invert y-axis
        ax.plot(x, y, "b-")

    ax.set_aspect("equal")
    ax.set_title(title)

    if fig is None:
        plt.show()

    return fig, ax


def load_and_parse_data(min_ascii_length=3):
    uploaded = files.upload()
    file_content = next(iter(uploaded.values()))
    data = json.loads(file_content.decode("utf-8"))
    for i in range(len(data)):
        strokes = np.array(data[i]["points"])
        strokes[:, 0:1] *= data[i]["metadata"]["aspectRatio"]
        strokes[:, 0] -= strokes[0, 0]
        data[i]["points"] = strokes
    data = [d for d in data if len(d["metadata"]["asciiSequence"]) >= min_ascii_length]
    return data


def decompose_offsets(offsets):
    dx, dy = offsets[:, 0], offsets[:, 1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return np.column_stack((theta, r, offsets[:, 2]))


def reconstruct_offsets(polar_data):
    theta, r = polar_data[:, 0], polar_data[:, 1]
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack((dx, dy, polar_data[:, 2]))


def strokes_to_offsets(points):
    offsets = np.zeros_like(points)
    offsets[1:, 0:2] = np.diff(points[:, 0:2], axis=0)  # Compute dx, dy
    offsets[:, 2] = points[:, 2]  # Copy pen_down directly
    offsets_dec = decompose_offsets(offsets)
    return offsets_dec


def offsets_to_strokes(offsets_dec):
    offsets = reconstruct_offsets(offsets_dec)
    absolute_coords = np.cumsum(offsets[:, :2], axis=0)
    stroke_data = np.hstack((absolute_coords, offsets[:, 2:3]))
    return stroke_data


def combine_handwriting_examples(examples, space_width=0.17):
    assert (
        len(set(ex["metadata"]["author"] for ex in examples)) == 1
    ), "All examples must have the same author"

    combined_metadata = {
        "author": examples[0]["metadata"]["author"],
        "asciiSequence": " ".join(ex["metadata"]["asciiSequence"] for ex in examples),
        "pointCount": sum(ex["metadata"]["pointCount"] for ex in examples),
        "strokeCount": sum(ex["metadata"]["strokeCount"] for ex in examples),
        "aspectRatio": examples[0]["metadata"]["aspectRatio"],
    }

    combined_points, current_x_offset, total_width = [], 0, 0

    for i, example in enumerate(examples):
        points = example["points"]
        word_width = np.max(points[:, 0]) - np.min(points[:, 0])
        total_width += word_width

        normalized_points = points.copy()
        normalized_points[:, 0] -= np.min(points[:, 0])
        normalized_points[:, 0] += current_x_offset

        combined_points.append(normalized_points)
        current_x_offset += word_width

        if i < len(examples) - 1:
            combined_points.append(
                np.array(
                    [[current_x_offset + space_width, normalized_points[-1, 1], 0]]
                )
            )
            current_x_offset += space_width
            total_width += space_width
            combined_metadata["pointCount"] += 1

    combined_points = np.vstack(combined_points)
    return {"metadata": combined_metadata, "points": combined_points}


def rotate_points(points, max_angle=10):
    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    x, y = points[:, 0], points[:, 1]
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle

    points[:, 0], points[:, 1] = x_rot, y_rot
    return points


def shear_points(points, shear_range=(-0.4, 0.4)):
    shear_factor = np.random.uniform(*shear_range)
    x, y = points[:, 0], points[:, 1]
    x_sheared = x + shear_factor * y
    points[:, 0] = x_sheared
    return points


def generate_word_combos(
    raw_json,
    desired_num_combos=10000,
    num_words=3,
    max_angle=4,
    shear_range=(-0.4, 0.4),
):
    num_combos = comb(len(raw_json), num_words)
    print(
        f"For a dataset of {len(raw_json)} examples we can generate {num_combos} combinations of {num_words} examples."
    )
    print(
        f"Generating {desired_num_combos} random (and thus possibly overlapping) combos..."
    )
    combo_json = []
    for i in range(desired_num_combos):
        ixs = np.random.choice(len(raw_json), size=num_words, replace=False)
        words_to_merge = [raw_json[i] for i in ixs]
        example = combine_handwriting_examples(words_to_merge)
        example["points"] = shear_points(example["points"], shear_range)
        example["points"] = rotate_points(example["points"], max_angle)
        combo_json.append(example)
    return combo_json


def load_and_combine_examples(desired_num_combos=10000, num_words=3):
    data = load_and_parse_data()
    return generate_word_combos(data, desired_num_combos, num_words)


def remove_random_points(stroke, remove_percentage=0.04):
    num_points = np.random.randint(len(stroke))
    num_remove = int(num_points * remove_percentage)
    indices = np.random.choice(
        range(1, num_points - 1), num_remove, replace=False
    ).astype(np.int32)
    return np.delete(stroke, indices, axis=0)


def efficient_downsample(stroke, fraction=0.65):
    n = len(stroke)
    keep = int(n * fraction)
    drop_indices = np.random.choice(n, n - keep, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[drop_indices] = False
    pen_up_mask = stroke[:, -1] == 0
    mask |= pen_up_mask
    return stroke[mask]


class StrokeDataset(Dataset):
    def __init__(
        self,
        strokes,
        texts,
        chars,
        max_seq_length=1100,
        max_text_length=50,
        name="",
        augment=False,
    ):
        self.name = name
        self.strokes = strokes
        self.texts = texts
        self.chars = chars
        self.augment = augment

        self.theta_bins = np.linspace(-np.pi, np.pi, 226)
        mag_bins_pen_down = np.concatenate(
            [
                np.asarray([0]),
                np.linspace(0.005, 0.050, 50),
                np.geomspace(0.051, 4, 121)[:-1],
            ]
        )
        mag_bins_pen_up = mag_bins_pen_down + max(mag_bins_pen_down) + 1
        self.mag_bins = np.concatenate([mag_bins_pen_down, mag_bins_pen_up])

        self.feature_sizes = [len(self.theta_bins), len(self.mag_bins)]
        self.cumulative_sizes = np.cumsum([0] + self.feature_sizes)

        self.PAD_TOKEN = sum(self.feature_sizes)
        self.END_TOKEN = sum(self.feature_sizes) + 1

        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.char_PAD_TOKEN = 0

        self.max_seq_length = max_seq_length
        self.max_text_length = max_text_length

    def augment_stroke(self, stroke):
        stroke[:, 1:2] = stroke[:, 1:2] * np.random.uniform(0.8, 1.2)
        noise = np.random.normal(0, 0.002, stroke[:, 1:2].shape)
        stroke[:, 1:2] += noise
        stroke = stroke[np.random.randint(1, 8) : -np.random.randint(1, 8)]
        stroke = efficient_downsample(stroke, 0.4)
        return stroke

    def __len__(self):
        return len(self.strokes)

    def get_vocab_size(self):
        return sum(self.feature_sizes) + 2

    def get_char_vocab_size(self):
        return len(self.chars) + 1

    def get_stroke_seq_length(self):
        return self.max_seq_length

    def get_text_seq_length(self):
        return self.max_text_length

    def encode_stroke(self, stroke):
        theta_idx = np.digitize(stroke[:, 0], self.theta_bins) - 1
        mag_idx = (
            np.digitize(stroke[:, 1], self.mag_bins[: len(self.mag_bins) // 2]) - 1
        )
        mag_idx[stroke[:, 2] == 0] += len(self.mag_bins) // 2
        encoded = np.column_stack(
            [theta_idx + self.cumulative_sizes[0], mag_idx + self.cumulative_sizes[1]]
        )
        return encoded.flatten()

    def decode_stroke(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()
        ix = ix[(ix != self.PAD_TOKEN) & (ix != self.END_TOKEN)]
        ix = ix[: (len(ix) // 2) * 2]
        ix = ix.reshape(-1, 2)
        theta = self.theta_bins[
            (ix[:, 0] - self.cumulative_sizes[0]).clip(0, len(self.theta_bins) - 1)
        ]
        mag_idx = ix[:, 1] - self.cumulative_sizes[1]
        pen = (mag_idx < len(self.mag_bins) // 2).astype(int)
        mag_idx[pen == 0] -= len(self.mag_bins) // 2
        mag = self.mag_bins[: len(self.mag_bins) // 2][
            mag_idx.clip(0, len(self.mag_bins) // 2 - 1)
        ]
        return np.column_stack([theta, mag, pen])

    def encode_text(self, text):
        return torch.tensor(
            [self.stoi.get(ch, self.char_PAD_TOKEN) for ch in text], dtype=torch.long
        )

    def decode_text(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()
        return "".join([self.itos.get(i, "") for i in ix if i != self.char_PAD_TOKEN])

    def __getitem__(self, idx):
        stroke = self.strokes[idx]
        text = self.texts[idx]

        if self.augment:
            stroke = self.augment_stroke(stroke.copy())

        stroke_offsets = self.strokes_to_polar_offsets(stroke)
        encoded_stroke = self.encode_stroke(stroke_offsets)
        x = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)
        y = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)

        seq_len = min(len(encoded_stroke), self.max_seq_length - 1)
        x[:seq_len] = torch.tensor(encoded_stroke[:seq_len], dtype=torch.long)
        x[seq_len] = self.END_TOKEN

        y[:seq_len] = x[1 : seq_len + 1]
        y[seq_len] = self.END_TOKEN

        encoded_text = self.encode_text(text)
        c = torch.full((self.max_text_length,), self.char_PAD_TOKEN, dtype=torch.long)
        text_len = min(len(encoded_text), self.max_text_length)
        c[:text_len] = encoded_text[:text_len]
        return x, c, y

    @staticmethod
    def strokes_to_polar_offsets(stroke):
        offsets = np.diff(stroke[:, :2], axis=0)
        theta = np.arctan2(offsets[:, 1], offsets[:, 0])
        r = np.hypot(offsets[:, 0], offsets[:, 1])
        pen_state = stroke[1:, 2]
        return np.column_stack((theta, r, pen_state))


def create_datasets(augment=True, max_seq_length=1100, num_words=3):
    raw_json = load_and_parse_data()

    test_set_size = min(1000, int(len(raw_json) * 0.10))
    rp = torch.randperm(len(raw_json)).tolist()
    train_examples = [raw_json[i] for i in rp[:-test_set_size]]
    test_examples = [raw_json[i] for i in rp[-test_set_size:]]

    train_examples = generate_word_combos(
        train_examples, desired_num_combos=98000, num_words=num_words
    )
    test_examples = generate_word_combos(
        test_examples, desired_num_combos=2000, num_words=num_words
    )

    train_strokes = [copy.deepcopy(v["points"]) for v in train_examples]
    train_texts = [
        copy.deepcopy(v["metadata"]["asciiSequence"]) for v in train_examples
    ]

    test_strokes = [copy.deepcopy(v["points"]) for v in test_examples]
    test_texts = [copy.deepcopy(v["metadata"]["asciiSequence"]) for v in test_examples]

    chars = "abcdefghijklmnopqrstuvwxyz "
    print(f"Number of examples in the train dataset: {len(train_examples)}")
    print(f"Number of examples in the test dataset: {len(test_examples)}")
    print(f"Max token sequence length: {max_seq_length}")
    print(f"Number of unique characters in the ascii vocabulary: {len(chars)}")
    print("Ascii vocabulary:")
    print(f"\t'{chars}'")

    print(
        f"Split up the dataset into {len(train_examples)} training examples and {len(test_examples)} test examples"
    )

    train_dataset = StrokeDataset(
        train_strokes, train_texts, chars, max_seq_length, name="train", augment=augment
    )
    test_dataset = StrokeDataset(
        test_strokes, test_texts, chars, max_seq_length, name="test", augment=augment
    )
    return train_dataset, test_dataset


class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_ctx_head = config.n_ctx_head
        self.n_embd = config.n_embd

    def forward(self, x, context):
        B, T, C = x.size()
        _, T_ctx, _ = context.size()

        q = (
            self.c_attn_q(x)
            .view(B, T, self.n_ctx_head, C // self.n_ctx_head)
            .transpose(1, 2)
        )
        k, v = self.c_attn_kv(context).split(self.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2)
        v = v.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class CausalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_ctx_head = config.n_ctx_head
        self.n_embd = config.n_embd

    def forward(self, x, context):
        B, T, C = x.size()
        _, T_ctx, _ = context.size()

        q = (
            self.c_attn_q(x)
            .view(B, T, self.n_ctx_head, C // self.n_ctx_head)
            .transpose(1, 2)
        )
        k, v = self.c_attn_kv(context).split(self.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2)
        v = v.view(B, T_ctx, self.n_ctx_head, C // self.n_ctx_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        mask = torch.tril(torch.ones(T, T_ctx)).view(1, 1, T, T_ctx).to(x.device)
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class NewGELU(nn.Module):
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
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config, cross_attention_class):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = cross_attention_class(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x, context):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), context)
        x = x + self.mlpf(self.ln_3(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config

        if config.cross_attention_type == "causal":
            cross_attention_class = CausalCrossAttention
        elif config.cross_attention_type == "standard":
            cross_attention_class = CrossAttention
        else:
            raise NotImplementedError(
                f"Cross Attention type {config.cross_attention_type} not implemented!"
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                wce=nn.Embedding(config.context_vocab_size, config.n_embd),
                wcpe=nn.Embedding(config.context_block_size, config.n_embd),
                h=nn.ModuleList(
                    [
                        Block(config, cross_attention_class)
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of Transformer parameters: {:.0f}".format(n_params))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, context, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        context_t = context.size(-1)
        context_pos = torch.arange(
            0, context_t, dtype=torch.long, device=device
        ).unsqueeze(0)
        context_emb = self.transformer.wce(context)
        context_pos_emb = self.transformer.wcpe(context_pos)
        c = context_emb + context_pos_emb

        for block in self.transformer.h:
            x = block(x, c)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss
