import argparse
import functools
import json
import logging
from datetime import datetime
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from google.colab import files
from torch.nn import functional as F
from torch.utils.data import DataLoader


# Set up logging
def setup_logger():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M"
    )
    logger = logging.getLogger(__name__)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Cursive Transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.json",
        help="path to config file",
    )
    return parser.parse_args()


def plot_strokes(stroke, title, fig=None, ax=None):
    """Plot a single stroke"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 2), dpi=300)

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
        ax.plot(x, y, "b-", linewidth=1)

    ax.set_aspect("equal")
    ax.set_title(title)
    if fig is None:
        plt.show()
    return fig, ax


@functools.lru_cache(maxsize=5)
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


def generate_word_combos(raw_json, desired_num_combos=10000, num_words=3):
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
        combo_json.append(combine_handwriting_examples(words_to_merge))
    return combo_json


def load_and_combine_examples(desired_num_combos=10000, num_words=3):
    data = load_and_parse_data()
    return generate_word_combos(data, desired_num_combos, num_words)


def decompose_offsets(offsets):
    dx, dy = offsets[:, 0], offsets[:, 1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return np.column_stack((r, theta, offsets[:, 2]))


def reconstruct_offsets(polar_data):
    r, theta = polar_data[:, 0], polar_data[:, 1]
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack((dx, dy, polar_data[:, 2]))


def strokes_to_offsets(points):
    # Calculate differences (dx, dy), not considering pen_down
    offsets = np.zeros_like(points)
    offsets[1:, 0:2] = np.diff(points[:, 0:2], axis=0)  # Compute dx, dy
    offsets[:, 2] = points[:, 2]  # Copy pen_down directly

    # Decouple direction from magnitude (this will help with tokenization)
    offsets_dec = decompose_offsets(offsets)
    return offsets_dec


def offsets_to_strokes(offsets_dec):
    # Calculate cumulative sums to get absolute positions
    offsets = reconstruct_offsets(offsets_dec)

    absolute_coords = np.cumsum(offsets[:, :2], axis=0)
    stroke_data = np.hstack((absolute_coords, offsets[:, 2:3]))
    return stroke_data


def horizontal_shear(stroke, shear_range=(-0.4, 0.4)):
    shear_factor = np.random.uniform(*shear_range)
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    stroke[:, :2] = np.dot(stroke[:, :2], shear_matrix.T)
    return stroke


def remove_random_points(stroke, remove_percentage=0.04):
    num_points = np.random.randint(len(stroke))
    num_remove = int(num_points * remove_percentage)
    indices = np.random.choice(
        range(1, num_points - 1), num_remove, replace=False
    ).astype(np.int32)
    return np.delete(stroke, indices, axis=0)


def downsample(arr, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1")
    if fraction == 1:
        return arr
    new_length = int(len(arr) * (1 - fraction))
    indices = np.linspace(0, len(arr) - 1, new_length, dtype=int)
    return arr[indices]


def get_time_string(fmt="%m%d_%H%M"):
    return datetime.now().strftime(fmt)


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None, args=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
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
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    steps = max(0, max_new_tokens - idx.size(1))
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
            logits[logits < v[:, [-1]]] = -float("Inf")
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


def save_samples(
    model, dataset, num=2, model_device="cpu", warmup_steps=100, do_sample=False
):
    """samples from the model and plots the decoded strokes"""
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
        # get the i'th row of sampled integers, as python list
        row = X_samp[i].detach().cpu().numpy()
        offset_samp = dataset.decode_stroke(row)
        point_samp = offsets_to_strokes(offset_samp)
        decoded_ascii = dataset.decode_text(context[i])

        # Plot the stroke
        fig, ax = plot_strokes(
            point_samp, f'Sample {i+1}: "{decoded_ascii}"'
        )  # plt.axis('off')
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
