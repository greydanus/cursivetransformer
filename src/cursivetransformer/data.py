import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import (
    downsample,
    generate_word_combos,
    horizontal_shear,
    load_and_parse_data,
    remove_random_points,
    strokes_to_offsets,
)


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
        self.strokes = (
            strokes  # List of Nx3 arrays, each representing a cursive sentence
        )
        self.texts = texts  # List of corresponding text strings
        self.chars = chars  # String of all possible characters
        self.augment = augment

        self.theta_bins = np.linspace(-np.pi, np.pi, 201)  # 100 bins for theta

        r_bins_pen_down = np.concatenate(
            [
                np.asarray([0]),
                np.linspace(0.005, 0.050, 50),  # Close around 0.01, 30 bins
                np.geomspace(0.051, 4, 81)[:-1],  # 150 exponential bins
            ]
        )
        r_bins_pen_up = (
            r_bins_pen_down + max(r_bins_pen_down) + 1
        )  # Offset for pen-up states
        self.r_bins = np.concatenate([r_bins_pen_down, r_bins_pen_up])

        self.feature_sizes = [len(self.r_bins), len(self.theta_bins)]
        self.cumulative_sizes = np.cumsum([0] + self.feature_sizes)

        # Add special tokens for strokes
        self.PAD_TOKEN = sum(self.feature_sizes)
        self.END_TOKEN = sum(self.feature_sizes) + 1

        # Character tokenization
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.char_PAD_TOKEN = 0

        self.max_seq_length = max_seq_length
        self.max_text_length = max_text_length

    def augment_stroke(self, stroke):

        stroke = remove_random_points(
            stroke, remove_percentage=0.01
        )  # Drop some points
        stroke = horizontal_shear(stroke, shear_range=(-0.33, 0.15))  # Horizontal shear

        stroke[:, 0:1] *= np.random.uniform(0.95, 1.05)
        stroke[:, 1:2] *= np.random.uniform(0.95, 1.05)

        # noise = np.random.normal(0, 0.001, stroke[:, :2].shape) # Random noise
        # stroke[:, :2] += noise

        angle = np.random.uniform(-0.08, 0.08)  # Random rotation
        rad = np.deg2rad(angle)
        rotation_matrix = np.array(
            [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]
        )
        stroke[:, :2] = np.dot(stroke[:, :2], rotation_matrix.T)

        # Random starting point
        # stroke = stroke[np.random.randint(1, 2):-np.random.randint(1, 2)]

        # Downsample stroke
        stroke[1:, 2:3] *= stroke[:-1, 2:3]  # pen_up will now always come in sets of 3+
        stroke[2:, 2:3] *= stroke[:-2, 2:3]
        stroke[3:, 2:3] *= stroke[:-3, 2:3]
        stroke = downsample(stroke, 0.63)  # stroke[::2]
        return stroke

    def __len__(self):
        return len(self.strokes)

    def get_vocab_size(self):
        return sum(self.feature_sizes) + 2  # +2 for PAD and END tokens

    def get_char_vocab_size(self):
        return len(self.chars) + 1  # +1 for PAD token

    def get_stroke_seq_length(self):
        return self.max_seq_length

    def get_text_seq_length(self):
        return self.max_text_length

    def encode_stroke(self, stroke):
        # Encode magnitude and pen state together
        r_idx = np.digitize(stroke[:, 0], self.r_bins[: len(self.r_bins) // 2]) - 1
        r_idx[stroke[:, 2] == 0] += len(self.r_bins) // 2  # Offset for pen-up states

        theta_idx = np.digitize(stroke[:, 1], self.theta_bins) - 1

        encoded = np.column_stack(
            [
                theta_idx + self.cumulative_sizes[1],
                r_idx + self.cumulative_sizes[0],
            ]
        )
        return encoded.flatten()

    def decode_stroke(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()

        # Remove PAD and END tokens
        ix = ix[(ix != self.PAD_TOKEN) & (ix != self.END_TOKEN)]

        # Reshape the flattened array back to Nx2
        ix = ix[: (len(ix) // 2) * 2]
        ix = ix.reshape(-1, 2)

        r_idx = ix[:, 1] - self.cumulative_sizes[0]
        pen = (r_idx < len(self.r_bins) // 2).astype(int)
        r_idx[pen == 0] -= len(self.r_bins) // 2
        r = self.r_bins[: len(self.r_bins) // 2][
            r_idx.clip(0, len(self.r_bins) // 2 - 1)
        ]
        theta = self.theta_bins[
            (ix[:, 0] - self.cumulative_sizes[1]).clip(0, len(self.theta_bins) - 1)
        ]

        return np.column_stack([r, theta, pen])

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

        # Encode stroke
        stroke_offsets = strokes_to_offsets(stroke)
        encoded_stroke = self.encode_stroke(stroke_offsets)
        x = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)
        y = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)

        seq_len = min(
            len(encoded_stroke), self.max_seq_length - 1
        )  # -1 to leave room for END token
        x[:seq_len] = torch.tensor(encoded_stroke[:seq_len], dtype=torch.long)
        x[seq_len] = self.END_TOKEN

        y[:seq_len] = x[1 : seq_len + 1]
        y[seq_len] = self.END_TOKEN

        # Encode text (context) and pad to max_text_length of 30
        encoded_text = self.encode_text(text)
        c = torch.full((self.max_text_length,), self.char_PAD_TOKEN, dtype=torch.long)
        text_len = min(len(encoded_text), self.max_text_length)
        c[:text_len] = encoded_text[:text_len]

        return x, c, y


def create_datasets(augment=True, max_seq_length=1100, num_words=3):
    np.random.seed(0)
    torch.manual_seed(0)
    data = load_and_parse_data()

    # partition the input data into a training and the test set
    test_set_size = min(
        1000, int(len(data) * 0.05)
    )  # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(data)).tolist()

    train_examples = generate_word_combos(
        [data[i] for i in rp[:-test_set_size]],
        desired_num_combos=249000,
        num_words=num_words,
    )
    train_examples = [
        train_examples[i] for i in torch.randperm(len(train_examples)).tolist()
    ]

    test_examples = generate_word_combos(
        [data[i] for i in rp[-test_set_size:]],
        desired_num_combos=1000,
        num_words=num_words,
    )
    test_examples = [
        test_examples[i] for i in torch.randperm(len(test_examples)).tolist()
    ]

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
    print(f'\t"{chars}"')
    print(
        f"Split up the dataset into {len(train_examples)} training examples and {len(test_examples)} test examples"
    )

    # wrap in dataset objects
    train_dataset = StrokeDataset(
        train_strokes, train_texts, chars, max_seq_length, name="train", augment=augment
    )
    test_dataset = StrokeDataset(
        test_strokes, test_texts, chars, max_seq_length, name="test", augment=augment
    )
    return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=int(1e10)
        )
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except (
            StopIteration
        ):  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
