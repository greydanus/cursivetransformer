# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, json, pickle, zipfile, functools, copy
import numpy as np
from math import comb

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


########## LOADING DATA AND COMBINING WORDS ##########

@functools.lru_cache(maxsize=5)
def load_and_parse_data(dataset_name):
    file_path = f'{CURRENT_DIR}/data/{dataset_name}.json.zip'
    print(f'Trying to load dataset file from {file_path}')

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        json_filename = zip_ref.namelist()[0]
        with zip_ref.open(json_filename) as file:
            data = json.load(file)

    for item in data:
        strokes = np.array(item['points'])
        strokes[:, 0] *= item['metadata']['aspectRatio']
        strokes[:, 0] -= strokes[0, 0]
        strokes[:, 1] -= 0.65
        item['points'] = strokes
    print(f'Succeeded in loading the {dataset_name} dataset; contains {len(data)} items.')
    return data
    
def combine_handwriting_examples(examples):
    return {
        'metadata': {
            'author': examples[0]['metadata']['author'],
            'asciiSequence': ' '.join(ex['metadata']['asciiSequence'] for ex in examples),
            'pointCount': sum(ex['metadata']['pointCount'] for ex in examples),
            'strokeCount': sum(ex['metadata']['strokeCount'] for ex in examples),
            'aspectRatio': examples[0]['metadata']['aspectRatio']
        },
        'points': [ex['points'].copy() for ex in examples]
    }

def generate_word_combos(raw_json, desired_num_combos=10000, num_words=3):
    num_combos = comb(len(raw_json), num_words)
    print(f'For a dataset of {len(raw_json)} examples we can generate {num_combos} combinations of {num_words} examples.')
    print(f'Generating {desired_num_combos} random combinations.')
    combo_json = []
    for _ in range(desired_num_combos):
        ixs = np.random.choice(len(raw_json), size=num_words, replace=False)
        examples_to_merge = [raw_json[ix] for ix in ixs]
        combo_json.append(combine_handwriting_examples(examples_to_merge))
    return combo_json

def word_offsets_to_points(word_offsets, space_width=0.17):
    word_points = []
    last_point = None
    
    for i, offsets in enumerate(word_offsets):
        points = offsets_to_strokes(offsets)
        if last_point is not None:
            lp = last_point[np.newaxis, :]
            points = points + lp

        if len(points) > 0:
          last_point = points[-1]
          last_point[0] = last_point[0] + space_width
          last_point[-1] = 0
        word_points.append(points)
    
    return np.vstack(word_points)


########## TOKENIZATION, AUGMENTATION, AND DATA IO ##########


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

def strokes_to_offsets(points, prev_points=None):
    offsets = np.zeros_like(points)
    offsets[1:, 0:2] = np.diff(points[:, 0:2], axis=0)  # Same dx, dy computation
    
    if prev_points is not None:
        offsets[0, 1] = points[0, 1] - prev_points[-1, 1]
        offsets[0, 0] = (prev_points[:, 0].max() - prev_points[-1, 0]) + \
                        (points[0, 0] - points[:, 0].min())
    
    offsets[:, 2] = points[:, 2]
    return decompose_offsets(offsets)

def offsets_to_strokes(offsets_dec):
    # Calculate cumulative sums over (dx, dt) to get absolute pen positions
    offsets = reconstruct_offsets(offsets_dec)

    absolute_coords = np.cumsum(offsets[:, :2], axis=0)  # just over (dx, dy) dimensions
    stroke_data = np.hstack((absolute_coords, offsets[:, 2:3]))
    return stroke_data

def random_horizontal_shear(stroke, shear_range=(-0.4, 0.4)):
    shear_factor = np.random.uniform(*shear_range)
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    stroke[:, :2] = np.dot(stroke[:, :2], shear_matrix.T)
    return stroke

def random_rotate(stroke, angle_range=(-.08, .08)):
    angle = np.random.uniform(*angle_range)
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]])
    stroke[:, :2] = np.dot(stroke[:, :2], rotation_matrix.T)
    return stroke

def downsample(arr, fraction):
    if not 0 <= fraction <= 1: raise ValueError("Fraction must be between 0 and 1")
    if fraction == 1: return arr
    result, stroke = [], []
    for point in arr:
        if point[2] == 1:
            stroke.append(point)
        else:
            if stroke:
                new_len = max(2, int(len(stroke) * (1 - fraction)))
                indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
                result.extend(np.array(stroke)[indices])
            result.append(point)
            stroke = []
    if stroke:
        new_len = max(2, int(len(stroke) * (1 - fraction)))
        indices = np.linspace(0, len(stroke) - 1, new_len, dtype=int)
        result.extend(np.array(stroke)[indices])
    return np.array(result)


class StrokeDataset(Dataset):
    def __init__(self, raw_word_strokes, texts, args, max_text_length=50, name=''):
        self.raw_word_strokes = raw_word_strokes  # List of lists of Nx3 arrays, each inner list representing words in a sentence
        self.texts = texts      # List of corresponding text strings
        self.args = args
        self.alphabet = args.alphabet  # String of all possible characters
        self.augment = args.augment
        self.max_seq_length = args.max_seq_length
        self.max_text_length = max_text_length
        self.name = name
        self.counter = 0

        self.theta_bins = np.linspace(-np.pi, np.pi, 220)

        r_bins_pen_down = np.concatenate([
                            np.asarray([0]),
                            np.linspace(0.0001, 0.060, 25),
                            np.geomspace(0.06001, 0.75, 90) ]) # 100 discrete radii
        r_bins_pen_up = r_bins_pen_down + max(r_bins_pen_down) + 1  # Offset for pen-up states
        self.r_bins = np.concatenate([r_bins_pen_down, r_bins_pen_up])  # 200 bins for: {radii x pen up/down}

        self.feature_sizes = [len(self.r_bins), len(self.theta_bins)]
        self.cumulative_sizes = np.cumsum([0] + self.feature_sizes)

        # Add special tokens for strokes
        self.PAD_TOKEN = sum(self.feature_sizes)
        self.END_TOKEN = sum(self.feature_sizes) + 1
        self.WORD_TOKEN = sum(self.feature_sizes) + 2

        # Character tokenization
        self.char_PAD_TOKEN = 0
        self.stoi = {ch:i+1 for i,ch in enumerate(self.alphabet)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def split_by_word_tokens(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        # Find pairs of WORD_TOKENs
        word_boundaries = np.where((tokens[:-1] == self.WORD_TOKEN) & (tokens[1:] == self.WORD_TOKEN))[0]
        # Split using these boundaries
        splits = np.split(tokens, word_boundaries + 1)
        return [s for s in splits if len(s) > 0]

    def concat_with_word_tokens(self, token_lists):
        word_tokens = np.array([self.WORD_TOKEN, self.WORD_TOKEN])
        return np.concatenate([np.concatenate([tokens, word_tokens]) if i < len(token_lists)-1 else tokens 
                             for i, tokens in enumerate(token_lists)])

    def augment_stroke(self, stroke):
        stroke = random_horizontal_shear(stroke, shear_range=(-0.30, 0.15)) # Horizontal shear
        stroke[:, 0:1] *= np.random.uniform(0.9, 1.1)
        stroke[:, 1:2] *= np.random.uniform(0.9, 1.1)
        stroke = random_rotate(stroke, angle_range=(-.08, .08))

        downsample_percent = self.args.downsample_mean + self.args.downsample_width * (np.random.rand()-.5)
        stroke = downsample(stroke, downsample_percent)
        return stroke

    def __len__(self):
        return len(self.raw_word_strokes)

    def get_vocab_size(self):
        return sum(self.feature_sizes) + 3  # +3 for PAD, END, and WORD tokens

    def get_char_vocab_size(self):
        return len(self.alphabet) + 1  # +1 for PAD token

    def get_stroke_seq_length(self):
        return self.max_seq_length

    def get_text_seq_length(self):
        return self.max_text_length

    def encode_stroke(self, stroke):
        # Encode magnitude and pen state together
        r_idx = np.digitize(stroke[:, 0], self.r_bins[:len(self.r_bins)//2]) - 1
        r_idx[stroke[:, 2] == 0] += len(self.r_bins) // 2  # Offset for pen-up states

        theta_idx = np.digitize(stroke[:, 1], self.theta_bins) - 1

        encoded = np.column_stack([
            theta_idx + self.cumulative_sizes[1],
            r_idx + self.cumulative_sizes[0],])
        return encoded.flatten()

    def decode_stroke(self, ix):
      ix_list = self.split_by_word_tokens(ix)
      return [self.decode_word_strokes(ix) for ix in ix_list]

    def decode_word_strokes(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()

        # Remove PAD, END, and WORD tokens
        ix = ix[(ix != self.PAD_TOKEN) & (ix != self.END_TOKEN) & (ix != self.WORD_TOKEN)]

        # Reshape the flattened array back to Nx2
        ix = ix[:(len(ix)//2)*2]
        ix = ix.reshape(-1, 2)

        r_idx = ix[:, 1] - self.cumulative_sizes[0]
        pen = (r_idx < len(self.r_bins) // 2).astype(int)
        r_idx[pen == 0] -= len(self.r_bins) // 2
        r = self.r_bins[:len(self.r_bins)//2][r_idx.clip(0, len(self.r_bins)//2 - 1)]
        theta = self.theta_bins[(ix[:, 0] - self.cumulative_sizes[1]).clip(0, len(self.theta_bins)-1)]

        return np.column_stack([r, theta, pen])

    def encode_text(self, text, do_padding=True):
        encoded_text = torch.tensor([self.stoi.get(ch, self.char_PAD_TOKEN) for ch in text], dtype=torch.long)
        if do_padding:
            c = torch.full((self.max_text_length,), self.char_PAD_TOKEN, dtype=torch.long)
            text_len = min(len(encoded_text), self.max_text_length)
            c[:text_len] = encoded_text[:text_len]
        else:
            c = encoded_text
        return c

    def decode_text(self, ix):
        if isinstance(ix, torch.Tensor):
            ix = ix.cpu().numpy()
        
        first_pad = np.where(ix == self.char_PAD_TOKEN)[0]
        end_idx = first_pad[0] if len(first_pad) > 0 else len(ix)
        return ''.join(self.itos.get(i, '') for i in ix[:end_idx])

    def __getitem__(self, idx):
        word_strokes = self.raw_word_strokes[idx]
        text = self.texts[idx]

        # Apply augmentation per word if enabled
        if self.augment:
            np.random.seed(self.args.seed+idx+self.counter)  # use the same augmentation across all words in sample
            word_strokes = [self.augment_stroke(word.copy()) for word in word_strokes]
        self.counter = (self.counter + 1) % 100000

        # Encode each word separately and combine with WORD_TOKENs
        encoded_words = [self.encode_stroke(
                         strokes_to_offsets(word_strokes[i],
                         prev_points=word_strokes[i-1] if i > 0 else None))
                              for i in range(len(word_strokes)) ]
        encoded_stroke = self.concat_with_word_tokens(encoded_words)

        # Create input and target sequences
        x = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)
        y = torch.full((self.max_seq_length,), self.PAD_TOKEN, dtype=torch.long)

        seq_len = min(len(encoded_stroke), self.max_seq_length - 1)  # -1 to leave room for END token
        x[:seq_len] = torch.tensor(encoded_stroke[:seq_len], dtype=torch.long)
        x[seq_len] = self.END_TOKEN

        y[:seq_len] = x[1:seq_len+1]
        y[seq_len] = self.END_TOKEN

        c = self.encode_text(text)
        return x, c, y


def create_datasets(args):
  np.random.seed(args.seed) ; torch.manual_seed(args.seed)
  data = load_and_parse_data(args.dataset_name)

  # partition the input data into a training and the test set
  test_set_size = min(1000, max(10, int(len(data) * 0.05))) # between 10 and 1000 examples: ideally 10% of dataset
  rp = torch.randperm(len(data)).tolist()

  train_examples = generate_word_combos([data[i] for i in rp[:-test_set_size]], desired_num_combos=args.train_size, num_words=args.num_words)
  train_examples = [train_examples[i] for i in torch.randperm(len(train_examples)).tolist()]

  test_examples = generate_word_combos([data[i] for i in rp[-test_set_size:]], desired_num_combos=args.test_size, num_words=args.num_words)
  test_examples = [test_examples[i] for i in torch.randperm(len(test_examples)).tolist()]

  train_word_strokes = [[copy.deepcopy(stroke) for stroke in v['points']] for v in train_examples]
  train_texts = [copy.deepcopy(v['metadata']['asciiSequence']) for v in train_examples]

  test_word_strokes = [[copy.deepcopy(stroke) for stroke in v['points']] for v in test_examples]
  test_texts = [copy.deepcopy(v['metadata']['asciiSequence']) for v in test_examples]

  print(f"Number of examples in the train dataset: {len(train_examples)}")
  print(f"Number of examples in the test dataset: {len(test_examples)}")
  print(f"Average number of words per example: {np.mean([len(strokes) for strokes in train_word_strokes]):.1f}")
  print(f"Max token sequence length: {args.max_seq_length}")
  print(f"Number of unique characters in the ascii vocabulary: {len(args.alphabet)}")
  print("Ascii vocabulary:")
  print(f'\t"{args.alphabet}"')
  print(f"Split up the dataset into {len(train_examples)} training examples and {len(test_examples)} test examples")

  # wrap in dataset objects
  train_dataset = StrokeDataset(train_word_strokes, train_texts, args, name='train')
  test_dataset = StrokeDataset(test_word_strokes, test_texts, args, name='test')
  return train_dataset, test_dataset


class InfiniteDataLoader:
    """
    From Andrej Karpathy: this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:  # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch