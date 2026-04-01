"""Grid generation, dataset, and dataloaders."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import Vocab, COLORS, SHAPES, RELATIONS, get_vocab
from .config import Config

# Fixed lengths (derived from vocabulary structure)
MAX_Q_LEN = 11   # type A = 11 tokens; type B = 5, padded to 11
MAX_ANS_LEN = 5  # type B = 5 tokens; type A = 2, padded to 5


def _check_relation(rel, r1, c1, r2, c2):
    """Return 'yes'/'no' for obj1 [rel] obj2."""
    if rel == 'left':
        return 'yes' if c1 < c2 else 'no'
    if rel == 'right':
        return 'yes' if c1 > c2 else 'no'
    if rel == 'above':
        return 'yes' if r1 < r2 else 'no'  # row 0 = top
    if rel == 'below':
        return 'yes' if r1 > r2 else 'no'
    raise ValueError(rel)


def _generate_grid(rng, n_colors=6, n_shapes=4, grid_size=8, min_obj=2, max_obj=5):
    """Returns (color_grid, shape_grid, objects).

    color_grid/shape_grid: int arrays of length grid_size^2.
    Empty cells use n_colors / n_shapes as the "empty" embedding index.
    objects: list of dicts with keys row, col, color, shape, cell.
    """
    n_cells = grid_size ** 2
    n_obj = rng.randint(min_obj, max_obj + 1)

    # Unique (color, shape) combinations
    n_combos = n_colors * n_shapes
    combo_ids = rng.choice(n_combos, n_obj, replace=False)
    colors = combo_ids // n_shapes
    shapes = combo_ids % n_shapes

    # Unique cell positions
    cells = rng.choice(n_cells, n_obj, replace=False)

    color_grid = np.full(n_cells, n_colors, dtype=np.int64)   # empty = n_colors
    shape_grid = np.full(n_cells, n_shapes, dtype=np.int64)   # empty = n_shapes

    objects = []
    for cell, color, shape in zip(cells, colors, shapes):
        color_grid[cell] = int(color)
        shape_grid[cell] = int(shape)
        objects.append({
            'row': int(cell // grid_size),
            'col': int(cell % grid_size),
            'color': int(color),
            'shape': int(shape),
            'cell': int(cell),
        })
    return color_grid, shape_grid, objects


def generate_sample(rng, vocab: Vocab, config: Config, q_type=None):
    """Generate one labeled sample dict."""
    color_grid, shape_grid, objects = _generate_grid(
        rng,
        n_colors=config.n_colors,
        n_shapes=config.n_shapes,
        grid_size=config.grid_size,
        min_obj=config.min_objects,
        max_obj=config.max_objects,
    )

    if q_type is None:
        q_type = 'A' if (rng.rand() < 0.5 and len(objects) >= 2) else 'B'
    if q_type == 'A' and len(objects) < 2:
        q_type = 'B'

    if q_type == 'A':
        idx1, idx2 = rng.choice(len(objects), 2, replace=False)
        obj1, obj2 = objects[idx1], objects[idx2]
        rel_idx = rng.randint(len(RELATIONS))
        rel = RELATIONS[rel_idx]
        answer = _check_relation(rel, obj1['row'], obj1['col'], obj2['row'], obj2['col'])

        # "is the [c1] [s1] to the [rel] of the [c2] [s2]"
        question = [
            'is', 'the', COLORS[obj1['color']], SHAPES[obj1['shape']],
            'to', 'the', rel, 'of',
            'the', COLORS[obj2['color']], SHAPES[obj2['shape']],
        ]  # always 11 tokens
        ans_tokens = [answer]
        target_cells = [obj1['cell'], obj2['cell']]

    else:  # Type B
        obj = objects[rng.randint(len(objects))]
        question = ['where', 'is', 'the', COLORS[obj['color']], SHAPES[obj['shape']]]
        # Pad type B question to MAX_Q_LEN=11
        question = question + ['<PAD>'] * (MAX_Q_LEN - len(question))
        ans_tokens = ['row', str(obj['row']), 'col', str(obj['col'])]
        target_cells = [obj['cell']]

    # Encode question (type A already 11 tokens; type B padded above)
    q_ids = vocab.encode(question)  # length MAX_Q_LEN

    # ans_input:  [BOS, a0, a1, ...] padded to MAX_ANS_LEN
    # ans_target: [a0, a1, ..., EOS]  padded to MAX_ANS_LEN
    ans_enc = vocab.encode(ans_tokens)
    ans_input = [vocab.BOS] + ans_enc
    ans_target = ans_enc + [vocab.EOS]
    while len(ans_input) < MAX_ANS_LEN:
        ans_input.append(vocab.PAD)
    while len(ans_target) < MAX_ANS_LEN:
        ans_target.append(vocab.PAD)

    return {
        'color_ids': color_grid,
        'shape_ids': shape_grid,
        'question_ids': np.array(q_ids, dtype=np.int64),
        'ans_input_ids': np.array(ans_input, dtype=np.int64),
        'ans_target_ids': np.array(ans_target, dtype=np.int64),
        'target_cells': target_cells,   # list of ints
        'q_type': q_type,               # 'A' or 'B'
    }


def generate_dataset(n_samples, seed, config, vocab):
    rng = np.random.RandomState(seed)
    samples = [generate_sample(rng, vocab, config) for _ in range(n_samples)]
    return samples


class GridDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Convert list of sample dicts to batched tensors."""
    color_ids = torch.tensor(np.stack([s['color_ids'] for s in batch]), dtype=torch.long)
    shape_ids = torch.tensor(np.stack([s['shape_ids'] for s in batch]), dtype=torch.long)
    question_ids = torch.tensor(np.stack([s['question_ids'] for s in batch]), dtype=torch.long)
    ans_input_ids = torch.tensor(np.stack([s['ans_input_ids'] for s in batch]), dtype=torch.long)
    ans_target_ids = torch.tensor(np.stack([s['ans_target_ids'] for s in batch]), dtype=torch.long)
    target_cells = [s['target_cells'] for s in batch]
    q_types = [s['q_type'] for s in batch]
    return {
        'color_ids': color_ids,
        'shape_ids': shape_ids,
        'question_ids': question_ids,
        'ans_input_ids': ans_input_ids,
        'ans_target_ids': ans_target_ids,
        'target_cells': target_cells,
        'q_types': q_types,
    }


def get_dataloaders(config: Config, vocab: Vocab, cache=None):
    """Generate (or reuse cached) train/val/test dataloaders."""
    if cache is not None and all(k in cache for k in ('train', 'val', 'test')):
        splits = cache
    else:
        splits = {
            'train': generate_dataset(config.n_train, seed=0, config=config, vocab=vocab),
            'val':   generate_dataset(config.n_val,   seed=1, config=config, vocab=vocab),
            'test':  generate_dataset(config.n_test,  seed=2, config=config, vocab=vocab),
        }
        if cache is not None:
            cache.update(splits)

    train_loader = DataLoader(
        GridDataset(splits['train']),
        batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        GridDataset(splits['val']),
        batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        GridDataset(splits['test']),
        batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    return train_loader, val_loader, test_loader, splits
