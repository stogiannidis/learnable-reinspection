"""Hugging Face Hub helpers for multi-process checkpoint loads.

Sharded ``safetensors`` trees on shared filesystems (NFS + DeepSpeed) often break
when every rank calls ``from_pretrained(repo_id)`` concurrently: non-zero ranks
can resolve the index before large shards are visible.  InternVL3 historically
fixed this by rank-0 ``snapshot_download`` then ``local_files_only=True`` for
everyone; we centralize that here for all backends.
"""
from __future__ import annotations

import os

import torch.distributed as dist


def resolve_pretrained_local_path(repo_or_path: str, *, repo_type: str = "model") -> str:
    """Return a path suitable for ``from_pretrained``.

    - If ``repo_or_path`` is already a local directory, return it unchanged.
    - If distributed is initialized with world size > 1 and the argument is a
      Hub id (not a directory), rank 0 downloads the snapshot; all ranks then
      resolve the same on-disk path with ``local_files_only=True`` so no rank
      hits partial shard metadata.
    - Otherwise return ``repo_or_path`` (single-process / single-GPU Hub load).
    """
    expanded = os.path.expanduser(repo_or_path)
    if os.path.isdir(expanded):
        return expanded

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return repo_or_path

    from huggingface_hub import snapshot_download

    if dist.get_rank() == 0:
        snapshot_download(repo_id=repo_or_path, repo_type=repo_type)
    dist.barrier()
    return snapshot_download(repo_id=repo_or_path, repo_type=repo_type, local_files_only=True)
