"""Saving and loading problem instances.

An instance is stored as one .npz file: every numpy array of the instance is stored under its own
name, and all scalar parameters are stored as one JSON string under the key "meta".
"""
import json

import numpy as np


def save_npz(path, arrays, meta):
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: data[k] for k in data.files if k != "meta"}
    return arrays, meta


def to_numpy(x):
    """Convert a reported solution to numpy: arrays, PyTorch tensors, and dicts of them."""
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if hasattr(x, "detach"):  # PyTorch tensor
        x = x.detach().cpu().numpy()
    x = np.array(x, copy=True)
    return x if np.iscomplexobj(x) else x.astype(float)
