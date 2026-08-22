"""Absolute-path validation for openpi_rtc entry points.

Every data path (checkpoint, raw HDF5 dir/file) must be an absolute path and
must exist *before* any heavy work (model load, jax import, dataset convert)
starts. This keeps the package machine-independent: nothing is derived from
the current working directory or the repo layout.
"""

from __future__ import annotations

import os


def _fmt(path) -> str:
    return repr(os.fspath(path)) if path is not None else "<未指定>"


def require_abs_dir(path, label: str, *, contains=()) -> str:
    """Absolute directory that exists (optionally with required subdirs)."""
    p = os.fspath(path) if path is not None else ""
    if not p:
        raise SystemExit(f"ERROR: {label} 未指定（需要绝对路径）")
    if not os.path.isabs(p):
        raise SystemExit(f"ERROR: {label} 必须是绝对路径: {_fmt(p)}")
    if not os.path.isdir(p):
        raise SystemExit(f"ERROR: {label} 不存在: {_fmt(p)}")
    for name in contains:
        if not os.path.isdir(os.path.join(p, name)):
            raise SystemExit(f"ERROR: {label} 缺少 {name}/ 子目录: {_fmt(p)}")
    return p


def require_checkpoint(path) -> str:
    """Checkpoint dir with params/ + assets/ (Orbax checkpoint layout)."""
    return require_abs_dir(path, "checkpoint", contains=("params", "assets"))


def require_hdf5_dir(path, label: str) -> str:
    """Absolute dir that exists and contains at least one .hdf5 file."""
    p = require_abs_dir(path, label)
    files = [f for f in os.listdir(p) if f.endswith(".hdf5")]
    if not files:
        raise SystemExit(f"ERROR: {label} 下没有 .hdf5 文件: {_fmt(p)}")
    return p


def require_dataset(path, label: str) -> str:
    """Absolute .hdf5 file, or absolute dir containing .hdf5 files."""
    if path is None or not os.path.isabs(os.fspath(path)):
        raise SystemExit(f"ERROR: {label} 必须是绝对路径: {_fmt(path)}")
    p = os.fspath(path)
    if os.path.isdir(p):
        return require_hdf5_dir(p, label)
    if os.path.isfile(p):
        if not p.endswith(".hdf5"):
            raise SystemExit(f"ERROR: {label} 不是 .hdf5 文件: {_fmt(p)}")
        return p
    raise SystemExit(f"ERROR: {label} 不存在: {_fmt(p)}")
