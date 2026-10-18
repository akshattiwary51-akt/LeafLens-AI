"""Portable manifest path handling.

The committed manifests store Windows-style paths
(``data\\preprocessed\\...``) that do not resolve on Linux/macOS.
"""
from pathlib import Path, PureWindowsPath


def resolve_image_path(raw, root=None):
    """Convert a manifest path (Windows or POSIX separators) to a Path."""
    p = Path(*PureWindowsPath(str(raw)).parts)
    return Path(root) / p if root is not None and not p.is_absolute() else p
