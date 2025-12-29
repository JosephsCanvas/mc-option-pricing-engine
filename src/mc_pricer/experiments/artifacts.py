"""Artifact generation for reproducible research experiments.

This module provides utilities to save experimental results with full metadata
including git commit, environment information, and timestamps for paper-ready
reproducibility.
"""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ArtifactMetadata:
    """Metadata for reproducible experiment artifacts.

    Attributes
    ----------
    timestamp : str
        ISO 8601 timestamp of experiment run.
    git_commit : str | None
        Git commit SHA if available.
    git_branch : str | None
        Git branch name if available.
    git_dirty : bool
        Whether working directory has uncommitted changes.
    python_version : str
        Python version string.
    numpy_version : str
        NumPy version string.
    platform_system : str
        Operating system (Linux, Windows, Darwin).
    platform_release : str
        OS release version.
    platform_machine : str
        Machine type (x86_64, arm64, etc.).
    """

    timestamp: str
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    python_version: str
    numpy_version: str
    platform_system: str
    platform_release: str
    platform_machine: str


def get_git_info() -> tuple[str | None, str | None, bool]:
    """Get git commit, branch, and dirty status.

    Returns
    -------
    tuple
        (commit_sha, branch_name, is_dirty)
    """
    try:
        # Get commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        commit = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        branch = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        is_dirty = bool(result.stdout.strip())

        return commit, branch, is_dirty
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None, None, False


def collect_metadata() -> ArtifactMetadata:
    """Collect full experiment metadata.

    Returns
    -------
    ArtifactMetadata
        Complete metadata for reproducibility.
    """
    commit, branch, dirty = get_git_info()

    return ArtifactMetadata(
        timestamp=datetime.now().isoformat(),
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
    )


def save_artifact(
    data: dict[str, Any],
    output_path: str | Path,
    include_metadata: bool = True,
) -> None:
    """Save experiment artifact as JSON with metadata.

    Parameters
    ----------
    data : dict
        Experiment data to save (results, config, etc.).
    output_path : str or Path
        Output file path.
    include_metadata : bool
        Whether to include environment metadata.
    """
    artifact = {"data": data}

    if include_metadata:
        metadata = collect_metadata()
        artifact["metadata"] = {
            "timestamp": metadata.timestamp,
            "git": {
                "commit": metadata.git_commit,
                "branch": metadata.git_branch,
                "dirty": metadata.git_dirty,
            },
            "environment": {
                "python_version": metadata.python_version,
                "numpy_version": metadata.numpy_version,
                "platform": {
                    "system": metadata.platform_system,
                    "release": metadata.platform_release,
                    "machine": metadata.platform_machine,
                },
            },
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)


def format_summary_table(
    headers: list[str],
    rows: list[list[Any]],
    title: str | None = None,
) -> str:
    """Format data as markdown table for paper-ready output.

    Parameters
    ----------
    headers : list[str]
        Column headers.
    rows : list[list]
        Data rows.
    title : str, optional
        Table title.

    Returns
    -------
    str
        Markdown formatted table.
    """
    lines = []

    if title:
        lines.append(f"\n### {title}\n")

    # Header
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    # Separator
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    # Rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

    return "\n".join(lines)
