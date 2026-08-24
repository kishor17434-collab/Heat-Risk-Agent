from __future__ import annotations

from pathlib import Path
from typing import Iterable


class ProjectPreflightError(RuntimeError):
    """Raised when required project artifacts are missing."""


def validate_required_paths(
    required: Iterable[tuple[str, str | Path]],
    *,
    project_root: str | Path | None = None,
) -> list[Path]:
    """
    Validate that each required file exists and return resolved Path objects.

    Parameters
    ----------
    required:
        Sequence of (label, path) tuples. Relative paths are resolved against
        project_root if provided; otherwise they are resolved from the repo root.
    project_root:
        Override the base directory used to resolve relative paths.

    Returns
    -------
    list[Path]
        Resolved absolute paths for the validated files.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    resolved: list[Path] = []
    missing: list[str] = []

    for label, raw_path in required:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (root / candidate).absolute()
        else:
            candidate = candidate.absolute()

        resolved.append(candidate)
        if not candidate.exists():
            missing.append(f"- {label}: {candidate}")

    if missing:
        details = "\n".join(missing)
        raise ProjectPreflightError(
            "Project preflight failed. Missing required files:\n"
            f"{details}\n\n"
            "Run the project pipeline first: python scripts/run_pipeline.py\n"
            "Then train the model: python scripts/run_train.py"
        )

    return resolved
