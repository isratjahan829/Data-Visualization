"""Provenance, hashing and the QC count ledger.

Deliverable D1. The acceptance test the action report sets for the cohort is
arithmetic: "Input minus exclusions equals final cohort for every group."
``QCLedger.assert_reconciles`` enforces exactly that, so a cohort number can
never reach a table without an accounting trail behind it.

Relevant audit findings this module exists to prevent recurring:

* The reported 1,804,571-isolate cohort equals the *unfiltered* row total of
  the seven source CSVs, while Methods claim N50/contamination/completeness
  filters. Zero exclusions were recorded because none were applied
  (finding A1).
* The C. jejuni export is malformed and was read with
  ``on_bad_lines="skip"``, discarding an unrecorded number of rows
  (finding A2).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

__all__ = ["sha256_file", "QCLedger", "RunManifest", "read_csv_strict"]

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    """Stable hash of a dataframe's contents, for derived-table verification."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def read_csv_strict(path: str | Path, **kwargs) -> tuple[pd.DataFrame, dict]:
    """Read a CSV, refusing to discard malformed rows silently.

    On a tokenising error the function does **not** fall back to
    ``on_bad_lines="skip"``. It re-reads the file counting the physical lines
    and the rows a permissive parse would keep, then raises with both numbers,
    so the discrepancy becomes a documented exclusion decision instead of an
    invisible one.
    """
    path = Path(path)
    try:
        frame = pd.read_csv(path, **kwargs)
        return frame, {
            "path": str(path),
            "sha256": sha256_file(path),
            "n_rows": len(frame),
            "n_columns": frame.shape[1],
            "parser": "c",
            "rows_skipped": 0,
        }
    except pd.errors.ParserError as error:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            physical_lines = sum(1 for _ in handle) - 1  # minus header
        permissive = pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)
        raise ValueError(
            f"{path.name} is malformed: the strict parser failed ({error}). "
            f"The file holds {physical_lines} data lines; a permissive parse keeps "
            f"{len(permissive)}, so {physical_lines - len(permissive)} rows would be "
            "discarded. Fix the export or record this as an explicit exclusion rule "
            "in the QC ledger -- do not read it with on_bad_lines='skip'."
        ) from error


@dataclass
class QCLedger:
    """Input -> exclusions -> final, per cohort group, with a reconciliation test."""

    rows: list[dict] = field(default_factory=list)

    def add(
        self,
        group: str,
        *,
        n_input: int,
        exclusions: dict[str, int],
        n_final: int,
        source_sha256: str | None = None,
    ) -> None:
        self.rows.append(
            {
                "group": group,
                "n_input": int(n_input),
                "n_final": int(n_final),
                "n_excluded_total": int(sum(exclusions.values())),
                "exclusions": dict(exclusions),
                "source_sha256": source_sha256,
                "reconciles": int(n_input) - int(sum(exclusions.values())) == int(n_final),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        records = []
        for row in self.rows:
            record = {key: value for key, value in row.items() if key != "exclusions"}
            record.update({f"excluded__{rule}": n for rule, n in row["exclusions"].items()})
            records.append(record)
        return pd.DataFrame(records)

    def assert_reconciles(self) -> None:
        """Raise unless every group satisfies input - exclusions == final."""
        broken = [row["group"] for row in self.rows if not row["reconciles"]]
        if broken:
            raise AssertionError(
                "QC ledger does not reconcile for: "
                + ", ".join(broken)
                + ". Every cohort count must equal input minus recorded exclusions."
            )


@dataclass
class RunManifest:
    """Everything needed to re-run: inputs, versions, config, seeds, outputs."""

    run_id: str
    config: dict
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, config: dict) -> "RunManifest":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(run_id=stamp, config=dict(config))

    def add_input(self, path: str | Path, **extra) -> None:
        path = Path(path)
        self.inputs.append({"path": str(path), "sha256": sha256_file(path), **extra})

    def add_output(self, path: str | Path, **extra) -> None:
        path = Path(path)
        self.outputs.append({"path": str(path), "sha256": sha256_file(path), **extra})

    def environment(self) -> dict:
        packages = {}
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "xgboost",
            "lightgbm",
            "shap",
        ):
            try:
                from importlib.metadata import version

                packages[name] = version(name)
            except Exception:
                packages[name] = "not installed"
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "git_commit": _git_commit(),
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "environment": self.environment(),
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return path


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
