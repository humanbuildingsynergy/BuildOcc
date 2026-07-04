"""
Script smoke tests.

Runs validate_strata and validate_signals in --mock mode to verify the full
OccupantAgent + SimulationEnvironment + ActivityScheduler stack end-to-end.
No API key required. Takes ~5–10 seconds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "scripts" / "experiments" / "outputs"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validate_strata_mock() -> None:
    r = _run(
        "scripts/validate_strata.py",
        "--mock", "--seeds", "1", "--days", "1", "--force",
    )
    assert r.returncode == 0, f"validate_strata failed:\n{r.stderr}"
    assert (OUT_DIR / "strata_actions.csv").exists()
    assert (OUT_DIR / "strata_summary.csv").exists()
    assert (OUT_DIR / "strata_metadata.json").exists()

    # Output should cover all 4 strata
    csv = (OUT_DIR / "strata_summary.csv").read_text()
    for stratum in ("O1", "O2", "O3", "O4"):
        assert stratum in csv, f"Stratum {stratum} missing from strata_summary.csv"


def test_validate_signals_mock() -> None:
    r = _run(
        "scripts/validate_signals.py",
        "--mock", "--seeds", "1", "--force",
    )
    assert r.returncode == 0, f"validate_signals failed:\n{r.stderr}"
    assert (OUT_DIR / "signal_responses.csv").exists()
    assert (OUT_DIR / "signal_summary.csv").exists()
    assert (OUT_DIR / "signals_metadata.json").exists()

    csv = (OUT_DIR / "signal_responses.csv").read_text()
    for sig in ("A", "B", "C"):
        assert sig in csv, f"Signal type {sig} missing from signal_responses.csv"


def test_validate_strata_produces_four_strata() -> None:
    r = _run(
        "scripts/validate_strata.py",
        "--mock", "--seeds", "1", "--days", "1", "--force",
    )
    assert r.returncode == 0
    import csv as csv_mod
    rows = list(csv_mod.DictReader((OUT_DIR / "strata_summary.csv").open()))
    strata_found = {row["stratum"] for row in rows}
    assert strata_found == {"O1", "O2", "O3", "O4"}


# ── Live API tests (require ANTHROPIC_API_KEY) ───────────────────────────────

def test_evaluate_one_day_live(tmp_path):
    """Run evaluate.py for one day with the real LLM. Skipped if no API key."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    r = _run(
        "scripts/evaluate.py",
        "--stratum", "O1",
        "--seed", "42",
        "--days", "1",
        "--no-atus-ref",
        "--llm-provider", "anthropic",
        "--output-dir", str(tmp_path),
        "--force",
    )
    assert r.returncode == 0, f"evaluate.py failed:\n{r.stderr}"
    outputs = list(tmp_path.glob("eval_*.json"))
    assert len(outputs) == 1, "Expected exactly one eval output file"
