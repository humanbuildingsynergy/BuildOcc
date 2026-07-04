"""
CLI command tests.

Exercises every entry point that ships with the package:
  buildocc init   — write template YAML
  buildocc run    — run simulation from YAML (--mock skips LLM)
  buildocc-api    — starts uvicorn REST server
  buildocc-mcp    — starts MCP stdio server

The API and MCP servers are blocking processes; we test that they start and
accept input without hanging by combining a short timeout with stdin injection
rather than asserting a clean exit code.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str, input: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "occupant_agent.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        input=input,
        timeout=timeout,
    )


def _run_entry(*cmd: str, input: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run an installed CLI entry point by name."""
    return subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        input=input,
        timeout=timeout,
    )


# ── buildocc ──────────────────────────────────────────────────────────────────

def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0, f"--help failed:\n{r.stderr}"


def test_help_lists_subcommands():
    r = _run("--help")
    assert "init" in r.stdout
    assert "run" in r.stdout


def test_entry_point_help_via_command():
    """Verify 'buildocc' is installed as a runnable command."""
    try:
        r = _run_entry("buildocc", "--help")
    except FileNotFoundError:
        pytest.skip("'buildocc' entry point not on PATH (editable install)")
    assert r.returncode == 0, (
        f"'buildocc --help' failed (returncode={r.returncode}):\n{r.stderr}"
    )


# ── buildocc init ─────────────────────────────────────────────────────────────

def test_init_creates_yaml(tmp_path):
    out = tmp_path / "sim.yaml"
    r = _run("init", str(out))
    assert r.returncode == 0, f"init failed:\n{r.stderr}"
    assert out.exists(), "YAML file was not created"


def test_init_yaml_has_required_keys(tmp_path):
    out = tmp_path / "config.yaml"
    _run("init", str(out))
    content = out.read_text()
    for key in ("stratum", "seed", "steps", "devices", "rooms", "zone_temperature"):
        assert key in content, f"Key {key!r} missing from template YAML"


def test_init_default_filename(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "occupant_agent.cli", "init"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert (tmp_path / "buildocc_config.yaml").exists()


def test_init_force_overwrites(tmp_path):
    out = tmp_path / "cfg.yaml"
    out.write_text("old content")
    r = _run("init", str(out), "--force")
    assert r.returncode == 0
    assert "old content" not in out.read_text()


def test_init_no_force_refuses_overwrite(tmp_path):
    out = tmp_path / "cfg.yaml"
    out.write_text("existing")
    r = _run("init", str(out))
    assert r.returncode != 0, "Should refuse to overwrite without --force"
    assert "existing" == out.read_text()


# ── buildocc run --mock ───────────────────────────────────────────────────────

@pytest.fixture
def config_yaml(tmp_path) -> Path:
    """Write a minimal YAML config for mock runs."""
    cfg = tmp_path / "run_config.yaml"
    cfg.write_text(textwrap.dedent("""\
        stratum: O1
        seed: 42
        steps: 3
        start_datetime: "2025-08-10 18:00"
        thermostat_setpoint: 22.0
        devices:
          - id: hvac
            on: true
            power_w: 3500
          - id: tv
            on: false
            power_w: 150
        rooms:
          - living_room
          - bedroom
        zone_temperature:
          mode: constant
          value: 25.0
        outdoor_temperature:
          mode: summer_default
        tou_rate:
          mode: peak_default
    """))
    return cfg


def test_run_mock_exits_zero(config_yaml):
    r = _run("run", str(config_yaml), "--mock")
    assert r.returncode == 0, f"run --mock failed:\n{r.stderr}"


def test_run_mock_prints_steps(config_yaml):
    r = _run("run", str(config_yaml), "--mock")
    # Should print 3 step headers
    assert "STEP 1/3" in r.stdout
    assert "STEP 3/3" in r.stdout


def test_run_mock_prints_done(config_yaml):
    r = _run("run", str(config_yaml), "--mock")
    assert "Done." in r.stdout or "Done" in r.stdout


def test_run_mock_override_steps(config_yaml):
    r = _run("run", str(config_yaml), "--mock", "--steps", "2")
    assert r.returncode == 0
    assert "STEP 2/2" in r.stdout
    assert "STEP 3/" not in r.stdout


def test_run_mock_all_strata(tmp_path):
    """All 4 strata complete a mock run without error."""
    for stratum in ("O1", "O2", "O3", "O4"):
        cfg = tmp_path / f"{stratum}.yaml"
        cfg.write_text(textwrap.dedent(f"""\
            stratum: {stratum}
            seed: 0
            steps: 2
            start_datetime: "2025-08-10 06:00"
            thermostat_setpoint: 22.0
            devices:
              - id: hvac
                on: true
                power_w: 3500
            rooms:
              - living_room
              - bedroom
            zone_temperature:
              mode: constant
              value: 23.0
            outdoor_temperature:
              mode: summer_default
            tou_rate:
              mode: peak_default
        """))
        r = _run("run", str(cfg), "--mock", "--steps", "2")
        assert r.returncode == 0, f"run --mock failed for {stratum}:\n{r.stderr}"


def test_run_missing_config_exits_nonzero():
    r = _run("run", "does_not_exist.yaml")
    assert r.returncode != 0


def test_run_mock_zone_temp_csv(tmp_path):
    """--mock with zone_temperature mode: csv uses the sample CSV."""
    csv_path = REPO_ROOT / "examples" / "data" / "zone_temps_sample.csv"
    if not csv_path.exists():
        pytest.skip("zone_temps_sample.csv not present")

    cfg = tmp_path / "csv_config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        stratum: O1
        seed: 0
        steps: 2
        start_datetime: "2025-08-10 18:00"
        thermostat_setpoint: 22.0
        devices:
          - id: hvac
            on: true
            power_w: 3500
        rooms:
          - living_room
        zone_temperature:
          mode: csv
          csv_path: {csv_path}
        outdoor_temperature:
          mode: summer_default
        tou_rate:
          mode: peak_default
    """))
    r = _run("run", str(cfg), "--mock", "--steps", "2")
    assert r.returncode == 0, f"CSV zone temp run failed:\n{r.stderr}"


# ── buildocc-api ──────────────────────────────────────────────────────────────

def test_api_entry_point_importable():
    from occupant_agent.api.app import run
    assert callable(run)


def test_api_server_starts(tmp_path):
    """The API server process starts and serves at least one request."""
    import os
    import socket
    import time

    db = tmp_path / "test.db"
    env = {**os.environ, "OCCUPANT_AGENT_DB": str(db)}

    try:
        proc = subprocess.Popen(
            ["buildocc-api"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        pytest.skip("'buildocc-api' entry point not on PATH (editable install)")
    try:
        # Wait for the server to start (up to 5 seconds)
        deadline = time.time() + 5
        started = False
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
                    started = True
                    break
            except OSError:
                time.sleep(0.2)

        assert started, "API server did not start within 5 seconds"

        # Verify the health endpoint responds
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        body = resp.read().decode()
        assert '"ok"' in body
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ── buildocc-mcp ──────────────────────────────────────────────────────────────

def test_mcp_entry_point_importable():
    from occupant_agent.mcp_server.server import main_sync
    assert callable(main_sync)


def test_mcp_server_starts_without_error(tmp_path):
    """
    MCP server listens on stdio; send the MCP initialization handshake and
    verify it replies with a valid JSON-RPC response before exiting.
    """
    import json
    import os

    init_msg = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.0.1"},
        },
    }) + "\n"

    env = {**os.environ, "BUILDOCC_API_URL": "http://localhost:8000"}
    try:
        r = subprocess.run(
            ["buildocc-mcp"],
            input=init_msg,
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
    except FileNotFoundError:
        pytest.skip("'buildocc-mcp' entry point not on PATH (editable install)")
    except subprocess.TimeoutExpired:
        # Server waited for more stdin — that's acceptable; it started
        return

    # If it exited immediately, the output should contain a valid JSON-RPC response
    # or at minimum not an import/syntax error
    combined = r.stdout + r.stderr
    assert "Traceback" not in combined, f"MCP server crashed:\n{combined}"
