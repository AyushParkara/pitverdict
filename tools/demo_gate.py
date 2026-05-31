from __future__ import annotations

"""One-command demo gate.

Runs a headless MDCE decision analysis (CLI) and validates outputs.

This is meant to be the last-mile "does the demo still work" command.
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str]) -> None:
    import subprocess

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MDCE demo gate (CLI run + validators).")
    p.add_argument("--root", default=None, help="Project root override.")
    p.add_argument(
        "--preset",
        default="custom",
        help="Scenario preset to run (e.g. custom, high_uncertainty_stack).",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Where to write CLI artifacts.",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Optional processed CSV. If omitted, uses default loader behavior.",
    )
    p.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="Set MDCE_ALLOW_NONCOMMERCIAL_DATA=1 for this gate run.",
    )
    p.add_argument(
        "--with-robustness",
        action="store_true",
        help="Also run the deterministic fuzz harness (no crashes / no invalid output).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # `--root` controls where the CLI resolves data/outputs; it is not where the code lives.
    # Always execute the repo's scripts from PROJECT_ROOT so callers can point `--root`
    # at a temp folder, Colab/Drive checkout, etc.
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    env = dict(os.environ)
    if args.allow_noncommercial:
        env["MDCE_ALLOW_NONCOMMERCIAL_DATA"] = "1"
    else:
        env.pop("MDCE_ALLOW_NONCOMMERCIAL_DATA", None)

    run_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_mdce_decision.py"),
        "--root",
        str(root),
        "--no-granite",
    ]
    run_cmd += ["--output-dir", str(out_dir)]
    if args.preset:
        run_cmd += ["--preset", str(args.preset)]
    if args.input:
        # `--input` should resolve relative to `--root` for predictable demo/test runs.
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        run_cmd += ["--input", str(input_path)]

    # Run the analysis.
    import subprocess

    proc = subprocess.run(run_cmd, text=True, capture_output=True, env=env)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # When multiple runs write into the same outputs folder, "latest" selection can
    # race. Prefer validating the exact JSON artifact the CLI just reported.
    json_from_run: str | None = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("JSON:"):
            json_from_run = line.split(":", 1)[1].strip()
            if json_from_run:
                break

    # Validate latest artifacts.
    validate_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "validate_mdce_outputs.py"),
        "--outputs-dir",
        str(out_dir),
    ]
    if json_from_run:
        validate_cmd += ["--json", json_from_run]
    _run(validate_cmd)

    # Optional: if an input CSV is provided, run cheap dataset hygiene validation.
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "validate_mdce_dataset.py"),
                "--input",
                str(input_path),
            ]
        )

    # Optional: robustness gate (deterministic fuzz harness). Non-zero exit on hard failures.
    if args.with_robustness:
        _run([sys.executable, str(PROJECT_ROOT / "tools" / "mdce_fuzz.py")])

    # Print the latest JSON path for convenience.
    latest: Path | None = None
    if json_from_run:
        p = Path(json_from_run)
        if p.exists():
            latest = p
    if latest is None:
        json_files = sorted(out_dir.glob("mdce_decision_run_*.json"))
        if json_files:
            latest = json_files[-1]
    if latest is not None:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        print("Latest JSON:", latest)
        print("Run ID:", payload.get("run_id"))


if __name__ == "__main__":
    main()
