"""Command-Line Telemetry Entrypoint (`adpulse/cli.py`).

Usage:
    python3 -m adpulse.cli --mode demo-all
    python3 -m adpulse.cli --mode ranker
    python3 -m adpulse.cli --mode pacing
    python3 -m adpulse.cli --mode ope
    python3 -m adpulse.cli --mode diagnostics
    python3 -m adpulse.cli --mode cax
    python3 -m adpulse.cli --mode offsearch
    python3 -m adpulse.cli --mode pal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Sequence, Tuple

from . import __version__
from .auction_pacer import run_pacing_benchmark
from .campaign_diagnostics import run_diagnostics_benchmark
from .catalog_ranker import generate_synthetic_catalog_and_queries, run_ranker_benchmark
from .cax_offsearch_pal import (
    run_cax_benchmark,
    run_offsearch_benchmark,
    run_pal_and_ha_benchmark,
)
from .off_policy_eval import run_ope_benchmark


SUPPORTED_MODES: Tuple[str, ...] = (
    "demo-all",
    "ranker",
    "pacing",
    "ope",
    "diagnostics",
    "cax",
    "offsearch",
    "pal",
)


def execute_cli_mode(mode: str = "demo-all", seed: int = 42) -> Dict[str, Any]:
    """Execute an AdPulse subsystem mode and return structured JSON-ready telemetry."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from: {', '.join(SUPPORTED_MODES)}")

    t0 = time.perf_counter()
    data: Dict[str, Any] = {}

    if mode in ("demo-all", "ranker"):
        data["ranker"] = run_ranker_benchmark(seed=seed).to_dict()

    if mode in ("demo-all", "pacing"):
        data["pacing"] = run_pacing_benchmark(seed=seed).to_dict()

    if mode in ("demo-all", "ope"):
        data["ope"] = run_ope_benchmark(seed=seed).to_dict()

    if mode in ("demo-all", "diagnostics"):
        data["diagnostics"] = run_diagnostics_benchmark(seed=seed).to_dict()

    if mode in ("demo-all", "cax"):
        base_cat, _ = generate_synthetic_catalog_and_queries(num_skus=500, num_queries=120, seed=seed)
        data["cax"] = run_cax_benchmark(base_cat, seed=seed).to_dict()

    if mode in ("demo-all", "offsearch"):
        data["offsearch"] = run_offsearch_benchmark(seed=seed).to_dict()

    if mode in ("demo-all", "pal"):
        data["pal"] = run_pal_and_ha_benchmark(seed=seed).to_dict()

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "engine": "AdPulse",
        "version": __version__,
        "mode": mode,
        "seed": seed,
        "wall_time_ms": wall_ms,
        "telemetry": data if mode == "demo-all" else data[mode],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adpulse",
        description="AdPulse v2 - Sponsored Catalog Ranker, CAX, Off-Search PDP, PAL Calibration, GSP Pacer & OPE",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="demo-all",
        help="Subsystem telemetry mode to execute (default: demo-all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line compact JSON instead of pretty-printed JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = execute_cli_mode(mode=args.mode, seed=args.seed)
    if args.compact:
        sys.stdout.write(json.dumps(payload) + "\n")
    else:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
