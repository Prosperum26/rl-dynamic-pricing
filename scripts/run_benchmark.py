#!/usr/bin/env python
"""
Run standardized offline benchmarks and write CSV + Markdown reports.

Usage (from project root):
    python -m scripts.run_benchmark
    python -m scripts.run_benchmark --output reports/benchmark
    python -m scripts.run_benchmark --pricing-episodes 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import PROCESSED_SALES_PATH, PROJECT_ROOT as ROOT
from training.benchmark_eval import (
    evaluate_demand_benchmarks,
    evaluate_pricing_benchmarks,
    evaluate_simulator_monotonicity,
    write_benchmark_reports,
)

DEFAULT_OUTPUT = ROOT / "reports" / "benchmark"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate demand & pricing benchmarks; export CSV and Markdown."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory for reports (default: reports/benchmark)",
    )
    parser.add_argument(
        "--pricing-episodes",
        type=int,
        default=20,
        help="Episodes per pricing policy per category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pricing rollouts",
    )
    parser.add_argument(
        "--skip-pricing",
        action="store_true",
        help="Skip pricing simulator benchmarks (faster)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional subfolder tag (e.g. run1). Default: timestamp subfolder",
    )
    args = parser.parse_args()

    if not PROCESSED_SALES_PATH.exists():
        print(f"Missing processed data: {PROCESSED_SALES_PATH}")
        print("Run: python -m scripts.preprocess_ecommerce")
        return 1

    base_out = Path(args.output)
    sub = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_out / sub

    print("=" * 60)
    print("RL Dynamic Pricing — Benchmark run")
    print("=" * 60)
    print(f"Output: {output_dir}")

    print("\n[1/3] Demand forecasting benchmarks...")
    demand_summary, demand_by_cat, demand_meta = evaluate_demand_benchmarks()

    print("\n[2/3] Simulator price sensitivity...")
    sim_frames = []
    from config.constants import PRODUCT_CATEGORIES

    for cat in PRODUCT_CATEGORIES:
        sim_frames.append(evaluate_simulator_monotonicity(category=cat))
    import pandas as pd

    simulator_df = pd.concat(sim_frames, ignore_index=True)

    pricing_df = pd.DataFrame()
    if not args.skip_pricing:
        print(f"\n[3/3] Pricing benchmarks ({args.pricing_episodes} episodes/category)...")
        pricing_df = evaluate_pricing_benchmarks(
            n_episodes=args.pricing_episodes,
            seed=args.seed,
        )
    else:
        print("\n[3/3] Pricing benchmarks skipped.")

    out = write_benchmark_reports(
        output_dir,
        demand_summary,
        demand_by_cat,
        pricing_df,
        simulator_df,
        demand_meta,
        run_notes=f"pricing_episodes={args.pricing_episodes}, seed={args.seed}",
    )

    # Convenience: copy latest summary paths to base_out root
    import shutil

    base_out.mkdir(parents=True, exist_ok=True)
    for name in (
        "benchmark_results.csv",
        "BENCHMARK_REPORT.md",
        "REFERENCES.md",
    ):
        src = out / name
        if src.exists():
            shutil.copy2(src, base_out / name)

    (base_out / "LATEST_RUN.txt").write_text(str(out.resolve()), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print(f"  Report:  {out / 'BENCHMARK_REPORT.md'}")
    print(f"  CSV:     {out / 'benchmark_results.csv'}")
    print(f"  Latest:  {base_out / 'BENCHMARK_REPORT.md'} (copy)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
