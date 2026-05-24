"""
Benchmark thresholds and bibliographic references for evaluation reports.

Citations are embedded in generated Markdown/CSV (column benchmark_source_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BenchmarkSource:
    id: str
    citation: str
    url: str
    used_for: str


BENCHMARK_SOURCES: List[BenchmarkSource] = [
    BenchmarkSource(
        id="hyndman2021",
        citation=(
            "Hyndman, R.J., & Athanasopoulos, G. (2021). "
            "Forecasting: Principles and Practice (3rd ed). OTexts."
        ),
        url="https://otexts.com/fpp3/",
        used_for=(
            "Naive forecast (lag-1) and seasonal naive (same day-of-week) baselines; "
            "forecast accuracy on hold-out samples (Ch. 5–6)."
        ),
    ),
    BenchmarkSource(
        id="makridakis2022",
        citation=(
            "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). "
            "M5 accuracy competition: Results, findings and conclusions. "
            "International Journal of Forecasting, 38(4), 1346–1364."
        ),
        url="https://doi.org/10.1016/j.ijforecast.2022.08.004",
        used_for=(
            "Retail-style demand forecasting benchmarks; weighted errors (WMAPE family) "
            "common in hierarchical sales competitions."
        ),
    ),
    BenchmarkSource(
        id="project_mvp_demand",
        citation="rl-dynamic-pricing project (internal MVP gate, train_demand_predictor.py).",
        url="",
        used_for="Pass criterion: test R² ≥ 0.30 on time-based hold-out (processed_sales.csv).",
    ),
    BenchmarkSource(
        id="ferreira2016",
        citation=(
            "Ferreira, K.J., Lee, B.H.A., & Simchi-Levi, D. (2016). "
            "Analytics for an Online Retailer: Large-Scale Dynamic Pricing. "
            "Management Science, 62(6), 1705–1719."
        ),
        url="https://doi.org/10.1287/mnsc.2015.2213",
        used_for=(
            "Dynamic pricing policy evaluation against fixed-price and simple "
            "rule-based baselines in a simulated market."
        ),
    ),
    BenchmarkSource(
        id="sutton2018",
        citation="Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed). MIT Press.",
        url="http://incompleteideas.net/book/the-book-2nd.html",
        used_for=(
            "Episodic return (cumulative reward/profit) over a fixed horizon; "
            "compare learned policy to hand-crafted policies in the same environment."
        ),
    ),
    BenchmarkSource(
        id="stable_baselines3",
        citation="Raffin, A., et al. (2021). Stable-Baselines3: Reliable RL Implementations. JMLR 22(268), 1–8.",
        url="https://www.jmlr.org/papers/v22/20-1444.html",
        used_for="PPO agent implementation and deterministic evaluation rollouts.",
    ),
    BenchmarkSource(
        id="project_mvp_pricing",
        citation="rl-dynamic-pricing project (internal simulator benchmark).",
        url="",
        used_for=(
            "Pass criterion: mean 30-day episode profit > fixed mid-price policy "
            "and > random pricing, same PricingEnv + LightGBM demand simulator. "
            "Myopic oracle (greedy 1-step profit) as strong non-RL baseline."
        ),
    ),
]

SOURCES_BY_ID: Dict[str, BenchmarkSource] = {s.id: s for s in BENCHMARK_SOURCES}


# --- Demand forecasting gates (test hold-out, time split) ---
DEMAND_R2_PASS = 0.30
DEMAND_WMAPE_IMPROVE_VS_BEST_NAIVE = True  # LightGBM WMAPE < best naive WMAPE

# --- Pricing gates (simulator, per category) ---
PRICING_EPISODES_DEFAULT = 20
PRICING_PROFIT_BEAT_MID = True
PRICING_PROFIT_BEAT_RANDOM = True

# --- Simulator qualitative check ---
PRICE_MONOTONICITY_MIN_FRACTION = 0.85  # share of price steps with non-increasing demand


def format_references_markdown() -> str:
    lines = ["## Tài liệu tham chiếu (benchmark sources)\n"]
    for src in BENCHMARK_SOURCES:
        url_part = f" — {src.url}" if src.url else ""
        lines.append(f"- **[{src.id}]** {src.citation}{url_part}")
        lines.append(f"  - *Dùng cho:* {src.used_for}\n")
    return "\n".join(lines)


def source_citation(source_id: str) -> str:
    src = SOURCES_BY_ID.get(source_id)
    return src.citation if src else source_id
