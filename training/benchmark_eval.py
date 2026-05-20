"""
Offline benchmark evaluation: demand forecasting + pricing policies in simulator.

Produces tabular results consumed by scripts/run_benchmark.py for CSV/Markdown export.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from benchmarks.standards import (
    DEMAND_R2_PASS,
    PRICE_MONOTONICITY_MIN_FRACTION,
    PRICING_EPISODES_DEFAULT,
    SOURCES_BY_ID,
)
from config.constants import (
    DEMAND_TARGET_COL,
    EPISODE_LENGTH,
    N_PRICE_ACTIONS,
    PRICE_LEVELS,
    PROCESSED_SALES_PATH,
    PRODUCT_CATEGORIES,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
)
from environment.pricing_env import PricingEnv
from training.demand_features import (
    ENCODER_PATH,
    MODEL_PATH,
    DemandFeatureEncoder,
    DemandPredictorWrapper,
    load_processed_sales,
    time_based_split,
)
from training.demand_metrics import compute_demand_metrics
from training.ppo_paths import list_available_ppo_models, ppo_model_path, resolve_ppo_path_for_category
from training.train_ppo import load_demand_predictor, run_episode


# ---------------------------------------------------------------------------
# Demand baselines (Hyndman FPP3-style)
# ---------------------------------------------------------------------------


def _build_history_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train = train_df.copy()
    test = test_df.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    return pd.concat([train, test], ignore_index=True).sort_values(
        ["category", "date"]
    ).reset_index(drop=True)


def predict_naive_global_mean(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    mu = float(train_df[DEMAND_TARGET_COL].mean())
    return np.full(len(test_df), mu)


def predict_naive_category_mean(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    means = train_df.groupby("category")[DEMAND_TARGET_COL].mean()
    global_mu = float(train_df[DEMAND_TARGET_COL].mean())
    return test_df["category"].map(lambda c: means.get(c, global_mu)).astype(float).values


def predict_naive_lag1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Lag-1 per category (last observed demand); Hyndman naive forecast."""
    hist = _build_history_frame(train_df, test_df)
    preds = []
    for cat in hist["category"].unique():
        sub = hist[hist["category"] == cat].copy()
        sub["pred"] = sub[DEMAND_TARGET_COL].shift(1)
        cat_mean = float(train_df.loc[train_df["category"] == cat, DEMAND_TARGET_COL].mean())
        sub["pred"] = sub["pred"].fillna(cat_mean)
        preds.append(sub[sub["_split"] == "test"][["date", "category", "pred"]])
    out = pd.concat(preds, ignore_index=True)
    merged = test_df.merge(out, on=["date", "category"], how="left")
    return merged["pred"].fillna(train_df[DEMAND_TARGET_COL].mean()).values


def predict_seasonal_naive_dow(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Seasonal naive: same day-of-week per category mean from train."""
    stats = train_df.groupby(["category", "day_of_week"])[DEMAND_TARGET_COL].mean()
    cat_means = train_df.groupby("category")[DEMAND_TARGET_COL].mean()
    global_mu = float(train_df[DEMAND_TARGET_COL].mean())

    def lookup(row):
        key = (row["category"], row["day_of_week"])
        if key in stats.index:
            return float(stats.loc[key])
        if row["category"] in cat_means.index:
            return float(cat_means.loc[row["category"]])
        return global_mu

    return test_df.apply(lookup, axis=1).values


def predict_ridge_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoder: DemandFeatureEncoder,
) -> np.ndarray:
    """Ridge regression on same features as LightGBM (linear benchmark)."""
    enc = DemandFeatureEncoder()
    enc.fit(train_df)
    X_train = enc.transform(train_df)
    y_train = train_df[DEMAND_TARGET_COL].values
    X_test = enc.transform(test_df)
    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return np.clip(model.predict(X_test), 0, None)


def predict_lightgbm_saved(
    test_df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> Optional[np.ndarray]:
    if not model_path.exists() or not encoder_path.exists():
        return None
    model = joblib.load(model_path)
    encoder = DemandFeatureEncoder.load(encoder_path)
    X_test = encoder.transform(test_df)
    return np.clip(model.predict(X_test), 0, None)


def evaluate_demand_benchmarks(
    data_path: Path = PROCESSED_SALES_PATH,
    test_size: float = TRAIN_TEST_SPLIT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Returns (summary_df, by_category_df, meta_dict).
    """
    df = load_processed_sales(data_path)
    train_df, test_df = time_based_split(df, test_size=test_size)
    y_test = test_df[DEMAND_TARGET_COL].values

    models: List[Tuple[str, str, Callable[[], np.ndarray]]] = [
        ("naive_global_mean", "hyndman2021", lambda: predict_naive_global_mean(train_df, test_df)),
        ("naive_category_mean", "hyndman2021", lambda: predict_naive_category_mean(train_df, test_df)),
        ("naive_lag1", "hyndman2021", lambda: predict_naive_lag1(train_df, test_df)),
        ("seasonal_naive_dow", "hyndman2021", lambda: predict_seasonal_naive_dow(train_df, test_df)),
    ]

    encoder_for_ridge = DemandFeatureEncoder()
    encoder_for_ridge.fit(train_df)
    models.append(
        (
            "ridge_same_features",
            "hyndman2021",
            lambda: predict_ridge_baseline(train_df, test_df, encoder_for_ridge),
        )
    )

    lgb_pred = predict_lightgbm_saved(test_df)
    if lgb_pred is not None:
        models.append(("lightgbm_v2", "project_mvp_demand", lambda: lgb_pred))

    summary_rows = []
    by_cat_rows = []
    naive_wmapes = []

    for model_name, source_id, pred_fn in models:
        y_pred = np.clip(pred_fn(), 0, None)
        metrics = compute_demand_metrics(y_test, y_pred)
        pass_r2 = metrics["r2"] >= DEMAND_R2_PASS if model_name == "lightgbm_v2" else None
        beat_naive = None

        if model_name != "lightgbm_v2" and not np.isnan(metrics.get("wmape", float("nan"))):
            naive_wmapes.append((model_name, metrics["wmape"]))
        elif model_name == "lightgbm_v2" and naive_wmapes:
            best_naive_wmape = min(w for _, w in naive_wmapes)
            beat_naive = metrics["wmape"] < best_naive_wmape

        summary_rows.append({
            "section": "demand_forecast",
            "model": model_name,
            "category": "all",
            "metric": "r2",
            "value": metrics["r2"],
            "unit": "ratio",
            "benchmark_source_id": source_id,
            "pass": pass_r2 if pass_r2 is not None else "",
            "notes": "MVP gate R²≥0.30" if model_name == "lightgbm_v2" else "",
        })
        for mk, mv in [
            ("wmape", metrics["wmape"]),
            ("mape", metrics["mape"]),
            ("rmse", metrics["rmse"]),
            ("mae", metrics["mae"]),
            ("r2_demand_positive", metrics["r2_demand_positive"]),
        ]:
            summary_rows.append({
                "section": "demand_forecast",
                "model": model_name,
                "category": "all",
                "metric": mk,
                "value": mv,
                "unit": "ratio" if mk.startswith("r2") or mk.endswith("ape") else "units",
                "benchmark_source_id": "makridakis2022" if "wmape" in mk or "mape" in mk else source_id,
                "pass": beat_naive if model_name == "lightgbm_v2" and mk == "wmape" else "",
                "notes": "Beat best naive WMAPE" if model_name == "lightgbm_v2" and mk == "wmape" else "",
            })

        for cat in test_df["category"].unique():
            mask = test_df["category"] == cat
            if mask.sum() < 2:
                continue
            m_cat = compute_demand_metrics(y_test[mask], y_pred[mask])
            by_cat_rows.append({
                "section": "demand_forecast",
                "model": model_name,
                "category": cat,
                "metric": "r2",
                "value": m_cat["r2"],
                "unit": "ratio",
                "benchmark_source_id": source_id,
                "pass": "",
                "notes": f"n_test={int(mask.sum())}",
            })
            by_cat_rows.append({
                "section": "demand_forecast",
                "model": model_name,
                "category": cat,
                "metric": "wmape",
                "value": m_cat["wmape"],
                "unit": "ratio",
                "benchmark_source_id": "makridakis2022",
                "pass": "",
                "notes": "",
            })

    meta = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_size": test_size,
        "date_test_start": str(test_df["date"].min().date()),
        "date_test_end": str(test_df["date"].max().date()),
        "pct_zero_demand_test": float((test_df[DEMAND_TARGET_COL] == 0).mean()),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(by_cat_rows), meta


# ---------------------------------------------------------------------------
# Simulator sanity (price → demand)
# ---------------------------------------------------------------------------


def evaluate_simulator_monotonicity(
    category: str = "Books",
    n_prices: int = 15,
) -> pd.DataFrame:
    """Check demand non-increasing when price increases (LightGBM + price context)."""
    if not MODEL_PATH.exists():
        return pd.DataFrame([{
            "section": "simulator",
            "model": "lightgbm_v2",
            "category": category,
            "metric": "monotonic_violation_rate",
            "value": float("nan"),
            "unit": "ratio",
            "benchmark_source_id": "project_mvp_pricing",
            "pass": "skip",
            "notes": "Demand model not found",
        }])

    predictor = DemandPredictorWrapper.load()
    prices = np.linspace(PRICE_LEVELS[0], PRICE_LEVELS[-1], n_prices)
    demands = [
        predictor.predict_demand(float(p), day_of_week=2, month=6, category=category)
        for p in prices
    ]
    violations = sum(
        1 for i in range(len(demands) - 1) if demands[i + 1] > demands[i] + 1e-6
    )
    violation_rate = violations / max(1, len(demands) - 1)
    ok = violation_rate <= (1 - PRICE_MONOTONICITY_MIN_FRACTION)

    rows = [{
        "section": "simulator",
        "model": "lightgbm_v2",
        "category": category,
        "metric": "monotonic_violation_rate",
        "value": violation_rate,
        "unit": "ratio",
        "benchmark_source_id": "ferreira2016",
        "pass": "pass" if ok else "fail",
        "notes": f"Demand should fall with price; {violations}/{len(demands)-1} upward steps",
    }]
    for p, d in zip(prices, demands):
        rows.append({
            "section": "simulator_curve",
            "model": "lightgbm_v2",
            "category": category,
            "metric": "predicted_demand",
            "value": d,
            "unit": "units",
            "benchmark_source_id": "ferreira2016",
            "pass": "",
            "notes": f"price={p:.0f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pricing policies (simulator)
# ---------------------------------------------------------------------------


def _median_price_index(category: str) -> int:
    if not PROCESSED_SALES_PATH.exists():
        return N_PRICE_ACTIONS // 2
    df = load_processed_sales(PROCESSED_SALES_PATH)
    sub = df[df["category"] == category]
    if sub.empty:
        return N_PRICE_ACTIONS // 2
    med = float(sub["price"].median())
    return int(np.argmin(np.abs(np.array(PRICE_LEVELS) - med)))


def _pricing_strategies() -> Dict[str, Callable]:
    return {
        "random": lambda obs, env: env.action_space.sample(),
        "low_price": lambda obs, env: 0,
        "mid_price": lambda obs, env: N_PRICE_ACTIONS // 2,
        "high_price": lambda obs, env: N_PRICE_ACTIONS - 1,
    }


def _run_pricing_episodes(
    policy: Callable,
    category: str,
    demand_predictor,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    profits, revenues, units, prices = [], [], [], []
    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = PricingEnv(
            demand_predictor=demand_predictor,
            product_category=category,
            seed=ep_seed,
        )
        outcome = run_episode(env, policy, ep_seed)
        hist = env.get_history_df()
        profits.append(outcome["profit"])
        revenues.append(float((hist["prices"] * hist["sales"]).sum()) if len(hist) else 0.0)
        units.append(float(hist["sales"].sum()) if len(hist) else 0.0)
        prices.append(float(hist["prices"].mean()) if len(hist) else 0.0)
    return {
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_units_sold": float(np.mean(units)),
        "mean_price": float(np.mean(prices)),
    }


def evaluate_pricing_benchmarks(
    n_episodes: int = PRICING_EPISODES_DEFAULT,
    seed: int = RANDOM_SEED,
    use_demand_model: bool = True,
) -> pd.DataFrame:
    demand_predictor = load_demand_predictor() if use_demand_model else None
    rows = []
    strategies = _pricing_strategies()

    for category in PRODUCT_CATEGORIES:
        median_idx = _median_price_index(category)

        def make_median_policy(idx):
            return lambda obs, env, i=idx: i

        cat_strategies = {**strategies, "median_hist_price": make_median_policy(median_idx)}

        # PPO if checkpoint exists
        ppo_path = resolve_ppo_path_for_category(category)
        if ppo_path.exists():
            from stable_baselines3 import PPO

            ppo = PPO.load(str(ppo_path))

            def ppo_policy(obs, env, model=ppo):
                action, _ = model.predict(obs, deterministic=True)
                return int(action)

            cat_strategies["ppo"] = ppo_policy

        results: Dict[str, Dict[str, float]] = {}
        for name, policy in cat_strategies.items():
            results[name] = _run_pricing_episodes(
                policy, category, demand_predictor, n_episodes, seed
            )

        mid_profit = results.get("mid_price", {}).get("mean_profit", 0.0)
        rand_profit = results.get("random", {}).get("mean_profit", 0.0)

        for name, stats in results.items():
            source = "stable_baselines3" if name == "ppo" else "ferreira2016"
            if name == "ppo":
                source = "project_mvp_pricing"
            pass_flag = ""
            if name == "ppo":
                pass_flag = (
                    "pass"
                    if stats["mean_profit"] > mid_profit and stats["mean_profit"] > rand_profit
                    else "fail"
                )
            for metric, val in stats.items():
                rows.append({
                    "section": "pricing_simulator",
                    "model": name,
                    "category": category,
                    "metric": metric,
                    "value": val,
                    "unit": "usd" if "profit" in metric or "revenue" in metric else (
                        "units" if "units" in metric else "usd"
                    ),
                    "benchmark_source_id": source,
                    "pass": pass_flag if name == "ppo" and metric == "mean_profit" else "",
                    "notes": (
                        f"{n_episodes} episodes x {EPISODE_LENGTH} days; "
                        f"seed={seed}; demand={'LightGBM' if use_demand_model else 'analytical'}"
                    ),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _df_to_markdown_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Simple markdown table without tabulate dependency."""
    if df.empty:
        return "_No data_"
    idx_name = df.index.name or "model"
    headers = [idx_name] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            cells.append(f"{v:{float_fmt}}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_benchmark_reports(
    output_dir: Path,
    demand_summary: pd.DataFrame,
    demand_by_cat: pd.DataFrame,
    pricing_df: pd.DataFrame,
    simulator_df: pd.DataFrame,
    demand_meta: Dict[str, Any],
    run_notes: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    demand_all = pd.concat([demand_summary, demand_by_cat], ignore_index=True)
    combined = pd.concat(
        [demand_all, simulator_df, pricing_df],
        ignore_index=True,
    )

    combined.to_csv(output_dir / "benchmark_results.csv", index=False)
    demand_all.to_csv(output_dir / "demand_benchmark.csv", index=False)
    pricing_df.to_csv(output_dir / "pricing_benchmark.csv", index=False)
    simulator_df.to_csv(output_dir / "simulator_benchmark.csv", index=False)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": ts, "demand_meta": demand_meta, "notes": run_notes}, f, indent=2)

    md = _build_markdown_report(
        ts, demand_summary, demand_by_cat, pricing_df, simulator_df, demand_meta, run_notes
    )
    (output_dir / "BENCHMARK_REPORT.md").write_text(md, encoding="utf-8")

    from benchmarks.standards import format_references_markdown

    (output_dir / "REFERENCES.md").write_text(format_references_markdown(), encoding="utf-8")

    return output_dir


def _build_markdown_report(
    ts: str,
    demand_summary: pd.DataFrame,
    demand_by_cat: pd.DataFrame,
    pricing_df: pd.DataFrame,
    simulator_df: pd.DataFrame,
    demand_meta: Dict[str, Any],
    run_notes: str,
) -> str:
    from benchmarks.standards import (
        DEMAND_R2_PASS,
        format_references_markdown,
        source_citation,
    )

    lines = [
        "# Báo cáo benchmark — RL Dynamic Pricing",
        "",
        f"**Thời gian chạy:** {ts}",
        "",
        "Báo cáo này so sánh mô hình dự báo demand và chính sách định giá với **baseline chuẩn** ",
        "được trích dẫn trong `REFERENCES.md` (cột `benchmark_source_id` trong CSV).",
        "",
        "---",
        "",
        "## 1. Dữ liệu & phương pháp",
        "",
        f"- Nguồn: `{PROCESSED_SALES_PATH.name}` (category × ngày)",
        f"- Train/test: time-based hold-out **{demand_meta.get('test_size', 0.2):.0%}** "
        f"(test từ {demand_meta.get('date_test_start')} đến {demand_meta.get('date_test_end')})",
        f"- Train rows: {demand_meta.get('n_train')}, test rows: {demand_meta.get('n_test')}",
        f"- % ngày demand=0 (test): {demand_meta.get('pct_zero_demand_test', 0):.1%}",
        "",
        "**Benchmark demand** (theo [hyndman2021]): naive mean, category mean, lag-1, seasonal naive (DOW), ",
        "Ridge cùng feature; **LightGBM v2** so với các baseline. ",
        f"Ngưỡng MVP nội bộ [project_mvp_demand]: **R² ≥ {DEMAND_R2_PASS}** trên test.",
        "",
        "**Benchmark pricing** (theo [ferreira2016], [sutton2018]): cùng `PricingEnv` + LightGBM simulator, ",
        f"{PRICING_EPISODES_DEFAULT} episode × {EPISODE_LENGTH} ngày; so Random / Low / Mid / High / ",
        "median giá lịch sử / PPO (nếu có checkpoint). ",
        "Ngưỡng MVP [project_mvp_pricing]: PPO mean profit > mid-price và > random.",
        "",
    ]
    if run_notes:
        lines.extend([f"*Ghi chú run:* {run_notes}", ""])

    lines.extend(["---", "", "## 2. Kết quả demand (toàn bộ test set)", ""])

    metrics_show = ["r2", "wmape", "mape", "rmse", "mae"]
    dsub = demand_summary[demand_summary["category"] == "all"]
    for metric in metrics_show:
        part = dsub[dsub["metric"] == metric].sort_values("model")
        if part.empty:
            continue
        lines.append(f"### {metric.upper()}")
        lines.append("")
        lines.append("| Model | Value | Source | Pass |")
        lines.append("|-------|------:|--------|------|")
        for _, r in part.iterrows():
            val = r["value"]
            disp = f"{val:.4f}" if isinstance(val, (int, float)) and metric.startswith("r2") else (
                f"{val:.2%}" if metric.endswith("ape") and isinstance(val, float) and not np.isnan(val)
                else f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            )
            src = r["benchmark_source_id"]
            lines.append(f"| {r['model']} | {disp} | [{src}] | {r.get('pass', '')} |")
        lines.append("")

    lines.extend(["---", "", "## 3. Demand theo category (R²)", ""])
    r2_cat = demand_by_cat[demand_by_cat["metric"] == "r2"]
    if not r2_cat.empty:
        pivot = r2_cat.pivot(index="model", columns="category", values="value")
        pivot.index.name = "model"
        lines.append(_df_to_markdown_table(pivot.round(4)))
        lines.append("")

    lines.extend(["---", "", "## 4. Simulator — độ nhạy giá", ""])
    mono = simulator_df[simulator_df["metric"] == "monotonic_violation_rate"]
    for _, r in mono.iterrows():
        lines.append(
            f"- **{r['category']}**: violation rate = {r['value']:.2%} "
            f"({r['pass']}) — [{r['benchmark_source_id']}] {r['notes']}"
        )
    lines.append("")

    lines.extend(["---", "", "## 5. Pricing trong simulator (mean profit USD)", ""])
    if not pricing_df.empty:
        prof = pricing_df[pricing_df["metric"] == "mean_profit"]
        pivot_p = prof.pivot(index="model", columns="category", values="value")
        pivot_p.index.name = "model"
        lines.append(_df_to_markdown_table(pivot_p.round(2), float_fmt=".2f"))
        lines.append("")
        ppo_pass = prof[prof["model"] == "ppo"][["category", "value", "pass"]]
        if not ppo_pass.empty:
            lines.append("**PPO vs MVP gate [project_mvp_pricing]:**")
            for _, r in ppo_pass.iterrows():
                lines.append(f"- {r['category']}: ${r['value']:.2f} → **{r['pass']}**")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 6. File đầu ra",
        "",
        "| File | Nội dung |",
        "|------|----------|",
        "| `benchmark_results.csv` | Bảng dài (mọi section) |",
        "| `demand_benchmark.csv` | Demand only |",
        "| `pricing_benchmark.csv` | Pricing only |",
        "| `simulator_benchmark.csv` | Price–demand curve & monotonicity |",
        "| `REFERENCES.md` | Trích dẫn benchmark |",
        "",
        "---",
        "",
        format_references_markdown(),
    ])
    return "\n".join(lines)
