"""
Streamlit Dashboard — RL Dynamic Pricing

Interactive web UI using the trained LightGBM demand model and PPO agent.

Run:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    DASHBOARD_DEFAULT_INITIAL_INVENTORY,
    DASHBOARD_TITLE,
    DEFAULT_PRODUCT_CATEGORY,
    EPISODE_LENGTH,
    LOGS_DIR,
    MAX_PRICE,
    MIN_PRICE,
    MODELS_DIR,
    N_PRICE_ACTIONS,
    PRICE_LEVELS,
    PRICE_STEP,
    PRODUCT_CATEGORIES,
    UNIT_COST,
)
from environment.pricing_env import PricingEnv
from training.demand_features import (
    METADATA_PATH,
    MODEL_PATH,
    DemandPredictorWrapper,
    load_metadata,
)
from dashboard.product_data import (
    load_product_daily,
    load_product_monthly,
    top_products_in_period,
)
from training.ppo_paths import list_available_ppo_models, resolve_ppo_path_for_category

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        color: #f0f4f8;
    }
    .status-ok { color: #3dd68c; font-weight: 600; }
    .status-miss { color: #f0ad4e; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

LEGACY_PPO_PATH = MODELS_DIR / "best_model" / "best_model.zip"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Model loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PPO agent...")
def load_ppo_model(path: str):
    from stable_baselines3 import PPO
    return PPO.load(path)


@st.cache_resource(show_spinner="Loading demand predictor...")
def load_demand_model():
    return DemandPredictorWrapper.load()


def price_to_index(price: float) -> int:
    return int(np.argmin(np.abs(np.array(PRICE_LEVELS) - price)))


def make_simulation_env(
    demand_predictor: Optional[DemandPredictorWrapper],
    category: str,
    seed: int,
) -> PricingEnv:
    return PricingEnv(
        demand_predictor=demand_predictor,
        product_category=category,
        seed=seed,
    )


def run_episode(
    env: PricingEnv,
    policy: Callable,
    n_days: int,
    reset_options: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    obs, info = env.reset(seed=seed, options=reset_options)
    for _ in range(n_days):
        action = policy(obs, env)
        obs, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    return env.get_history_df(), info


def build_demand_curve(
    predictor: DemandPredictorWrapper,
    category: str,
    day_of_week: int,
    month: int,
    prices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for p in prices:
        d = predictor.predict_demand(p, day_of_week, month, category)
        rows.append({"price": p, "predicted_demand": max(0.0, d)})
    return pd.DataFrame(rows)


def find_training_logs() -> list[Path]:
    if not LOGS_DIR.exists():
        return []
    return sorted(LOGS_DIR.glob("ppo_pricing_*/evaluations.npz"), reverse=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Control panel")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🎮 Live simulation",
        "🔮 What-if explorer",
        "📊 Strategy comparison",
        "📈 Training metrics",
        "🛍️ Product insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Market")

category = st.sidebar.selectbox(
    "Product category",
    PRODUCT_CATEGORIES,
    index=PRODUCT_CATEGORIES.index(DEFAULT_PRODUCT_CATEGORY),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Models")

auto_ppo_path = resolve_ppo_path_for_category(category)
ppo_auto_mode = st.sidebar.checkbox(
    "Auto PPO per category",
    value=True,
    help="Load models/best_model/best_model_<category>.zip when available.",
)

if ppo_auto_mode:
    resolved_ppo = auto_ppo_path
    if resolved_ppo.exists():
        st.sidebar.caption(f"PPO: `{resolved_ppo.name}`")
    else:
        st.sidebar.caption("PPO: no checkpoint for this category yet")
    ppo_path = str(resolved_ppo)
else:
    ppo_path = st.sidebar.text_input(
        "PPO model path",
        value=str(auto_ppo_path if auto_ppo_path.exists() else LEGACY_PPO_PATH),
    )

use_ppo = st.sidebar.checkbox("Use trained PPO", value=True)

available_ppo = list_available_ppo_models()
if available_ppo:
    with st.sidebar.expander("Available PPO checkpoints"):
        for label, path in sorted(available_ppo.items()):
            st.text(f"{label}: {path.name}")

use_ml_demand = st.sidebar.checkbox(
    "Use LightGBM demand model",
    value=True,
    help="When enabled, demand follows the trained predictor instead of the analytical formula.",
)

ppo_model = None
demand_predictor = None
demand_meta = {}

if use_ppo and Path(ppo_path).exists():
    try:
        ppo_model = load_ppo_model(ppo_path)
        st.sidebar.markdown('<p class="status-ok">✓ PPO loaded</p>', unsafe_allow_html=True)
    except Exception as exc:
        st.sidebar.markdown(f'<p class="status-miss">✗ PPO: {exc}</p>', unsafe_allow_html=True)
elif use_ppo:
    st.sidebar.markdown('<p class="status-miss">✗ PPO file not found</p>', unsafe_allow_html=True)
    if ppo_auto_mode:
        st.sidebar.caption(
            f"Train: `python -m training.train_ppo --category {category}` "
            f"or `--train-all-categories`"
        )

if use_ml_demand and MODEL_PATH.exists():
    try:
        demand_predictor = load_demand_model()
        if METADATA_PATH.exists():
            demand_meta = load_metadata()
        st.sidebar.markdown('<p class="status-ok">✓ Demand model loaded</p>', unsafe_allow_html=True)
    except Exception as exc:
        st.sidebar.markdown(f'<p class="status-miss">✗ Demand: {exc}</p>', unsafe_allow_html=True)
elif use_ml_demand:
    st.sidebar.markdown('<p class="status-miss">✗ Demand model not found</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Price grid: ${MIN_PRICE:.0f}–${MAX_PRICE:.0f} (step ${PRICE_STEP:.0f})")
st.sidebar.caption(f"Unit cost: ${UNIT_COST:.0f} | Episode: {EPISODE_LENGTH} days")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if page == "🏠 Overview":
    st.title(f"💰 {DASHBOARD_TITLE}")
    st.markdown(
        "Interactive pricing lab powered by **LightGBM demand forecasting** "
        "and a **PPO reinforcement-learning agent** trained on your e-commerce data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price actions", N_PRICE_ACTIONS)
    c2.metric("Categories", len(PRODUCT_CATEGORIES))
    c3.metric("Episode length", f"{EPISODE_LENGTH} d")
    if demand_meta:
        m = demand_meta.get("metrics", {})
        c4.metric("Demand R² / MAPE", f"{m.get('r2', 0):.2f} / {m.get('mape', 0):.0%}")
    else:
        c4.metric("Demand model", "—")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Pipeline")
        st.markdown(
            """
            1. **Preprocess** transactions → `data/processed/processed_sales.csv`  
            2. **Train demand** → LightGBM (global, per category)  
            3. **Train PPO** → pricing policy in simulated market  
            4. **Explore here** → simulate, what-if, compare strategies  
            5. **Product insights** → top SKU demand theo tháng/năm (từ raw)  
            """
        )

    with right:
        st.subheader("Model status")
        status_rows = [
            ("PPO agent", "Ready" if ppo_model else "Not loaded", Path(ppo_path).name),
            ("PPO path", "auto" if ppo_auto_mode else "manual", str(Path(ppo_path))),
            ("Demand predictor", "Ready" if demand_predictor else "Not loaded", str(MODEL_PATH)),
            ("Active category", category, ""),
        ]
        st.table(pd.DataFrame(status_rows, columns=["Component", "Status", "Path"]))

    st.info(
        "Start with **Live simulation** to watch the PPO agent price over a month, "
        "or **What-if explorer** to probe demand at different prices."
    )


# ---------------------------------------------------------------------------
# Live simulation
# ---------------------------------------------------------------------------

elif page == "🎮 Live simulation":
    st.title("🎮 Live pricing simulation")
    st.caption(
        "Episode với PPO + LightGBM. Obs gồm tháng (sin/cos) + category. "
        "Demand float; sales = ceil(demand). Giá có thể đổi theo mùa/context — "
        "policy mới (obs v2) cần train lại PPO."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_price = st.number_input(
            "Starting price ($)",
            min_value=float(MIN_PRICE),
            max_value=float(MAX_PRICE),
            value=400.0,
            step=float(PRICE_STEP),
        )
    with col2:
        initial_inventory = st.number_input(
            "Initial inventory",
            min_value=10,
            max_value=500,
            value=DASHBOARD_DEFAULT_INITIAL_INVENTORY,
            step=10,
        )
    with col3:
        sim_days = st.slider("Days", 7, 90, EPISODE_LENGTH)
    with col4:
        start_dow = st.selectbox("Start day", range(7), format_func=lambda i: DAY_NAMES[i])
        start_month = st.selectbox("Start month", range(1, 13), format_func=lambda m: MONTH_NAMES[m - 1])

    policy_label = "Random policy"
    if ppo_model is not None:
        policy_label = "PPO agent (deterministic)"

        def ppo_policy(obs, env):
            action, _ = ppo_model.predict(obs, deterministic=True)
            return int(action)
        policy = ppo_policy
    else:
        st.warning("PPO not loaded — using random actions.")

        def random_policy(obs, env):
            return env.action_space.sample()
        policy = random_policy

    predictor = demand_predictor if use_ml_demand else None
    if not predictor and use_ml_demand:
        st.warning("Demand model not loaded — using analytical demand.")

    if st.button("▶️ Run episode", type="primary", use_container_width=True):
        with st.spinner("Simulating..."):
            env = make_simulation_env(predictor, category, seed=42)
            history, info = run_episode(
                env,
                policy,
                sim_days,
                reset_options={
                    "initial_price_idx": price_to_index(initial_price),
                    "initial_inventory": initial_inventory,
                    "day_of_week": start_dow,
                    "month": start_month,
                    "product_category": category,
                },
                seed=42,
            )
            st.session_state["sim_history"] = history
            st.session_state["sim_info"] = info
            st.session_state["sim_policy"] = policy_label

    if "sim_history" in st.session_state:
        history = st.session_state["sim_history"]
        info = st.session_state["sim_info"]

        st.success(
            f"Done — **{st.session_state.get('sim_policy', 'Policy')}** | "
            f"Profit **${info['total_profit']:.2f}** | Revenue **${info['total_revenue']:.2f}**"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total profit", f"${info['total_profit']:.2f}")
        m2.metric("Total revenue", f"${info['total_revenue']:.2f}")
        m3.metric("Avg price", f"${history['prices'].mean():.2f}")
        m4.metric("Units sold", int(history["sales"].sum()))

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Price chosen by agent", "Demand vs sales", "Inventory"),
            vertical_spacing=0.08,
            row_heights=[0.35, 0.35, 0.3],
        )
        fig.add_trace(
            go.Scatter(
                x=history["day"],
                y=history["prices"],
                mode="lines+markers",
                name="Price",
                line=dict(color="#4f9cf9", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=history["day"],
                y=history["demands"],
                mode="lines+markers",
                name="Demand (float)",
                line=dict(color="#94a3b8", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=history["day"],
                y=history["sales"],
                mode="markers",
                name="Sales (ceil)",
                marker=dict(color="#3dd68c", size=9),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=history["day"],
                y=history["inventory"],
                mode="lines",
                name="Inventory",
                fill="tozeroy",
                line=dict(color="#f59e0b"),
            ),
            row=3,
            col=1,
        )
        fig.update_layout(height=720, template="plotly_dark", hovermode="x unified")
        fig.update_yaxes(title_text="USD", row=1, col=1)
        fig.update_yaxes(title_text="Units", row=2, col=1)
        fig.update_yaxes(title_text="Units", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Episode data"):
            st.dataframe(history, use_container_width=True)


# ---------------------------------------------------------------------------
# What-if explorer
# ---------------------------------------------------------------------------

elif page == "🔮 What-if explorer":
    st.title("🔮 Demand what-if explorer")
    st.caption("Probe the LightGBM demand model — change price and calendar context interactively.")

    if demand_predictor is None:
        st.error("Load the demand model first (`python -m training.train_demand_predictor`).")
        st.stop()

    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Scenario")
        whatif_price = st.slider(
            "Price ($)",
            float(MIN_PRICE),
            float(MAX_PRICE),
            400.0,
            float(PRICE_STEP),
        )
        whatif_dow = st.selectbox("Day of week", range(7), format_func=lambda i: DAY_NAMES[i], key="whatif_dow")
        whatif_month = st.selectbox(
            "Month", range(1, 13), format_func=lambda m: MONTH_NAMES[m - 1], key="whatif_month"
        )
        whatif_cat = st.selectbox("Category", PRODUCT_CATEGORIES, index=PRODUCT_CATEGORIES.index(category))

        pred = demand_predictor.predict_demand(
            whatif_price, whatif_dow, whatif_month, whatif_cat
        )
        est_profit = (whatif_price - UNIT_COST) * max(0.0, pred)

        st.markdown("---")
        st.metric("Predicted demand (units)", f"{max(0, pred):.2f}")
        st.metric("Est. gross margin / day", f"${est_profit:.2f}")
        st.caption(f"Margin ≈ (price − ${UNIT_COST:.0f}) × demand")

    with right:
        st.subheader("Demand vs price curve")
        n_points = st.slider("Curve resolution", 15, 50, 25)
        prices = np.linspace(MIN_PRICE, MAX_PRICE, n_points)
        curve_df = build_demand_curve(
            demand_predictor, whatif_cat, whatif_dow, whatif_month, prices
        )

        fig = px.line(
            curve_df,
            x="price",
            y="predicted_demand",
            markers=True,
            title=f"Demand curve — {whatif_cat}, {DAY_NAMES[whatif_dow]}, {MONTH_NAMES[whatif_month - 1]}",
            labels={"price": "Price ($)", "predicted_demand": "Predicted demand"},
        )
        fig.add_vline(
            x=whatif_price,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Your price ${whatif_price:.0f}",
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Compare categories at this price")
    compare_rows = []
    for cat in PRODUCT_CATEGORIES:
        d = demand_predictor.predict_demand(whatif_price, whatif_dow, whatif_month, cat)
        compare_rows.append({
            "category": cat,
            "predicted_demand": round(max(0, d), 2),
            "est_margin": round((whatif_price - UNIT_COST) * max(0, d), 2),
        })
    compare_df = pd.DataFrame(compare_rows)
    fig_bar = px.bar(
        compare_df,
        x="category",
        y="predicted_demand",
        color="category",
        title=f"Demand by category @ ${whatif_price:.0f}",
        text="predicted_demand",
    )
    fig_bar.update_layout(template="plotly_dark", showlegend=False, height=320)
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

elif page == "📊 Strategy comparison":
    st.title("📊 Strategy comparison")
    st.caption("Same market (LightGBM demand) — compare PPO vs fixed-price baselines.")

    n_episodes = st.slider("Episodes per strategy", 3, 30, 10)
    compare_category = st.selectbox(
        "Category",
        PRODUCT_CATEGORIES,
        index=PRODUCT_CATEGORIES.index(category),
        key="compare_cat",
    )

    predictor = demand_predictor if use_ml_demand else None

    strategies: dict[str, Callable] = {
        "Random": lambda obs, env: env.action_space.sample(),
        "Low price": lambda obs, env: 0,
        "Mid price": lambda obs, env: N_PRICE_ACTIONS // 2,
        "High price": lambda obs, env: N_PRICE_ACTIONS - 1,
    }
    if ppo_model is not None:
        strategies["PPO agent"] = lambda obs, env: int(
            ppo_model.predict(obs, deterministic=True)[0]
        )

    if st.button("▶️ Run comparison", type="primary", use_container_width=True):
        results = {name: {"profits": [], "revenues": []} for name in strategies}
        progress = st.progress(0.0)
        total = len(strategies) * n_episodes
        done = 0

        for name, policy in strategies.items():
            for ep in range(n_episodes):
                env = make_simulation_env(predictor, compare_category, seed=ep)
                _, info = run_episode(
                    env,
                    policy,
                    EPISODE_LENGTH,
                    reset_options={"product_category": compare_category},
                    seed=ep,
                )
                results[name]["profits"].append(info["total_profit"])
                results[name]["revenues"].append(info["total_revenue"])
                done += 1
                progress.progress(done / total)

        st.session_state["compare_results"] = results
        progress.empty()

    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]
        summary = []
        for name, data in results.items():
            summary.append({
                "Strategy": name,
                "Avg profit": np.mean(data["profits"]),
                "Std profit": np.std(data["profits"]),
                "Avg revenue": np.mean(data["revenues"]),
            })
        summary_df = pd.DataFrame(summary).sort_values("Avg profit", ascending=False)
        summary_df["Avg profit"] = summary_df["Avg profit"].map(lambda x: f"${x:,.2f}")
        summary_df["Std profit"] = summary_df["Std profit"].map(lambda x: f"${x:,.2f}")
        summary_df["Avg revenue"] = summary_df["Avg revenue"].map(lambda x: f"${x:,.2f}")

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        for name, data in results.items():
            fig.add_trace(
                go.Box(y=data["profits"], name=name, boxpoints="all", jitter=0.3, pointpos=-1.8)
            )
        fig.update_layout(
            title="Profit distribution",
            yaxis_title="Profit ($)",
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

elif page == "📈 Training metrics":
    st.title("📈 PPO training metrics")

    log_files = find_training_logs()
    if not log_files:
        st.warning("No evaluation logs found. Train with: `python -m training.train_ppo`")
    else:
        selected_log = st.selectbox(
            "Training run",
            log_files,
            format_func=lambda p: p.parent.name,
        )
        data = np.load(selected_log)
        timesteps = data["timesteps"]
        results = data["results"]  # (n_evals, n_episodes)
        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        c1, c2 = st.columns(2)
        c1.metric("Eval checkpoints", len(timesteps))
        c2.metric("Best mean reward", f"{mean_rewards.max():.2f}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=mean_rewards,
                mode="lines+markers",
                name="Mean eval reward",
                line=dict(color="#4f9cf9", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([timesteps, timesteps[::-1]]),
                y=np.concatenate([mean_rewards + std_rewards, (mean_rewards - std_rewards)[::-1]]),
                fill="toself",
                fillcolor="rgba(79, 156, 249, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±1 std",
                showlegend=True,
            )
        )
        fig.update_layout(
            title="Evaluation reward during PPO training",
            xaxis_title="Timesteps",
            yaxis_title="Mean episode reward (scaled)",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    if demand_meta:
        st.subheader("Demand model (LightGBM)")
        m1, m2, m3, m4 = st.columns(4)
        metrics = demand_meta.get("metrics", {})
        m1.metric("Test R²", f"{metrics.get('r2', 0):.4f}")
        m2.metric("R² (demand > 0)", f"{metrics.get('r2_demand_positive', 0):.4f}")
        m3.metric("MAPE", f"{metrics.get('mape', 0):.1%}" if metrics.get("mape") is not None else "—")
        m4.metric("RMSE / MAE", f"{metrics.get('rmse', 0):.2f} / {metrics.get('mae', 0):.2f}")
        with st.expander("Feature list & hyperparameters"):
            st.json(demand_meta)


# ---------------------------------------------------------------------------
# Product insights (Direction B — SKU-level from raw aggregates)
# ---------------------------------------------------------------------------

elif page == "🛍️ Product insights":
    st.title("🛍️ Product insights")
    st.caption(
        "Phân tích theo **Product_ID** từ dữ liệu gốc (không dùng model RL). "
        "Chạy `python -m scripts.preprocess_products` nếu chưa có file processed."
    )

    product_monthly = load_product_monthly()
    product_daily = load_product_daily()

    if product_monthly.empty:
        st.error(
            f"Chưa có `product_monthly.csv`. Chạy:\n\n"
            f"`python -m scripts.preprocess_products` hoặc\n"
            f"`python -m scripts.preprocess_ecommerce --with-products`"
        )
        st.stop()

    years = sorted(product_monthly["year"].unique())
    insight_category = st.selectbox(
        "Category",
        PRODUCT_CATEGORIES,
        index=PRODUCT_CATEGORIES.index(category),
        key="insight_category",
    )
    insight_year = st.selectbox("Year", years, index=len(years) - 1, key="insight_year")

    period_mode = st.radio(
        "Time window",
        ["Single month", "Full year"],
        horizontal=True,
    )
    insight_month = None
    if period_mode == "Single month":
        months_available = sorted(
            product_monthly.loc[
                (product_monthly["category"] == insight_category)
                & (product_monthly["year"] == insight_year),
                "month",
            ].unique()
        )
        if not months_available:
            st.warning("No data for this category/year.")
            st.stop()
        insight_month = st.selectbox(
            "Month",
            months_available,
            format_func=lambda m: MONTH_NAMES[int(m) - 1],
            key="insight_month",
        )

    top_n = st.slider("Top N products", 5, 30, 10)
    sort_metric = st.selectbox(
        "Rank by",
        ["demand", "revenue_proxy", "transaction_count"],
        format_func=lambda x: {
            "demand": "Conversions (sum probability)",
            "revenue_proxy": "Revenue proxy",
            "transaction_count": "Transactions",
        }[x],
    )

    top_df = top_products_in_period(
        product_monthly,
        category=insight_category,
        year=int(insight_year),
        month=int(insight_month) if insight_month else None,
        top_n=top_n,
        sort_by=sort_metric,
    )

    period_label = (
        f"{MONTH_NAMES[int(insight_month) - 1]} {insight_year}"
        if insight_month
        else f"Full year {insight_year}"
    )

    st.subheader(f"Top {top_n} products — {insight_category}, {period_label}")

    if top_df.empty:
        st.info("No products in this selection.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total demand (top N)", f"{top_df['demand'].sum():.0f}")
        c2.metric("Total transactions", f"{int(top_df['transaction_count'].sum())}")
        c3.metric("Top product share", f"{top_df['share_pct'].iloc[0]:.1f}%")

        display_df = top_df.rename(columns={
            "product_id": "Product ID",
            "demand": "Demand",
            "transaction_count": "Transactions",
            "avg_price": "Avg price ($)",
            "revenue_proxy": "Revenue proxy ($)",
            "share_pct": "Share (%)",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig_bar = px.bar(
            top_df,
            x="product_id",
            y=sort_metric,
            title=f"Top products by {sort_metric}",
            labels={"product_id": "Product ID", sort_metric: sort_metric},
            text=sort_metric,
        )
        fig_bar.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Demand trend for one product")

    if not product_daily.empty:
        products_in_cat = sorted(
            product_daily.loc[product_daily["category"] == insight_category, "product_id"].unique()
        )
        selected_product = st.selectbox("Product ID", products_in_cat, key="trend_product")
        trend = product_daily[
            (product_daily["product_id"] == selected_product)
            & (product_daily["category"] == insight_category)
        ].sort_values("date")

        if trend.empty:
            st.info("No daily series for this product.")
        else:
            fig_trend = px.line(
                trend,
                x="date",
                y="demand",
                title=f"Daily demand — {selected_product} ({insight_category})",
                markers=True,
            )
            fig_trend.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("Daily product file not found; only monthly rankings available.")

    with st.expander("Data source note"):
        st.markdown(
            """
            - **demand** = tổng `Purchase Probability` (0/1) theo ngày hoặc tháng  
            - **revenue_proxy** = tổng `effective_price × probability`  
            - Đây là **lịch sử quan sát**, khác với **Live simulation** (model + PPO dự báo tương lai)
            """
        )


# Footer
st.markdown("---")
st.caption(
    "RL Dynamic Pricing | Streamlit + LightGBM + PPO (Stable-Baselines3) | "
    f"Category: **{category}**"
)
