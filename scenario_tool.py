
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Scenario Tool", layout="wide")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CSV_PATH = "data/all_devices_summary_9.csv"

OUTCOME_OPTIONS = {
    "yield_numeric":          "Yield (%)",
    "Impedance_kOhm_numeric": "Impedance (kΩ)",
    "Blind_Ch_Imp_numeric":   "Blind Channel Impedance",
    "CAR_uV_numeric":         "CAR (µV)",
    "NoCAR_uV_numeric":       "NoCAR (µV)",
    "pct_above_1M":           "% Channels ≥ 1 MΩ",
    "pct_below_100k":         "% Channels < 100 kΩ",
}

FACTOR_OPTIONS = {
    "AJP_Pattern":     "AJP Pattern",
    "Wafer_Type":      "Wafer Type",
    "Amplifier_Board": "Amplifier Board",
    "Device_Category": "Device Category",
}

MIN_N_WARNING = 5

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Amplifier_board" in df.columns and "Amplifier_Board" not in df.columns:
        df = df.rename(columns={"Amplifier_board": "Amplifier_Board"})

    # Yield
    if "Yield" in df.columns:
        df["yield_numeric"] = pd.to_numeric(
            df["Yield"].astype(str).str.rstrip("%").replace("", "0"), errors="coerce"
        )
    else:
        df["yield_numeric"] = np.nan

    # Impedance
    if "Impedance_kOhm" in df.columns:
        df["Impedance_kOhm_numeric"] = pd.to_numeric(
            df["Impedance_kOhm"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )
    else:
        df["Impedance_kOhm_numeric"] = np.nan

    # Blind Ch Imp
    if "Blind_Ch_Imp" in df.columns:
        df["Blind_Ch_Imp_numeric"] = pd.to_numeric(
            df["Blind_Ch_Imp"].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
            errors="coerce",
        )
    else:
        df["Blind_Ch_Imp_numeric"] = np.nan

    # CAR / NoCAR
    for col in ["CAR_uV", "NoCAR_uV"]:
        if col in df.columns:
            df[f"{col}_numeric"] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[f"{col}_numeric"] = np.nan

    # Wafer type
    def extract_wafer(val):
        import re
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip().upper()
        m = re.search(r"^([A-Z]\d+)", s)
        return m.group(1) if m else "Unknown"

    for candidate in ["Wafer_origin", "Wafer_Device_ID", "Wafer_ID"]:
        if candidate in df.columns:
            df["Wafer_Type"] = df[candidate].apply(extract_wafer)
            break
    else:
        df["Wafer_Type"] = "Unknown"

    if "Amplifier_Board" in df.columns:
        df["Device_Category"] = df["Amplifier_Board"].astype(str).str[0].str.upper()
    else:
        df["Device_Category"] = "Unknown"

    if "AJP_Pattern" not in df.columns:
        df["AJP_Pattern"] = "Unknown"
    df["AJP_Pattern"] = df["AJP_Pattern"].fillna("Unknown").astype(str)

    # Channel-level derived metrics
    imp_cols = [c for c in df.columns if c.startswith("Imp_Ch")]
    for col in imp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if imp_cols:
        imp_mat = df[imp_cols]
        total_ch = imp_mat.notna().sum(axis=1)
        above_1m = (imp_mat >= 1e6).sum(axis=1)
        below_100k = (imp_mat < 1e5).sum(axis=1)
        df["pct_above_1M"]   = np.where(total_ch > 0, 100 * above_1m  / total_ch, np.nan)
        df["pct_below_100k"] = np.where(total_ch > 0, 100 * below_100k / total_ch, np.nan)
    else:
        df["pct_above_1M"]   = np.nan
        df["pct_below_100k"] = np.nan

    return df


# ─────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────
def group_stats(vals):
    vals = vals.dropna()
    n = len(vals)
    if n == 0:
        return {"N": 0, "Mean": np.nan, "Median": np.nan, "SD": np.nan,
                "Min": np.nan, "Max": np.nan, "CI_lo": np.nan, "CI_hi": np.nan}
    mean = vals.mean()
    med  = vals.median()
    sd   = vals.std(ddof=1) if n > 1 else np.nan
    se   = sd / math.sqrt(n) if n > 1 else np.nan
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
    ci_lo = mean - t_crit * se if n > 1 else np.nan
    ci_hi = mean + t_crit * se if n > 1 else np.nan
    return {"N": n, "Mean": round(mean, 3), "Median": round(med, 3),
            "SD": round(sd, 3) if not np.isnan(sd) else np.nan,
            "Min": round(vals.min(), 3), "Max": round(vals.max(), 3),
            "CI_lo": round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
            "CI_hi": round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan}


def cohens_d(v1, v2):
    v1, v2 = v1.dropna(), v2.dropna()
    if len(v1) < 2 or len(v2) < 2:
        return np.nan
    pooled_sd = math.sqrt(((len(v1)-1)*v1.std(ddof=1)**2 + (len(v2)-1)*v2.std(ddof=1)**2) /
                           (len(v1) + len(v2) - 2))
    return (v2.mean() - v1.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def run_ttest(v1, v2):
    v1, v2 = v1.dropna(), v2.dropna()
    if len(v1) < 2 or len(v2) < 2:
        return np.nan, np.nan
    t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
    return round(t_stat, 4), round(p_val, 4)


def effect_size_label(d):
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:   return "negligible"
    elif ad < 0.5: return "small"
    elif ad < 0.8: return "medium"
    else:           return "large"


def suggest_next_step(n_base, n_cand, p_val, d):
    if n_base < MIN_N_WARNING or n_cand < MIN_N_WARNING:
        return "⚠️ Too little data — treat as exploratory only"
    if np.isnan(p_val):
        return "⚠️ Could not compute significance — check data"
    if p_val < 0.05 and abs(d) >= 0.5:
        direction = "improvement" if d > 0 else "decline"
        return f"✅ Promising {direction} — consider prospective testing"
    elif p_val < 0.05 and abs(d) < 0.5:
        return "🔶 Statistically significant but effect is small — investigate further"
    else:
        return "⬜ No clear benefit detected — insufficient evidence to switch"


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def comparison_plot(base_vals, cand_vals, base_label, cand_label, outcome_label):
    fig = go.Figure()
    rng = np.random.default_rng(42)

    def add_violin_strip(vals, x_pos, label, color):
        vals = vals.dropna()
        if len(vals) == 0:
            return
        # Violin
        fig.add_trace(go.Violin(
            x=[x_pos] * len(vals),
            y=vals,
            name=label,
            side="positive",
            width=0.6,
            line_color=color,
            fillcolor=color.replace(")", ", 0.18)").replace("rgb", "rgba"),
            box_visible=True,
            meanline_visible=True,
            points=False,
            showlegend=True,
        ))
        # Jitter strip
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        fig.add_trace(go.Scatter(
            x=[x_pos + j for j in jitter],
            y=vals,
            mode="markers",
            marker=dict(color=color, size=7, opacity=0.65,
                        line=dict(width=0.5, color="white")),
            name=label,
            showlegend=False,
        ))

    add_violin_strip(base_vals, 0, base_label, "rgb(30, 120, 180)")
    add_violin_strip(cand_vals, 1, cand_label, "rgb(200, 80, 50)")

    fig.update_layout(
        title=f"{base_label}  vs  {cand_label}",
        height=480,
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=[base_label, cand_label],
            title="Group",
        ),
        yaxis_title=outcome_label,
        violingap=0.3,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return fig


# ─────────────────────────────────────────────
# Interpretation text builder
# ─────────────────────────────────────────────
def build_interpretation(base_label, cand_label, outcome_label,
                          s_base, s_cand, d, p_val):
    n_total = s_base["N"] + s_cand["N"]
    if s_base["N"] == 0 or s_cand["N"] == 0:
        return "⚠️ One or both groups returned no data with the current filters."

    diff = round(s_cand["Mean"] - s_base["Mean"], 3)
    pct_chg = round(100 * diff / s_base["Mean"], 1) if s_base["Mean"] not in (0, np.nan) else np.nan
    direction = "higher" if diff > 0 else "lower"
    es_label = effect_size_label(d)

    lines = [
        f"**Historically**, *{cand_label}* shows a mean {outcome_label} of **{s_cand['Mean']}** "
        f"vs **{s_base['Mean']}** for *{base_label}* — a difference of "
        f"**{diff:+.3f}** ({direction}" +
        (f", {abs(pct_chg):.1f}% change" if not np.isnan(pct_chg) else "") + ").",
        f"This comparison is based on **{n_total} total devices** "
        f"({s_base['N']} baseline, {s_cand['N']} candidate).",
    ]

    if not np.isnan(p_val):
        sig_str = f"p = {p_val}" + (" (statistically significant)" if p_val < 0.05 else " (not significant)")
        lines.append(f"Welch's t-test: {sig_str}. Cohen's d = {round(d, 3) if not np.isnan(d) else 'N/A'} ({es_label} effect).")

    if s_cand["SD"] and s_base["SD"] and not np.isnan(s_cand["SD"]) and not np.isnan(s_base["SD"]):
        if s_cand["SD"] > 1.5 * s_base["SD"]:
            lines.append("⚠️ The candidate group has notably **higher variability** — a larger prospective sample may be needed to confirm this result.")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# Suggested power analysis params
# ─────────────────────────────────────────────
def power_analysis_hint(s_base, s_cand, d):
    if np.isnan(d) or s_base["SD"] is np.nan or s_cand["SD"] is np.nan:
        return None
    pooled_sd = math.sqrt((
        max(s_base["N"]-1,1) * (s_base["SD"] or 0)**2 +
        max(s_cand["N"]-1,1) * (s_cand["SD"] or 0)**2
    ) / max(s_base["N"] + s_cand["N"] - 2, 1))
    return {
        "observed_d": round(d, 3),
        "pooled_sd":  round(pooled_sd, 3),
        "baseline_mean": s_base["Mean"],
        "candidate_mean": s_cand["Mean"],
    }


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
st.title("🔬 Scenario Tool")
st.caption("Historical comparison + planning aid for experimental setup decisions")

# ── Data source ──────────────────────────────
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload CSV (optional)", type=["csv"])

if upload is not None:
    raw_df = pd.read_csv(upload)
    df = load_data.__wrapped__(raw_df) if hasattr(load_data, "__wrapped__") else None
    # fallback: re-use the same preprocessing inline via a helper
    # (simpler: just write a separate uncached version)
    @st.cache_data
    def _process(uploaded):
        return load_data.__wrapped__(uploaded)
    # Since load_data reads a path, we preprocess inline for uploads
    import io
    df = load_data(io.StringIO(upload.getvalue().decode("utf-8")))
else:
    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"No CSV found at `{CSV_PATH}`. Upload a file using the sidebar.")
        st.stop()


# ── Step 1: Comparison factor ────────────────
st.header("① Comparison Setup")

col1, col2 = st.columns(2)
with col1:
    factor_key = st.selectbox(
        "Compare by",
        list(FACTOR_OPTIONS.keys()),
        format_func=lambda x: FACTOR_OPTIONS[x],
        help="The variable that defines your two groups",
    )

factor_vals = sorted(
    [v for v in df[factor_key].dropna().unique() if str(v) not in ("Unknown", "nan")]
)

with col2:
    outcome_key = st.selectbox(
        "Primary outcome metric",
        [k for k in OUTCOME_OPTIONS if k in df.columns or k in ("pct_above_1M", "pct_below_100k")],
        format_func=lambda x: OUTCOME_OPTIONS[x],
        help="The metric to compare between groups",
    )

# ── Step 2: Baseline & Candidate ────────────
st.subheader("Groups")
col3, col4 = st.columns(2)
with col3:
    default_base = factor_vals[0] if factor_vals else None
    baseline = st.selectbox(
        "Baseline group  (current / standard)",
        factor_vals,
        index=0,
        help="The 'what we do now' condition",
    )
with col4:
    cand_options = [v for v in factor_vals if v != baseline]
    candidate = st.selectbox(
        "Candidate group  (alternative)",
        cand_options,
        index=0 if cand_options else None,
        help="The 'what if we try this instead?' condition",
    )

# ── Step 3: Optional filters ─────────────────
with st.expander("③ Optional filters (keep comparison fair)", expanded=False):
    filter_cols = [k for k in FACTOR_OPTIONS if k != factor_key and k in df.columns]
    active_filters = {}
    f_cols = st.columns(len(filter_cols)) if filter_cols else []
    for i, fc in enumerate(filter_cols):
        with f_cols[i]:
            opts = sorted([v for v in df[fc].dropna().unique() if str(v) not in ("Unknown", "nan")])
            sel = st.multiselect(
                f"Filter: {FACTOR_OPTIONS[fc]}",
                opts,
                default=[],
                help=f"Leave blank to include all {FACTOR_OPTIONS[fc]} values",
            )
            if sel:
                active_filters[fc] = sel

st.divider()

# ── Filter & slice ────────────────────────────
temp = df.copy()
for fc, vals_sel in active_filters.items():
    temp = temp[temp[fc].isin(vals_sel)]

base_vals = temp.loc[temp[factor_key] == baseline, outcome_key]
cand_vals = temp.loc[temp[factor_key] == candidate, outcome_key]
base_vals = pd.to_numeric(base_vals, errors="coerce")
cand_vals = pd.to_numeric(cand_vals, errors="coerce")

s_base = group_stats(base_vals)
s_cand = group_stats(cand_vals)
d      = cohens_d(base_vals, cand_vals)
t_stat, p_val = run_ttest(base_vals, cand_vals)
outcome_label  = OUTCOME_OPTIONS[outcome_key]
base_label     = f"{FACTOR_OPTIONS[factor_key]}: {baseline}"
cand_label     = f"{FACTOR_OPTIONS[factor_key]}: {candidate}"

# ── Data sufficiency warnings ─────────────────
if s_base["N"] < MIN_N_WARNING or s_cand["N"] < MIN_N_WARNING:
    st.warning(
        f"**Low sample size warning:** "
        f"baseline N = {s_base['N']}, candidate N = {s_cand['N']}. "
        f"Treat this comparison as exploratory — at least {MIN_N_WARNING} devices per group is recommended."
    )

if s_base["N"] == 0 and s_cand["N"] == 0:
    st.error("No data found for either group with these filters. Adjust your selections.")
    st.stop()

# ── Output 1: Summary table ───────────────────
st.header("📊 Results")

def fmt(v, is_n=False):
    if is_n:
        return str(int(v)) if not np.isnan(v) else "—"
    return f"{v:.3f}" if not np.isnan(v) else "—"

diff_mean = round(s_cand["Mean"] - s_base["Mean"], 3) if not (np.isnan(s_cand["Mean"]) or np.isnan(s_base["Mean"])) else np.nan
pct_chg   = round(100 * diff_mean / s_base["Mean"], 1) if (not np.isnan(diff_mean) and s_base["Mean"] not in (0, np.nan)) else np.nan

summary_data = {
    "Metric": [
        "N devices",
        f"Mean {outcome_label}",
        f"Median {outcome_label}",
        "SD",
        "Min / Max",
        "95% CI for mean",
        "Difference (candidate − baseline)",
        "% Change",
    ],
    f"Baseline: {baseline}": [
        fmt(s_base["N"], is_n=True),
        fmt(s_base["Mean"]),
        fmt(s_base["Median"]),
        fmt(s_base["SD"]),
        f"{fmt(s_base['Min'])} / {fmt(s_base['Max'])}",
        f"[{fmt(s_base['CI_lo'])}, {fmt(s_base['CI_hi'])}]",
        "—",
        "—",
    ],
    f"Candidate: {candidate}": [
        fmt(s_cand["N"], is_n=True),
        fmt(s_cand["Mean"]),
        fmt(s_cand["Median"]),
        fmt(s_cand["SD"]),
        f"{fmt(s_cand['Min'])} / {fmt(s_cand['Max'])}",
        f"[{fmt(s_cand['CI_lo'])}, {fmt(s_cand['CI_hi'])}]",
        f"{diff_mean:+.3f}" if not np.isnan(diff_mean) else "—",
        f"{pct_chg:+.1f}%" if not np.isnan(pct_chg) else "—",
    ],
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ── Output 2: Plot ────────────────────────────
st.plotly_chart(
    comparison_plot(base_vals, cand_vals, baseline, candidate, outcome_label),
    use_container_width=True,
)

# ── Output 3: Interpretation ──────────────────
st.subheader("📝 Interpretation")
st.markdown(build_interpretation(
    base_label, cand_label, outcome_label,
    s_base, s_cand, d, p_val,
))

# ── Output 4: Recommendation panel ───────────
st.subheader("🗺️ Recommendation")
rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    es_label = effect_size_label(d)
    d_str    = f"{d:.3f}" if not np.isnan(d) else "N/A"
    st.metric("Cohen's d (effect size)", d_str, delta=es_label,
              delta_color="off")

with rec_col2:
    p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    sig   = "✅ Significant" if (not np.isnan(p_val) and p_val < 0.05) else "❌ Not significant"
    st.metric("p-value (Welch's t-test)", p_str, delta=sig, delta_color="off")

with rec_col3:
    st.metric("t-statistic", f"{t_stat:.3f}" if not np.isnan(t_stat) else "N/A")

st.info(suggest_next_step(s_base["N"], s_cand["N"], p_val, d))

# ── Output 5: Power analysis handoff ─────────
hint = power_analysis_hint(s_base, s_cand, d)
if hint:
    with st.expander("⚡ Send to Power Analysis planner", expanded=False):
        st.markdown(
            "Use the values below as starting parameters when planning your next prospective run:"
        )
        pa_col1, pa_col2, pa_col3, pa_col4 = st.columns(4)
        with pa_col1:
            st.metric("Observed Cohen's d", hint["observed_d"])
        with pa_col2:
            st.metric("Pooled SD", hint["pooled_sd"])
        with pa_col3:
            st.metric(f"Baseline mean ({outcome_label})", hint["baseline_mean"])
        with pa_col4:
            st.metric(f"Candidate mean ({outcome_label})", hint["candidate_mean"])

        alpha   = st.slider("α (significance level)", 0.01, 0.10, 0.05, 0.01)
        power   = st.slider("Target power (1 − β)", 0.70, 0.99, 0.80, 0.05)
        ratio   = st.slider("N candidate / N baseline ratio", 0.5, 3.0, 1.0, 0.25)

        # Two-sided t-test sample size (approximate)
        if not np.isnan(hint["observed_d"]) and hint["observed_d"] != 0:
            from scipy.stats import norm
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta  = norm.ppf(power)
            n1_est  = math.ceil(((z_alpha + z_beta) ** 2) * (1 + 1/ratio) / (hint["observed_d"] ** 2))
            n2_est  = math.ceil(n1_est * ratio)
            st.success(
                f"**Estimated sample size needed:**  "
                f"**{n1_est} baseline** devices  +  **{n2_est} candidate** devices  "
                f"(α = {alpha}, power = {power:.0%}, d = {hint['observed_d']})"
            )
        else:
            st.warning("Cannot estimate sample size — effect size is zero or unavailable.")

# ── Active filters summary ────────────────────
if active_filters:
    filter_parts = [f"{FACTOR_OPTIONS[k]} ∈ {v}" for k, v in active_filters.items()]
    st.caption("Active filters: " + "  |  ".join(filter_parts))

# ── Raw data preview ──────────────────────────
with st.expander("🔍 Raw data for these groups", expanded=False):
    display_cols = [factor_key, outcome_key] + list(active_filters.keys())
    display_cols += [c for c in ["Device_ID", "Amplifier_Board"] if c in temp.columns]
    display_cols = list(dict.fromkeys(display_cols))
    st.dataframe(
        temp[temp[factor_key].isin([baseline, candidate])][display_cols].reset_index(drop=True),
        use_container_width=True,
    )
