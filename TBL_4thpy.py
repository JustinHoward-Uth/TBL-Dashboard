import math
import re  # this is only used for wafer code extraction and color assignment, not for any user input parsing, so should be safe from injection concerns
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA

st.set_page_config(
    page_title="TBL Dashboard + Stats", layout="wide"
)  # this line of code sets the page title and layout for the Streamlit app. The title is "TBL Dashboard + Stats" and the layout is set to "wide", which allows the content to use the full width of the page. This is a common configuration for dashboards to maximize the space for visualizations and data tables.

CSV_Old = "data/all_devices_summary_9.csv"
CSV_New = "data/BrN_devices_summary_1.csv"


WAFER_COLORS = {
    "L1": "green",
    "L2": "teal",
    "R12": "blue",
    "R3": "red",
    "R4": "red",
    "R6": "purple",
    "R7": "blue",
    "R8": "blue",
}

FALLBACK_COLORS = [
    "orange",
    "cyan",
    "magenta",
    "brown",
    "pink",
    "gray",
    "olive",
    "navy",
    "coral",
    "lime",
]

ANALYSIS_FRIENDLY_COLUMNS = [  # these are columns we know are numeric or mostly numeric and should be offered for analysis by default, but we also dynamically include any other numeric columns in the dataset
    "yield_numeric",
    "Blind_Ch_Imp_numeric",
    "Impedance_kOhm_numeric",
    "CAR_uV_numeric",
    "NoCAR_uV_numeric",
]

GROUPABLE_COLUMNS = [  # these are columns we know are commonly used for grouping in the TBL context, but we also dynamically include any other categorical/object columns in the dataset
    "Amplifier_Board",
    "Wafer_Type",
    "AJP_Pattern",
    "Device_Category",
    "Device_Type_Group",
]


def extract_wafer_code(val):
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().upper()
    patterns = [r"^([A-Z]\d+)", r"^([A-Z]\d+)[\-_ ]", r"([A-Z]\d+)[\-_ ]?\d"]
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            return m.group(1)
    return "Unknown"


def resolve_wafer_color(wafer, assigned_colors):
    if wafer in WAFER_COLORS:
        return WAFER_COLORS[wafer]
    if wafer not in assigned_colors:
        idx = len(assigned_colors) % len(FALLBACK_COLORS)
        assigned_colors[wafer] = FALLBACK_COLORS[idx]
    return assigned_colors[wafer]


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Amplifier_board" in df.columns and "Amplifier_Board" not in df.columns:
        df = df.rename(columns={"Amplifier_board": "Amplifier_Board"})

    if "Yield" in df.columns:
        df["yield_numeric"] = df["Yield"].astype(str).str.rstrip("%").replace("", "0")
        df["yield_numeric"] = pd.to_numeric(df["yield_numeric"], errors="coerce")
    else:
        df["yield_numeric"] = np.nan

    if "Blind_Ch_Imp" in df.columns:
        blind_clean = (
            df["Blind_Ch_Imp"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        )
        df["Blind_Ch_Imp_numeric"] = pd.to_numeric(blind_clean, errors="coerce")
    else:
        df["Blind_Ch_Imp_numeric"] = np.nan

    if "Impedance_kOhm" in df.columns:
        imp_clean = (
            df["Impedance_kOhm"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        )
        df["Impedance_kOhm_numeric"] = pd.to_numeric(imp_clean, errors="coerce")
    else:
        df["Impedance_kOhm_numeric"] = np.nan

    if "CAR_uV" in df.columns:
        df["CAR_uV_numeric"] = pd.to_numeric(df["CAR_uV"], errors="coerce")
    else:
        df["CAR_uV_numeric"] = np.nan

    if "NoCAR_uV" in df.columns:
        df["NoCAR_uV_numeric"] = pd.to_numeric(df["NoCAR_uV"], errors="coerce")
    else:
        df["NoCAR_uV_numeric"] = np.nan

    wafer_source = None
    for candidate in ["Wafer_origin", "Wafer_Device_ID", "Wafer_ID"]:
        if candidate in df.columns:
            wafer_source = candidate
            break
    df["Wafer_Type"] = (
        df[wafer_source].apply(extract_wafer_code) if wafer_source else "Unknown"
    )

    if "Amplifier_Board" in df.columns:
        df["Device_Category"] = df["Amplifier_Board"].astype(str).str[0].str.upper()
    else:
        df["Device_Category"] = "Unknown"

    df["Device_Type_Group"] = df["Device_Category"].apply(
        lambda x: f"{x}-devices" if x != "Unknown" else "Unknown"
    )

    if "AJP_Pattern" not in df.columns:
        df["AJP_Pattern"] = "Unknown"
    df["AJP_Pattern"] = df["AJP_Pattern"].fillna("Unknown").astype(str)

    imp_cols = [c for c in df.columns if c.startswith("Imp_Ch")]
    for col in imp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, imp_cols


def build_imp_long(temp_df, imp_cols):
    id_vars = [
        c
        for c in [
            "Wafer_Type",
            "Device_ID",
            "Amplifier_Board",
            "AJP_Pattern",
            "Device_Type_Group",
        ]
        if c in temp_df.columns
    ]
    imp_long = pd.melt(  # this transforms the dataframe from wide format (with separate columns for each channel) to long format (with one column for channel name and one for impedance value), which is easier for plotting and analysis
        temp_df,
        id_vars=id_vars,
        value_vars=imp_cols,
        var_name="Channel",
        value_name="Imp_Ohm",
    ).dropna(subset=["Imp_Ohm"])
    imp_long["Above_1M"] = imp_long["Imp_Ohm"] >= 1e6
    return imp_long


def make_stats_table(temp_df, metric_col, group_col):
    rows = []
    for group, g in temp_df.groupby(group_col):
        vals = pd.to_numeric(g[metric_col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append(
            {
                "Group": group,
                "N": int(len(vals)),
                "Mean": round(vals.mean(), 4),
                "Median": round(vals.median(), 4),
                "Std": round(vals.std(), 4) if len(vals) > 1 else np.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["Group", "N", "Mean", "Median", "Std"])
    )


def plot_metric_scatter(temp_df, metric_col, metric_label, group_col, title):
    cols = [group_col, metric_col]
    for extra in ["Amplifier_Board", "Device_ID"]:
        if extra in temp_df.columns:
            cols.append(extra)
    plot_df = temp_df[cols].copy().dropna(subset=[group_col, metric_col])
    fig = px.strip(
        plot_df,
        x=group_col,
        y=metric_col,
        color=group_col,
        hover_data=[
            c for c in ["Amplifier_Board", "Device_ID"] if c in plot_df.columns
        ],
        title=title,
    )
    fig.update_traces(jitter=0.28, marker=dict(size=9, opacity=0.75))
    fig.update_layout(
        height=520, xaxis_title=group_col.replace("_", " "), yaxis_title=metric_label
    )
    return fig


def plot_impedance_box_scatter(imp_long, group_col, title):
    assigned_colors = {}
    valid_groups = [
        g
        for g in sorted(imp_long[group_col].dropna().unique().tolist())
        if str(g) != "Unknown"
    ]
    fig = go.Figure()
    positions = {g: i for i, g in enumerate(valid_groups)}
    rng = np.random.default_rng(42)

    for group in valid_groups:
        gdf = imp_long[imp_long[group_col] == group].copy()
        if len(gdf) == 0:
            continue
        base_x = positions[group]
        gdf["x_jittered"] = base_x + rng.uniform(-0.28, 0.28, len(gdf))
        color = (
            resolve_wafer_color(group, assigned_colors)
            if group_col == "Wafer_Type"
            else None
        )

        fig.add_trace(
            go.Box(
                x=[base_x] * len(gdf),
                y=gdf["Imp_Ohm"],
                name=str(group),
                line=dict(color=color) if color else None,
                fillcolor="rgba(0,0,0,0)",
                boxpoints=False,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gdf["x_jittered"],
                y=gdf["Imp_Ohm"],
                mode="markers",
                name=str(group),
                marker=dict(color=color, size=4, opacity=0.6)
                if color
                else dict(size=4, opacity=0.6),
                customdata=np.stack([gdf["Device_ID"].fillna("NA")], axis=-1)
                if "Device_ID" in gdf.columns
                else None,
                hovertemplate="Group: "
                + str(group)
                + "<br>Imp: %{y:.0f} Ω<br>Device: %{customdata[0]}<extra></extra>"
                if "Device_ID" in gdf.columns
                else None,
            )
        )

    fig.add_hline(y=1e6, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(
        title=title,
        height=650,
        yaxis_title="Impedance (Ω)",
        yaxis_type="log",
        xaxis=dict(
            title=group_col.replace("_", " "),
            tickmode="array",
            tickvals=list(positions.values()),
            ticktext=list(positions.keys()),
        ),
    )
    return fig


def plot_kde_violin_like(imp_long, group_col, title):
    assigned_colors = {}
    valid_groups = [
        g
        for g in sorted(imp_long[group_col].dropna().unique().tolist())
        if str(g) != "Unknown"
    ]
    fig = go.Figure()
    positions = {
        g: i for i, g in enumerate(valid_groups)
    }  # this creates a mapping of each group to a numeric position on the x-axis, which is used for plotting the KDE-based violins at the correct locations. The positions are assigned based on the sorted order of the unique groups in the data.

    for group in valid_groups:
        gdf = imp_long[imp_long[group_col] == group]
        valid_imp = gdf["Imp_Ohm"][
            (gdf["Imp_Ohm"] > 0) & (gdf["Imp_Ohm"] < 1e10)
        ].values
        if len(valid_imp) < 2:
            continue
        log_y = np.log10(valid_imp)
        hist, bin_edges = np.histogram(log_y, bins=50, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 0.35 * hist / hist.max() if hist.max() > 0 else hist
        base_x = positions[group]
        y_polygon = np.concatenate([10**centers, (10**centers)[::-1]])
        x_polygon = np.concatenate([base_x - width, base_x + width[::-1]])
        color = (
            resolve_wafer_color(group, assigned_colors)
            if group_col == "Wafer_Type"
            else "rgba(100,150,200,0.5)"
        )

        fig.add_trace(
            go.Scatter(
                x=x_polygon,
                y=y_polygon,
                fill="toself",
                fillcolor=color,
                line=dict(color="black", width=1),
                name=str(group),
                hoverinfo="skip",
            )
        )

    fig.add_hline(y=1e6, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(
        title=title,
        height=650,
        yaxis_title="Impedance (Ω)",
        yaxis_type="log",
        xaxis=dict(
            tickmode="array",
            tickvals=list(positions.values()),
            ticktext=list(positions.keys()),
            title=group_col.replace("_", " "),
        ),
    )
    return fig


def plot_channel_violin_subplots(temp_df, imp_cols, group_col, imp_min, imp_max, title):
    id_vars = [c for c in [group_col, "Amplifier_Board"] if c in temp_df.columns]
    imp_long = pd.melt(
        temp_df,
        id_vars=id_vars,
        value_vars=imp_cols,
        var_name="Channel",
        value_name="Impedance_Value_ohms",
    ).dropna(subset=["Impedance_Value_ohms"])
    imp_long = imp_long[
        (imp_long["Impedance_Value_ohms"] >= imp_min)
        & (imp_long["Impedance_Value_ohms"] <= imp_max)
    ]

    groups = [
        g
        for g in sorted(imp_long[group_col].dropna().unique().tolist())
        if str(g) != "Unknown"
    ]
    n_groups = len(groups)
    if n_groups == 0:
        return go.Figure()

    fig = make_subplots(
        rows=math.ceil(n_groups / 2),
        cols=2,
        subplot_titles=groups,
        vertical_spacing=0.10,
    )

    for i, group in enumerate(groups):
        gdf = imp_long[imp_long[group_col] == group]
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Violin(
                x=gdf["Channel"],
                y=gdf["Impedance_Value_ohms"],
                name=str(group),
                box_visible=True,
                meanline_visible=True,
                points=False,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        if "Amplifier_Board" in gdf.columns:
            for board in gdf["Amplifier_Board"].dropna().astype(str).unique():
                bdf = gdf[gdf["Amplifier_Board"].astype(str) == board]
                fig.add_trace(
                    go.Scatter(
                        x=bdf["Channel"],
                        y=bdf["Impedance_Value_ohms"],
                        mode="markers",
                        marker=dict(size=4, opacity=0.55),
                        name=board,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    for r in range(1, math.ceil(n_groups / 2) + 1):
        for c in range(1, 3):
            fig.add_hline(
                y=1e6, line_dash="dash", line_color="red", line_width=2, row=r, col=c
            )
            fig.update_yaxes(type="log", title_text="Impedance (Ω)", row=r, col=c)
            fig.update_xaxes(title_text="Channel", row=r, col=c)

    fig.update_layout(height=350 * math.ceil(n_groups / 2), title=title)
    return fig


def plot_low_impedance_by_wafer(imp_long, wafer_prefix, threshold=5000):
    mask = imp_long["Wafer_Type"].str.startswith(wafer_prefix, na=False)
    w_data = imp_long[mask].copy()
    if w_data.empty or "Device_ID" not in w_data.columns:
        return go.Figure(), pd.DataFrame()

    bad_devices = (
        w_data[w_data["Imp_Ohm"] < threshold]
        .groupby("Device_ID")
        .size()
        .reset_index(name=f"n_channels_below_{threshold}")
    )
    devices_to_plot = set(bad_devices["Device_ID"])
    plot_df = w_data[w_data["Device_ID"].isin(devices_to_plot)].copy()
    if plot_df.empty:
        return go.Figure(), bad_devices

    device_positions = {dev: i for i, dev in enumerate(sorted(devices_to_plot))}
    rng = np.random.default_rng(42)
    plot_df["x_pos"] = plot_df["Device_ID"].map(device_positions)
    plot_df["x_jittered"] = plot_df["x_pos"] + rng.uniform(-0.3, 0.3, len(plot_df))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["x_jittered"],
            y=plot_df["Imp_Ohm"],
            mode="markers",
            marker=dict(size=5, opacity=0.8),
            name=f"{wafer_prefix} low imp devices",
            customdata=np.stack([plot_df["Device_ID"], plot_df["Channel"]], axis=-1),
            hovertemplate="Device %{customdata[0]}<br>%{customdata[1]}<br>Imp %{y:.0f} Ω<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(
        title=f"{wafer_prefix} Devices with Any Channel < {threshold:,}Ω",
        height=650,
        yaxis_title="Impedance (Ω)",
        yaxis_type="log",
        xaxis=dict(
            title="Device (jittered)",
            tickmode="array",
            tickvals=list(device_positions.values()),
            ticktext=list(device_positions.keys()),
        ),
    )
    return fig, bad_devices


def get_numeric_analysis_columns(df):
    cols = [c for c in ANALYSIS_FRIENDLY_COLUMNS if c in df.columns]
    dynamic = [c for c in df.select_dtypes(include=np.number).columns if c not in cols]
    return cols + dynamic


def get_group_columns(df):
    cols = [c for c in GROUPABLE_COLUMNS if c in df.columns]
    dynamic = [c for c in df.columns if df[c].dtype == "object" and c not in cols]
    return cols + dynamic


def safe_result_text(obj):
    try:
        return obj.as_text()
    except Exception:
        return str(obj)


def run_ttest(data, value_col, group_col, g1, g2):
    d = data[[value_col, group_col]].dropna().copy()
    d[group_col] = d[group_col].astype(str)
    x1 = pd.to_numeric(d[d[group_col] == str(g1)][value_col], errors="coerce").dropna()
    x2 = pd.to_numeric(d[d[group_col] == str(g2)][value_col], errors="coerce").dropna()
    if len(x1) < 2 or len(x2) < 2:
        return {"error": "Need at least 2 values in each group for t-test."}
    stat, p = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    result_df = pd.DataFrame(
        {
            "group": [str(g1), str(g2)],
            "n": [len(x1), len(x2)],
            "mean": [x1.mean(), x2.mean()],
            "std": [x1.std(), x2.std()],
        }
    )
    return {
        "summary": pd.DataFrame({"statistic": [stat], "p_value": [p]}),
        "groups": result_df,
    }


def run_anova(data, value_col, group_col):
    d = data[[value_col, group_col]].dropna().copy()
    d[group_col] = d[group_col].astype(str)
    model = smf.ols(f"{value_col} ~ C({group_col})", data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)
    means = d.groupby(group_col)[value_col].agg(["count", "mean", "std"]).reset_index()
    return {
        "anova": table.reset_index(),
        "group_stats": means,
        "model_text": model.summary().as_text(),
    }


def run_linear_regression(data, y_col, x_col):
    d = data[[y_col, x_col]].dropna().copy()
    model = smf.ols(f"{y_col} ~ {x_col}", data=d).fit()
    out = d.copy()
    out["predicted"] = model.get_prediction(d[[x_col]]).summary_frame()["mean"].values
    return {"data": out, "model_text": model.summary().as_text()}


def run_glm(data, y_col, x_col, family_name):
    d = data[[y_col, x_col]].dropna().copy()
    family_map = {
        "Gaussian": sm.families.Gaussian(),
        "Gamma": sm.families.Gamma(),
        "Poisson": sm.families.Poisson(),
    }
    model = smf.glm(
        formula=f"{y_col} ~ {x_col}", data=d, family=family_map[family_name]
    ).fit()
    out = d.copy()
    out["predicted"] = model.get_prediction(d[[x_col]]).summary_frame()["mean"].values
    return {"data": out, "model_text": model.summary().as_text()}


def run_manova(data, dep_vars, group_col):
    d = data[dep_vars + [group_col]].dropna().copy()
    if len(dep_vars) < 2:
        return {"error": "MANOVA needs at least 2 dependent variables."}
    formula = " + ".join(dep_vars) + f" ~ C({group_col})"
    result = MANOVA.from_formula(formula, data=d).mv_test()
    return {"text": safe_result_text(result), "n_rows": len(d)}


def run_one_sample_ttest(data, value_col, mu_value):
    x = pd.to_numeric(data[value_col], errors="coerce").dropna()
    if len(x) < 2:
        return {"error": "Need at least 2 values for one-sample t-test."}
    stat, p = stats.ttest_1samp(x, popmean=mu_value, nan_policy="omit")
    return {
        "summary": pd.DataFrame(
            {
                "n": [len(x)],
                "sample_mean": [x.mean()],
                "sample_std": [x.std()],
                "test_value": [mu_value],
                "statistic": [stat],
                "p_value": [p],
            }
        )
    }


# df, imp_cols = load_data(CSV_PATH)
df_old, imp_cols_old = load_data(CSV_Old)
df_new, imp_cols_new = load_data(CSV_New)

df = pd.concat([df_old, df_new], ignore_index=True)
imp_cols = sorted(set(imp_cols_old + imp_cols_new))

all_device_categories = sorted(
    [c for c in df["Device_Category"].dropna().unique() if c != "Unknown"]
)
all_patterns = sorted(
    [p for p in df["AJP_Pattern"].dropna().unique() if p != "Unknown"]
)
all_wafers = sorted([w for w in df["Wafer_Type"].dropna().unique() if w != "Unknown"])
all_boards = (
    sorted([b for b in df["Amplifier_Board"].dropna().astype(str).unique()])
    if "Amplifier_Board" in df.columns
    else []
)

st.title("TBL Dashboard + Statistical Testing")
st.caption(
    "Dynamic filtering — handles any device category, wafer, or AJP pattern automatically"
)

with st.sidebar:
    st.header("Filters")

    selected_categories = st.multiselect(
        "Device Categories",
        options=all_device_categories,
        default=all_device_categories,
    )
    selected_wafers = st.multiselect(
        "Wafer Types", options=all_wafers, default=all_wafers
    )
    selected_patterns = st.multiselect(
        "AJP Patterns", options=all_patterns, default=all_patterns
    )

    if all_boards:
        selected_boards = st.multiselect(
            "Amplifier Boards", options=["All"] + all_boards, default=["All"]
        )
    else:
        selected_boards = ["All"]

    view_mode = st.radio(
        "Group plots by",
        options=["Wafer_Type", "Device_Type_Group", "AJP_Pattern", "Device_Category"],
        format_func=lambda x: x.replace("_", " "),
    )

    plot_type = st.selectbox(
        "Plot",
        [
            "yield",
            "blind",
            "car",
            "impedance",
            "imp_violin",
            "box_scatter",
            "kde_violin",
            "low_imp",
        ],
        format_func=lambda x: {
            "yield": "Yield",
            "blind": "Blind Ch Imp",
            "car": "CAR",
            "impedance": "Impedance (device-level)",
            "imp_violin": "Channel Violin Subplots",
            "box_scatter": "Impedance Box + Scatter",
            "kde_violin": "All-Wafer KDE Violin",
            "low_imp": "Devices with Any Channel < 5kΩ",
        }[x],
    )

    yield_min = (
        int(np.nanmin(df["yield_numeric"])) if df["yield_numeric"].notna().any() else 0
    )
    yield_max = (
        int(np.nanmax(df["yield_numeric"]))
        if df["yield_numeric"].notna().any()
        else 100
    )
    yield_threshold = st.slider(
        "Yield ≥ (%)",
        min_value=yield_min,
        max_value=max(yield_max, yield_min + 1),
        value=max(60, yield_min),
    )

    imp_dev_series = df["Impedance_kOhm_numeric"].dropna()
    imp_dev_min = int(imp_dev_series.min()) if not imp_dev_series.empty else 0
    imp_dev_max = int(imp_dev_series.max()) if not imp_dev_series.empty else 1000
    imp_device_range = st.slider(
        "Device Impedance (kΩ)",
        min_value=imp_dev_min,
        max_value=max(imp_dev_max, imp_dev_min + 1),
        value=(imp_dev_min, imp_dev_max),
    )

    blind_series = df["Blind_Ch_Imp_numeric"].dropna()
    blind_min = float(blind_series.min()) if not blind_series.empty else 0.0
    blind_max = float(blind_series.max()) if not blind_series.empty else 1.0
    blind_range = st.slider(
        "Blind Ch Imp",
        min_value=blind_min,
        max_value=max(blind_max, blind_min + 0.01),
        value=(blind_min, blind_max),
    )

    if imp_cols:
        all_imp_vals = pd.concat([df[c] for c in imp_cols]).dropna()
        imp_ch_min = int(all_imp_vals.min()) if not all_imp_vals.empty else 0
        imp_ch_max = int(all_imp_vals.max()) if not all_imp_vals.empty else 1000000
    else:
        imp_ch_min, imp_ch_max = 0, 1000000
    imp_channel_range = st.slider(
        "Channel Impedance (Ω)",
        min_value=imp_ch_min,
        max_value=max(imp_ch_max, imp_ch_min + 1),
        value=(imp_ch_min, imp_ch_max),
    )

temp_df = df.copy()

if selected_categories:
    temp_df = temp_df[temp_df["Device_Category"].isin(selected_categories)]
if selected_wafers:
    temp_df = temp_df[temp_df["Wafer_Type"].isin(selected_wafers)]
if selected_patterns:
    temp_df = temp_df[temp_df["AJP_Pattern"].isin(selected_patterns)]
if (
    selected_boards
    and "All" not in selected_boards
    and "Amplifier_Board" in temp_df.columns
):
    temp_df = temp_df[temp_df["Amplifier_Board"].astype(str).isin(selected_boards)]

temp_df = temp_df[
    (temp_df["yield_numeric"].fillna(-np.inf) >= yield_threshold)
    & (temp_df["Impedance_kOhm_numeric"].fillna(-np.inf) >= imp_device_range[0])
    & (temp_df["Impedance_kOhm_numeric"].fillna(np.inf) <= imp_device_range[1])
    & (temp_df["Blind_Ch_Imp_numeric"].fillna(-np.inf) >= blind_range[0])
    & (temp_df["Blind_Ch_Imp_numeric"].fillna(np.inf) <= blind_range[1])
].copy()

group_col = view_mode

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered rows", f"{len(temp_df):,}")
c2.metric("Wafer types", f"{temp_df['Wafer_Type'].nunique()}")
c3.metric("Device categories", f"{temp_df['Device_Category'].nunique()}")
c4.metric("AJP patterns", f"{temp_df['AJP_Pattern'].nunique()}")

st.markdown("---")
st.subheader("Dashboard")

metric_map = {
    "yield": ("yield_numeric", "Yield (%)"),
    "blind": ("Blind_Ch_Imp_numeric", "Blind Ch Imp"),
    "car": ("CAR_uV_numeric", "CAR (µV)"),
    "impedance": ("Impedance_kOhm_numeric", "Impedance (kΩ)"),
}

if plot_type in metric_map:
    metric_col, metric_label = metric_map[plot_type]
    fig = plot_metric_scatter(
        temp_df,
        metric_col,
        metric_label,
        group_col,
        f"{metric_label} by {group_col.replace('_', ' ')}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        make_stats_table(temp_df, metric_col, group_col), use_container_width=True
    )

elif plot_type == "box_scatter":
    if imp_cols:
        imp_long = build_imp_long(temp_df, imp_cols)
        imp_long = imp_long[
            (imp_long["Imp_Ohm"] >= imp_channel_range[0])
            & (imp_long["Imp_Ohm"] <= imp_channel_range[1])
        ]
        fig = plot_impedance_box_scatter(
            imp_long,
            group_col,
            f"Impedance Box + Scatter by {group_col.replace('_', ' ')}",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            make_stats_table(imp_long, "Imp_Ohm", group_col), use_container_width=True
        )
    else:
        st.warning("No impedance channel columns found.")

elif plot_type == "kde_violin":
    if imp_cols:
        imp_long = build_imp_long(temp_df, imp_cols)
        imp_long = imp_long[
            (imp_long["Imp_Ohm"] >= imp_channel_range[0])
            & (imp_long["Imp_Ohm"] <= imp_channel_range[1])
        ]
        fig = plot_kde_violin_like(
            imp_long, group_col, f"KDE-style Violin by {group_col.replace('_', ' ')}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            make_stats_table(imp_long, "Imp_Ohm", group_col), use_container_width=True
        )
    else:
        st.warning("No impedance channel columns found.")

elif plot_type == "imp_violin":
    if imp_cols:
        fig = plot_channel_violin_subplots(
            temp_df,
            imp_cols,
            group_col,
            imp_channel_range[0],
            imp_channel_range[1],
            f"Channel Impedance Violins by {group_col.replace('_', ' ')}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No impedance channel columns found.")

elif plot_type == "low_imp":
    if imp_cols:
        imp_long = build_imp_long(temp_df, imp_cols)
        wafer_prefixes = sorted(
            set(
                imp_long["Wafer_Type"]
                .dropna()
                .apply(
                    lambda x: (
                        re.match(r"^([A-Z]+)", str(x)).group(1)
                        if re.match(r"^([A-Z]+)", str(x))
                        else None
                    )
                )
                .dropna()
                .unique()
            )
        )
        selected_prefix = st.selectbox("Wafer prefix", wafer_prefixes)
        fig, low_table = plot_low_impedance_by_wafer(
            imp_long, selected_prefix, threshold=5000
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(low_table, use_container_width=True)
    else:
        st.warning("No impedance channel columns found.")

st.markdown("---")
st.subheader("Statistical Testing")

numeric_cols = get_numeric_analysis_columns(temp_df)
group_cols = get_group_columns(temp_df)

stats_col1, stats_col2 = st.columns([1, 1])

with stats_col1:
    test_type = st.selectbox(
        "Test Type",
        [
            "t-test (two groups)",
            "one-sample t-test",
            "ANOVA",
            "Linear Regression",
            "GLM",
            "MANOVA",
        ],
    )
    stat_group_col = st.selectbox(
        "Grouping Column", group_cols, index=0 if group_cols else None
    )
    response_col = st.selectbox(
        "Response Variable", numeric_cols, index=0 if numeric_cols else None
    )

    predictor_col = None
    if test_type in ["Linear Regression", "GLM"]:
        predictor_options = [c for c in numeric_cols if c != response_col]
        predictor_col = st.selectbox(
            "Predictor Variable",
            predictor_options,
            index=0 if predictor_options else None,
        )

    glm_family = (
        st.selectbox("GLM Family", ["Gaussian", "Gamma", "Poisson"])
        if test_type == "GLM"
        else None
    )

    dep_vars = []
    if test_type == "MANOVA":
        default_manova = [
            c
            for c in ["yield_numeric", "CAR_uV_numeric", "Impedance_kOhm_numeric"]
            if c in numeric_cols
        ]
        dep_vars = st.multiselect(
            "Dependent Variables", numeric_cols, default=default_manova
        )

    g1 = g2 = None
    if test_type == "t-test (two groups)" and stat_group_col:
        available_groups = sorted(
            temp_df[stat_group_col].dropna().astype(str).unique().tolist()
        )
        g1 = st.selectbox(
            "Group 1", available_groups, index=0 if available_groups else None
        )
        g2 = st.selectbox(
            "Group 2", available_groups, index=1 if len(available_groups) > 1 else 0
        )

    mu_value = (
        st.number_input("Test Mean", value=0.0)
        if test_type == "one-sample t-test"
        else None
    )
    run_analysis = st.button("Run Analysis")

with stats_col2:
    st.write("Current filtered dataset preview")
    st.dataframe(temp_df.head(20), use_container_width=True)

if run_analysis:
    if len(temp_df) == 0:
        st.error("No data available after filtering.")
    else:
        try:
            if test_type == "t-test (two groups)":
                result = run_ttest(temp_df, response_col, stat_group_col, g1, g2)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.dataframe(result["groups"], use_container_width=True)
                    st.dataframe(result["summary"], use_container_width=True)

            elif test_type == "one-sample t-test":
                result = run_one_sample_ttest(temp_df, response_col, mu_value)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.dataframe(result["summary"], use_container_width=True)

            elif test_type == "ANOVA":
                result = run_anova(temp_df, response_col, stat_group_col)
                st.dataframe(result["group_stats"], use_container_width=True)
                st.dataframe(result["anova"], use_container_width=True)
                st.text(result["model_text"])

            elif test_type == "Linear Regression":
                result = run_linear_regression(temp_df, response_col, predictor_col)
                fig_reg = px.scatter(
                    result["data"], x=predictor_col, y=response_col, opacity=0.7
                )
                fig_reg.add_traces(
                    go.Scatter(
                        x=result["data"][predictor_col],
                        y=result["data"]["predicted"],
                        mode="lines",
                        name="fit",
                    )
                )
                st.plotly_chart(fig_reg, use_container_width=True)
                st.text(result["model_text"])

            elif test_type == "GLM":
                result = run_glm(temp_df, response_col, predictor_col, glm_family)
                fig_glm = px.scatter(
                    result["data"], x=predictor_col, y=response_col, opacity=0.7
                )
                fig_glm.add_traces(
                    go.Scatter(
                        x=result["data"][predictor_col],
                        y=result["data"]["predicted"],
                        mode="lines",
                        name="fit",
                    )
                )
                st.plotly_chart(fig_glm, use_container_width=True)
                st.text(result["model_text"])

            elif test_type == "MANOVA":
                result = run_manova(temp_df, dep_vars, stat_group_col)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(f"Rows used: {result['n_rows']}")
                    st.text(result["text"])

        except Exception as e:
            st.exception(e)

st.markdown("---")
st.subheader("Filtered Data Preview")
st.dataframe(temp_df.head(200), use_container_width=True)
