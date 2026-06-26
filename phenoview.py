import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hashlib
import io
from scipy import stats

st.set_page_config(page_title="PhenoView", layout="wide")
# -----------------------------
# Helpers
# -----------------------------
st._config.set_option("theme.primaryColor", "#E7751D")
def build_big_palette():
    pal = []
    #deeper colors
    pal += px.colors.qualitative.D3
    pal += px.colors.qualitative.Plotly
    pal += px.colors.qualitative.G10
    pal += px.colors.qualitative.Dark24
    pal += px.colors.qualitative.Alphabet

    #keep order
    seen = set()
    out = []
    for c in pal:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

PALETTE = build_big_palette()

def clean_feature_names(cols):
    #deal with “CHOL..mg.dL.”
    s = pd.Series(cols, dtype="string")
    s = s.str.replace(r"[()]", "", regex=True)
    s = s.str.replace(r"\.+", " ", regex=True).str.strip()
    s = s.str.replace(r"#", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)

    #unit
    s = s.str.replace(r"\s+K\s*uL$", " (K/uL)", case=False, regex=True)
    s = s.str.replace(r"\s+M\s*uL$", " (M/uL)", case=False, regex=True)
    s = s.str.replace(r"\s+U\s*L$",  " (U/L)",  case=False, regex=True)
    s = s.str.replace(r"\s+mg\s*[/ ]?\s*dL$", " (mg/dL)", case=False, regex=True)
    s = s.str.replace(r"\s+g\s*[/ ]?\s*dL$",  " (g/dL)",  case=False, regex=True)
    s = s.str.replace(r"\s+fL$", " (fL)", case=False, regex=True)
    s = s.str.replace(r"\s+pg$", " (pg)", case=False, regex=True)
    s = s.str.replace(r"\s+mg$", " (mg)", case=False, regex=True)

    return s.tolist()

@st.cache_data
def zscore_df(df_num: pd.DataFrame) -> pd.DataFrame:
    #z-score (x - mean) / sd
    mu = df_num.mean(axis=0, skipna=True)
    sd = df_num.std(axis=0, skipna=True).replace(0, np.nan)
    return (df_num - mu) / sd

def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values (q-values)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    q = sorted_p * n / (np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q = np.minimum(q, 1.0)
    result = np.empty(n)
    result[sorted_idx] = q
    return result

def get_file_sig(uploaded) -> str:
    b = uploaded.getvalue()
    h = hashlib.md5(b).hexdigest()
    return f"upload::{uploaded.name}::{len(b)}::{h}"

@st.cache_data
def load_mouse_csv(path_or_file, _sig: str) -> pd.DataFrame:
    if isinstance(path_or_file, str):
        raw_df = pd.read_csv(path_or_file)
    else:
        raw_df = pd.read_csv(io.BytesIO(path_or_file.getvalue()))

    #delete Unnamed
    raw_df = raw_df.loc[
        :, ~raw_df.columns.astype(str).str.startswith("Unnamed")
    ].copy()

    #delete blank
    raw_df = raw_df.dropna(axis=1, how="all")

    return raw_df

def style_clean_axes(
    fig,
    width=900,
    height=900,
    axis_lw=2,
    tick_fs=18,
    title_fs=22,
    legend_fs=18,
    legend_title_fs=20
):
    fig.update_layout(
        width=int(width),
        height=int(height),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=80, r=40, t=60, b=70),
        font=dict(size=tick_fs, color="black"),
        legend=dict(
            font=dict(size=legend_fs, color="black"),
            title=dict(font=dict(size=legend_title_fs, color="black"))
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=axis_lw,
        ticks="outside",
        ticklen=6,
        tickwidth=axis_lw,
        mirror=False,
        tickfont=dict(size=tick_fs, color="black"),
        title_font=dict(size=title_fs, color="black"),
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=axis_lw,
        ticks="outside",
        ticklen=6,
        tickwidth=axis_lw,
        mirror=False,
        tickfont=dict(size=tick_fs, color="black"),
        title_font=dict(size=title_fs, color="black"),
    )

    return fig

def sidebar_fig_height(section_title: str, prefix: str, default_h: int = 900):
    st.sidebar.subheader(section_title)

    preset = st.sidebar.selectbox(
        "Height preset (px)",
        ["650", "750", "900", "1100", "1400", "Custom"],
        index=["650", "750", "900", "1100", "1400", "Custom"].index(str(default_h) if str(default_h) in ["650","750","900","1100","1400"] else "900"),
        key=f"{prefix}_h_preset"
    )

    if preset != "Custom":
        h = int(preset)
    else:
        h = st.sidebar.number_input("Height (px)", 400, 3000, default_h, 50, key=f"{prefix}_h_custom")

    return h

def detect_pair_col(columns):
    """
    Detect a pairing column from uploaded CSV, case-insensitive.
    Accept common names such as PairID / pairid / pair / replicate / set / matchedgroup.
    """
    cols_lower_map = {str(c).strip().lower(): c for c in columns}

    candidates = [
        "pairid", "pair_id", "pair", "replicate", "replicateid", "rep_id",
        "set", "setid", "matchedgroup", "matched_group", "matchid", "pairing"
    ]

    for cand in candidates:
        if cand in cols_lower_map:
            return cols_lower_map[cand]

    return None

import re

def natural_sort_key(val):
    s = str(val).strip()
    parts = re.split(r'(\d+)', s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)

@st.cache_data
def make_safe_distance(corr_df: pd.DataFrame) -> pd.DataFrame:
    D = 1 - corr_df
    D = D.fillna(1.0)

    D = (D + D.T) / 2
    D = D.clip(lower=0, upper=2)

    D_values = D.to_numpy(copy=True)
    np.fill_diagonal(D_values, 0.0)
    return pd.DataFrame(D_values, index=D.index, columns=D.columns)

PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "PhenoView_plot",
        "scale": 3
    }
}

def svg_download_button(fig, filename: str, label: str = "Download SVG"):
    try:
        svg_bytes = fig.to_image(format="svg")
        st.download_button(
            label=label,
            data=svg_bytes,
            file_name=filename,
            mime="image/svg+xml",
        )
    except Exception as e:
        st.error(f"SVG export failed: {e}")

# WCM theme colors
WCM_RED = "#B31B1B"
WCM_ORANGE_DARK = "#CF4520"   # dark orange
WCM_ORANGE_BRIGHT = "#E7751D" # bright orange
WCM_WHITE = "#FFFFFF"
WCM_BLACK = "#000000"

st.markdown(
    f"""
    <style>
    /* ---------- Base ---------- */
    .stApp {{
        background: {WCM_WHITE};
        color: {WCM_BLACK};
    }}

    /* ---------- Sidebar look ---------- */
    section[data-testid="stSidebar"] {{
        background: #FFF7F4;
        border-right: 2px solid {WCM_ORANGE_BRIGHT};
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {WCM_ORANGE_DARK} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
        color: {WCM_BLACK} !important;
        font-weight: 600;
    }}

    /* ---------- Main page headings ---------- */
    h1 {{
        color: {WCM_RED} !important;
    }}

    h3 {{
        color: {WCM_RED} !important;
    }}

    .stCaption {{
        color: #444;
    }}

    /* ---------- Buttons ---------- */
    div.stButton > button {{
        background-color: {WCM_ORANGE_DARK};
        color: {WCM_WHITE};
        border: 1px solid {WCM_ORANGE_DARK};
        border-radius: 8px;
    }}
    div.stButton > button:hover {{
        background-color: {WCM_RED};
        border: 1px solid {WCM_RED};
        color: {WCM_WHITE};
    }}

    /* ---------- Custom badge ---------- */
    .pv-badge {{
      background: #E7751D;
      color: #CF4520;
      border: 0;
      padding: 10px 12px;
      border-radius: 10px;
      font-weight: 400;
      display: inline-block;
      margin: 6px 0 10px 0;
    }}

    /* ---------- MultiSelect tags ---------- */
    div[data-testid="stMultiSelect"] [data-baseweb="tag"],
    div[data-testid="stMultiSelect"] button[data-baseweb="tag"]{{
      background-color: #E7751D !important;
      color: #FFFFFF !important;
      border: 0 !important;
      box-shadow: none !important;
    }}

    div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
    div[data-testid="stMultiSelect"] button[data-baseweb="tag"] *{{
      color: #FFFFFF !important;
    }}

    div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
    div[data-testid="stMultiSelect"] button[data-baseweb="tag"] svg{{
      fill: #FFFFFF !important;
    }}

    /* ---------- Banner ---------- */
    .pv-banner{{
      width: 100%;
      box-sizing: border-box;
      background: #FFF7F4;
      color: #CF4520;
      padding: 10px 12px;
      border-radius: 10px;
      margin: 6px 0 10px 0;
      font-weight: 400;
    }}

    /* ---------- Label text: black, no background ---------- */
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stRadio"] label p {{
        color: {WCM_BLACK} !important;
        background: transparent !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# UI -Data loading
# -----------------------------
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

col_logo, col_text = st.columns([1.3, 4.7])

with col_logo:
    st.image("phenoview_logo.png", width=220)

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is None:
    with col_text:
        st.markdown(
            """
            <div style="margin-top: 155px; font-size: 16px; color: #555; line-height: 1.6;">
                Explore features, group differences, heatmaps, correlations, and PCA/UMAP from CSV files.
                If paired grouped plots are needed, include an additional column such as PairID in the uploaded CSV.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="pv-banner">
        <b>ℹ️ Expected data format</b><br><br>
        This app assumes <i>tidy data</i>:
        <ul>
          <li>Each row = one sample</li>
          <li>Group/condition (e.g. genotype, treatment) must be stored in a <b>separate column</b></li>
        </ul>
        If paired grouped plots are needed, please include an extra column such as
        <code>PairID</code> or <code>Replicate</code> so matched samples can be grouped together.<br><br>
        If group information is encoded in column names or sample IDs
        (e.g. <code>Ctl1</code>, <code>Maf2</code>, <code>Control_LKS.SLAM_1</code>),
        please reshape or split the data into explicit metadata columns before uploading.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


#cache + reset UI
file_sig = get_file_sig(uploaded)
raw_df = load_mouse_csv(uploaded, file_sig)

#clean
if st.session_state.get("file_sig") != file_sig:
    st.session_state["file_sig"] = file_sig

    #clean @st.cache_data
    st.cache_data.clear()

    #clean old key
    keys_to_clear = [
        # correlation page
        "corr_sample_select", "corr_feature_select", "corr_mode_radio",
        # embedding page
        "embed_samples", "embed_method", "embed_color_one", "embed_color_all",
        "umap_neighbors", "umap_mindist",
        # metadata selectors
        "Select Sample/Mouse ID column",
        "Select Group/Genotype column",
        "one_per_mouse", "dedup_mode", "dedup_timepoint",
        "cap_on", "cap_n", "cap_method",
        "filter_Genotype", "filter_sex", "filter_timepoint",
        "filter_age_weeks", "filter_age", "filter_Age", "filter_Age_weeks",
        # interactive plots
        "ip_do_ttest", "ip_control_group", "ip_apply_fdr",
        "ip_do_paired_ttest", "ip_paired_control_group", "ip_apply_fdr_paired",
        "ip_show_paired",
        # heatmap
        "heatmap_sample_select", "heatmap_feature_select",
        "hm_cluster_rows", "hm_cluster_cols",
        # dynamic filters
        "enable_filters",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    # also clear dynamic filter keys (dynf_*)
    for k in list(st.session_state.keys()):
        if k.startswith("dynf_"):
            del st.session_state[k]

st.sidebar.header("Metadata setup")

all_cols = list(raw_df.columns)

#SampleID default guess
id_guess = None
for cand in ["SampleID", "sample_id", "SubjectID", "subject_id", "Mouse name", "mouse", "PatientID", "patient_id"]:
    if cand in all_cols:
        id_guess = cand
        break

id_col = st.sidebar.selectbox(
    "Select Sample ID column",
    options=all_cols,
    index=all_cols.index(id_guess) if id_guess in all_cols else 0,
    key="id_col_select"
)

#choose genotype/condition, optional
group_options = ["(none)"] + all_cols
default_group = "Genotype" if "Genotype" in all_cols else ("genotype" if "genotype" in all_cols else "(none)")
group_col = st.sidebar.selectbox(
    "Select Group/Genotype column",
    options=group_options,
    index=group_options.index(default_group) if default_group in group_options else 0,
    key="group_col_select"
)

#select condition column
condition_options = ["(none)"] + all_cols
default_condition = "Condition" if "Condition" in all_cols else ("condition" if "condition" in all_cols else "(none)")

condition_col = st.sidebar.selectbox(
    "Select Condition column",
    options=condition_options,
    index=condition_options.index(default_condition) if default_condition in condition_options else 0,
    key="condition_col_select"
)

pair_col = detect_pair_col(all_cols)

meta_cols = [id_col]

if group_col != "(none)":
    meta_cols.append(group_col)

if condition_col != "(none)" and condition_col not in meta_cols:
    meta_cols.append(condition_col)

if pair_col is not None and pair_col not in meta_cols:
    meta_cols.append(pair_col)

meta = raw_df[meta_cols].copy()

#standardize SubjectID / Group
meta = meta.rename(columns={id_col: "SubjectID"})

if group_col != "(none)":
    meta = meta.rename(columns={group_col: "Group"})
else:
    meta["Group"] = "All"

if condition_col != "(none)":
    meta = meta.rename(columns={condition_col: "Condition"})
else:
    meta["Condition"] = "All"

if pair_col is not None:
    meta = meta.rename(columns={pair_col: "PairID"})
else:
    meta["PairID"] = pd.NA
#defend Sample ID column and Group column not same
if group_col != "(none)" and id_col == group_col:
    st.sidebar.error("❌ Sample ID column and Group/Genotype column cannot be the same. Please select a different column for Group/Genotype or choose (none).")
    st.stop()

#generate SampleID
meta["SampleID"] = np.where(
    meta["Group"].astype(str).eq("All"),
    meta["SubjectID"].astype(str),
    meta["Group"].astype(str) + "_" + meta["SubjectID"].astype(str)
)
#ensure SampleID only timecourse must
if meta["SampleID"].duplicated().any():
    meta["SampleID"] = (
        meta["SampleID"].astype(str)
        + "__"
        + meta.groupby("SampleID").cumcount().add(1).astype(str)
    )
#PCA/UMAP Color by include raw_df all columns
color_df = raw_df.copy()
color_df["SampleID"] = meta["SampleID"].values
color_df = color_df.set_index("SampleID")

# =============================
# Sample filters (reduce sample list)
# =============================
st.sidebar.subheader("Sample filters")

enable_filters = st.sidebar.checkbox(
    "Enable sample filters",
    value=False,
    key="enable_filters"
)

#index = SampleID
df_filt = color_df.copy()

def is_numeric_series(s: pd.Series) -> bool:
    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.notna().mean() > 0.9

def apply_one_filter(df: pd.DataFrame, col: str, slot: int) -> pd.DataFrame:
    if col is None or col == "(none)" or col not in df.columns:
        return df

    s = df[col]

    if is_numeric_series(s):
        s_num = pd.to_numeric(s, errors="coerce")
        s_num_non = s_num.dropna()
        if s_num_non.empty:
            return df

        lo, hi = float(s_num_non.min()), float(s_num_non.max())
        step = (hi - lo) / 100 if hi > lo else 1.0

        rng = st.sidebar.slider(
            f"Filter {slot}: {col} range",
            min_value=lo,
            max_value=hi,
            value=(lo, hi),
            step=step,
            key=f"dynf_{slot}_num_{col}"
        )
        return df[s_num.between(rng[0], rng[1])].copy()

    s_str = s.astype("string").fillna("NA").astype(str)
    uniq = sorted(s_str.unique().tolist())

    if len(uniq) > 300:
        st.sidebar.warning(
            f"Filter {slot}: '{col}' has {len(uniq)} unique values (too many)."
        )
        return df

    picked = st.sidebar.multiselect(
        f"Filter {slot}: {col}",
        options=uniq,
        default=uniq,
        key=f"dynf_{slot}_cat_{col}"
    )
    return df[s_str.isin(picked)].copy()


# -----------------------------
# 1) Basic dynamic filters
# -----------------------------
if enable_filters:
    candidate_cols = [c for c in df_filt.columns if c not in ["SampleID"]]

    chosen_cols = []
    for slot in [1, 2, 3]:
        available = ["(none)"] + [c for c in candidate_cols if c not in chosen_cols]
        col_pick = st.sidebar.selectbox(
            f"Choose Filter {slot} column",
            options=available,
            index=0,
            key=f"dynf_{slot}_colpick"
        )
        if col_pick != "(none)":
            chosen_cols.append(col_pick)
            df_filt = apply_one_filter(df_filt, col_pick, slot)

st.sidebar.caption(f"After basic filters: {len(df_filt)}")

# -----------------------------
#Per-Group cap
# -----------------------------
use_cap = st.sidebar.checkbox(
    "Use per-Group cap",
    value=False,
    key="cap_on"
)

if use_cap:
    cap_n = st.sidebar.number_input(
        "Max per Group",
        min_value=5, max_value=500, value=50, step=5,
        key="cap_n"
    )
    cap_method = st.sidebar.selectbox(
        "Cap method",
        ["Random", "First (as-is)"],
        key="cap_method"
    )

    group_for_cap = None
    if group_col != "(none)" and group_col in df_filt.columns:
        group_for_cap = group_col
    elif "Group" in df_filt.columns:
        group_for_cap = "Group"

    if group_for_cap:
        if cap_method == "Random":
            df_filt = (
                df_filt.groupby(group_for_cap, group_keys=False)
                       .apply(lambda g: g.sample(n=min(len(g), cap_n), random_state=0))
            )
        else:
            df_filt = (
                df_filt.groupby(group_for_cap, group_keys=False)
                       .head(cap_n)
            )

st.sidebar.caption(f"After per-Group cap: {len(df_filt)}")


# -----------------------------
#Final (align everything by SampleID)
# -----------------------------
allowed_samples = df_filt.index.tolist()

# meta / raw_df
meta = meta.copy()
meta = meta.set_index("SampleID", drop=False)

# raw_df + SampleID -> index order same as meta
raw_df2 = raw_df.copy()
raw_df2["SampleID"] = meta["SampleID"].values
raw_df2 = raw_df2.set_index("SampleID", drop=True)

#filter by allowed_samples
raw_df2 = raw_df2.loc[allowed_samples].copy()
meta = meta.loc[allowed_samples].copy()
color_df = raw_df2.copy()

st.sidebar.caption(f"Samples (final): {len(allowed_samples)}")

# features: delete metadata from raw_df2
drop_cols = [id_col]
if group_col != "(none)":
    drop_cols.append(group_col)
if condition_col != "(none)":
    drop_cols.append(condition_col)
if pair_col is not None:
    drop_cols.append(pair_col)

feat = raw_df2.drop(columns=[c for c in drop_cols if c in raw_df2.columns], errors="ignore").copy()

def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    s = s.astype("string").str.strip()
    s = s.replace("", pd.NA)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

X = feat.apply(to_num, axis=0)
if isinstance(X, pd.Series):
    X = X.to_frame()

X = X.loc[:, X.notna().sum(axis=0) > 0].copy()
X = X.apply(pd.to_numeric, errors="coerce")

Xz = zscore_df(X)
# Pretty feature names map (for display)
pretty_cols = clean_feature_names(list(X.columns))
pretty_map = dict(zip(X.columns, pretty_cols))
st.markdown(
    f'<div class="pv-banner">Loaded {X.shape[0]} samples × {X.shape[1]} features</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select view",
    ["Preview", "PCA / UMAP", "Correlation heatmaps", "Interactive plots", "Heatmap (z-scored)"],
)

# -----------------------------
# Page: Preview
# -----------------------------
if page == "Preview":
    st.subheader("Data preview")
    st.write("Metadata (first 10 columns):")
    st.dataframe(meta.head(10))
    st.write("Features (first 10 columns):")
    st.dataframe(X.head(10).iloc[:, :10])

# -----------------------------
# Page: Interactive plots (bars / dots / violin)
# -----------------------------
elif page == "Interactive plots":
    st.subheader("Interactive plots")

    # --- Figure size (inches x inches) like PCA/UMAP ---
    st.sidebar.subheader("Interactive plot size")
    ip_size = st.sidebar.selectbox(
        "Size preset (inches)",
        ["4×3", "6×4", "6×6", "7×5", "8×6", "10×7.5", "Custom"],
        index=0,  # default 4×3
        key="ip_fig_inch_preset"
    )
    ip_dpi = st.sidebar.slider("DPI (interactive)", 72, 300, 150, 10, key="ip_fig_dpi")

    if ip_size != "Custom":
        w_in, h_in = ip_size.split("×")
        w_in, h_in = float(w_in), float(h_in)
    else:
        w_in = st.sidebar.number_input("Width (in)", 2.0, 20.0, 6.0, 0.5, key="ip_fig_w_in_custom")
        h_in = st.sidebar.number_input("Height (in)", 2.0, 20.0, 6.0, 0.5, key="ip_fig_h_in_custom")

    ip_w = int(w_in * ip_dpi)
    ip_h = int(h_in * ip_dpi)

    #choose parameter to draw
    feat_options = list(X.columns)
    if not feat_options:
        st.warning("No numeric features available. Please check your data.")
        st.stop()

    feat_choice = st.selectbox(
        "Choose a parameter",
        options=feat_options,
        format_func=lambda c: pretty_map.get(c, c),
    )

    if feat_choice is None or feat_choice not in X.columns:
        st.warning("Please select a valid feature before generating the plot.")
        st.stop()

    plot_type = st.radio("Plot type", ["Dots", "Bar (mean ± SEM)", "Violin"], horizontal=True)

    df_plot = pd.DataFrame({
        "SampleID": meta["SampleID"].values,
        "Group": meta["Group"].values,
        "Condition": meta["Condition"].values,
        "PairID": meta["PairID"].values if "PairID" in meta.columns else pd.NA,
        "Value": X[feat_choice].values
    }).dropna(subset=["Value"])

    df_plot["Group"] = df_plot["Group"].astype(str).str.strip()

    has_real_condition = (
            "Condition" in df_plot.columns and
            df_plot["Condition"].astype(str).nunique() > 1 and
            not df_plot["Condition"].astype(str).eq("All").all()
    )

    has_pair_plot = (
            "PairID" in df_plot.columns and
            df_plot["PairID"].notna().any() and
            df_plot["Group"].nunique() >= 2
    )

    # -----------------------------
    # Optional: t-test vs a chosen control group
    # -----------------------------
    st.markdown("### Overall statistics (optional)")

    group_list = sorted(df_plot["Group"].astype(str).unique().tolist())

    do_ttest = st.checkbox("Run t-test (each group vs control)", value=False, key="ip_do_ttest")

    if do_ttest:
        control_group = st.selectbox(
            "Select control group for t-test",
            options=group_list,
            index=0,
            key="ip_control_group"
        )

        has_real_condition = (
                "Condition" in df_plot.columns and
                df_plot["Condition"].astype(str).nunique() > 1 and
                not df_plot["Condition"].astype(str).eq("All").all()
        )

        rows = []
        df_t = pd.DataFrame(columns=["Group", "n_control", "n_group", "t", "p"])

        if has_real_condition:
            condition_list = sorted(df_plot["Condition"].astype(str).unique().tolist())

            for cond in condition_list:
                sub_df = df_plot[df_plot["Condition"].astype(str) == cond].copy()

                x0 = sub_df.loc[sub_df["Group"].astype(str) == control_group, "Value"].dropna().values

                if len(x0) < 2:
                    for gname in group_list:
                        if gname == control_group:
                            continue
                        rows.append({
                            "Condition": cond,
                            "Group": gname,
                            "n_control": len(x0),
                            "n_group": np.nan,
                            "t": np.nan,
                            "p": np.nan
                        })
                    continue

                for gname in group_list:
                    if gname == control_group:
                        continue

                    x1 = sub_df.loc[sub_df["Group"].astype(str) == gname, "Value"].dropna().values

                    if len(x1) < 2:
                        rows.append({
                            "Condition": cond,
                            "Group": gname,
                            "n_control": len(x0),
                            "n_group": len(x1),
                            "t": np.nan,
                            "p": np.nan
                        })
                        continue

                    t_stat, p_val = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")

                    rows.append({
                        "Condition": cond,
                        "Group": gname,
                        "n_control": len(x0),
                        "n_group": len(x1),
                        "t": t_stat,
                        "p": p_val
                    })

            if rows:
                df_t = pd.DataFrame(rows).sort_values(
                    by=["Condition", "p"],
                    na_position="last"
                ).reset_index(drop=True)

        else:
            x0 = df_plot.loc[df_plot["Group"].astype(str) == control_group, "Value"].dropna().values

            if len(x0) < 2:
                st.warning("Control group needs at least 2 samples for t-test.")
            else:
                for gname in group_list:
                    if gname == control_group:
                        continue

                    x1 = df_plot.loc[df_plot["Group"].astype(str) == gname, "Value"].dropna().values

                    if len(x1) < 2:
                        rows.append({
                            "Group": gname,
                            "n_control": len(x0),
                            "n_group": len(x1),
                            "t": np.nan,
                            "p": np.nan
                        })
                        continue

                    t_stat, p_val = stats.ttest_ind(x1, x0, equal_var=False, nan_policy="omit")

                    rows.append({
                        "Group": gname,
                        "n_control": len(x0),
                        "n_group": len(x1),
                        "t": t_stat,
                        "p": p_val
                    })

                if rows:
                    df_t = pd.DataFrame(rows).sort_values("p", na_position="last").reset_index(drop=True)

        if len(df_t) == 0:
            st.info("Please select at least two groups to run a t-test.")
        elif len(df_t) > 0:
            n_tests = int(df_t["p"].notna().sum())
            if n_tests > 1:
                st.warning(
                    f"\u26a0\ufe0f {n_tests} comparisons tested simultaneously \u2014 "
                    "p-values are uncorrected. Enable FDR correction below or interpret with caution."
                )
            apply_fdr = st.checkbox(
                "Apply Benjamini-Hochberg (BH) FDR correction",
                value=False,
                key="ip_apply_fdr"
            )
            if apply_fdr:
                mask = df_t["p"].notna()
                df_t.loc[mask, "p_adj (BH)"] = bh_correct(df_t.loc[mask, "p"].values)
            st.dataframe(df_t, use_container_width=True, hide_index=True)

    group_levels = sorted(df_plot["Group"].astype(str).unique().tolist())

    pal = PALETTE
    group_color_map = {g: pal[i % len(pal)] for i, g in enumerate(group_levels)}

    category_orders = {"Group": group_levels}

    df_plot["Group"] = pd.Categorical(df_plot["Group"].astype(str), categories=group_levels, ordered=True)

    y_title = pretty_map.get(feat_choice, feat_choice)

    def hex_to_rgba(hex_color: str, alpha: float = 0.75) -> str:
        hex_color = str(hex_color).lstrip("#")
        if len(hex_color) != 6:
            return f"rgba(0,0,0,{alpha})"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


    if plot_type == "Dots":
        fig = px.strip(
            df_plot, x="Group", y="Value",
            color="Group",
            color_discrete_map=group_color_map,
            category_orders=category_orders,
            hover_name="SampleID",
            title=f"{y_title} by Group"
        )
        fig.update_yaxes(title=y_title)
        fig.update_xaxes(title="Group")
        fig = style_clean_axes(fig, width=ip_w, height=ip_h)
        hm_config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "PhenoView_dots_plot",
                "scale": 1
            }
        }
        st.plotly_chart(fig, use_container_width=False, config=hm_config)
        svg_download_button(fig, "PhenoView_dots_plot.svg")


    elif plot_type == "Bar (mean ± SEM)":

        fig = go.Figure()

        if has_real_condition:

            g = df_plot.groupby(["Condition", "Group"])["Value"]

            df_sum = pd.DataFrame({

                "Mean": g.mean(),

                "SEM": g.std(ddof=1) / np.sqrt(g.count())

            }).reset_index()

            condition_levels = df_plot["Condition"].astype(str).drop_duplicates().tolist()

            group_levels = df_plot["Group"].astype(str).drop_duplicates().tolist()

            for gname in group_levels:
                sub = df_sum[df_sum["Group"].astype(str) == gname]

                fig.add_trace(go.Bar(

                    x=sub["Condition"].astype(str),

                    y=sub["Mean"],

                    name=gname,

                    marker=dict(

                        color=hex_to_rgba(group_color_map.get(gname), alpha=0.75),

                        line=dict(color="black", width=1)

                    ),

                    error_y=dict(

                        type="data",

                        array=sub["SEM"].fillna(0),

                        visible=True

                    )

                ))

            fig.update_layout(

                title=f"{y_title} (mean ± SEM)",

                barmode="group"

            )

            fig.update_xaxes(

                title="Condition",

                categoryorder="array",

                categoryarray=condition_levels

            )


        else:

            g = df_plot.groupby("Group")["Value"]

            df_sum = pd.DataFrame({

                "Group": g.mean().index,

                "Mean": g.mean().values,

                "SEM": (g.std(ddof=1) / np.sqrt(g.count())).values

            })

            df_sum["Group"] = df_sum["Group"].astype(str).str.strip()

            df_sum["BarColor"] = df_sum["Group"].map(group_color_map).apply(

                lambda c: hex_to_rgba(c, alpha=0.75)

            )

            fig.add_trace(go.Bar(

                x=df_sum["Group"],

                y=df_sum["Mean"],

                name="Mean",

                marker=dict(

                    color=df_sum["BarColor"],

                    line=dict(color="black", width=1)

                )

            ))

            fig.add_trace(go.Scatter(

                x=df_sum["Group"],

                y=df_sum["Mean"],

                error_y=dict(type="data", array=df_sum["SEM"], visible=True),

                mode="markers",

                marker=dict(color="black", size=8),

                showlegend=False

            ))

            fig.update_layout(

                title=f"{y_title} (mean ± SEM)",

                showlegend=False

            )

            fig.update_xaxes(title="Group")

        fig.update_yaxes(title=y_title)

        fig = style_clean_axes(fig, width=ip_w, height=ip_h)

        st.plotly_chart(fig, use_container_width=False, config=PLOTLY_CONFIG)
        svg_download_button(fig, "PhenoView_bar_plot.svg")

    else:  # Violin + box
        fig = px.violin(
            df_plot, x="Group", y="Value",
            color="Group",
            color_discrete_map=group_color_map,
            category_orders=category_orders,
            box=True, points="all",
            hover_name="SampleID",
            title=f"{y_title} by Group"
        )
        fig.update_yaxes(title=y_title)
        fig.update_xaxes(title="Group")
        fig = style_clean_axes(fig, width=ip_w, height=ip_h)
        st.plotly_chart(fig, use_container_width=False, config=PLOTLY_CONFIG)
        svg_download_button(fig, "PhenoView_violin_plot.svg")

    if has_pair_plot and plot_type != "Violin":
        show_paired = st.checkbox("Show paired grouped plot (requires PairID column)", value=False, key="ip_show_paired")

        if show_paired:
            df_pair = df_plot.dropna(subset=["PairID"]).copy()
            df_pair["PairID"] = df_pair["PairID"].astype(str).str.strip()
            df_pair = df_pair[df_pair["PairID"] != ""].copy()

            if len(df_pair) > 0:
                st.markdown("### Paired grouped plot")

                # -----------------------------
                # Optional: paired t-test using PairID
                # -----------------------------
                st.markdown("#### Paired statistics (optional)")

                do_paired_ttest = st.checkbox(
                    "Run paired t-test using PairID",
                    value=False,
                    key="ip_do_paired_ttest"
                )

                group_levels_pair = sorted(df_pair["Group"].astype(str).unique().tolist())

                if do_paired_ttest:
                    control_group_paired = st.selectbox(
                        "Select control group for paired t-test",
                        options=group_levels_pair,
                        index=0,
                        key="ip_paired_control_group"
                    )

                    min_n_per_group = 2

                    st.caption(
                        "Per-pair t-tests are computed only when both groups within a PairID contain at least 2 observations."
                    )

                    pair_test_rows = []

                    for gname in group_levels_pair:
                        if gname == control_group_paired:
                            continue

                        sub = df_pair[df_pair["Group"].isin([control_group_paired, gname])].copy()

                        for pid, sub_pair in sub.groupby("PairID"):
                            x0 = sub_pair.loc[
                                sub_pair["Group"] == control_group_paired, "Value"
                            ].dropna().values

                            x1 = sub_pair.loc[
                                sub_pair["Group"] == gname, "Value"
                            ].dropna().values

                            if len(x0) < min_n_per_group or len(x1) < min_n_per_group:
                                continue

                            t_stat, p_val = stats.ttest_ind(
                                x1, x0,
                                equal_var=False,
                                nan_policy="omit"
                            )

                            pair_test_rows.append({
                                "PairID": pid,
                                "Comparison": f"{gname} vs {control_group_paired}",
                                "n_control": len(x0),
                                "n_treatment": len(x1),
                                "t": t_stat,
                                "p": p_val,
                            })

                    df_pair_tests = pd.DataFrame(pair_test_rows)

                    if len(df_pair_tests) > 0:
                        df_pair_tests["PairID_str"] = df_pair_tests["PairID"].astype(str).str.strip()
                        df_pair_tests["PairID_sort"] = df_pair_tests["PairID_str"].map(natural_sort_key)

                        df_pair_tests = df_pair_tests.sort_values(
                            by=["Comparison", "PairID_sort"],
                            ignore_index=True
                        ).drop(columns=["PairID_sort", "PairID_str"])

                        st.markdown("##### Per-pair t-test results")
                        n_pair_tests = int(df_pair_tests["p"].notna().sum())
                        if n_pair_tests > 1:
                            st.warning(
                                f"\u26a0\ufe0f {n_pair_tests} comparisons tested simultaneously \u2014 "
                                "p-values are uncorrected. Enable FDR correction below or interpret with caution."
                            )
                        apply_fdr_paired = st.checkbox(
                            "Apply Benjamini-Hochberg (BH) FDR correction",
                            value=False,
                            key="ip_apply_fdr_paired"
                        )
                        if apply_fdr_paired:
                            mask = df_pair_tests["p"].notna()
                            df_pair_tests.loc[mask, "p_adj (BH)"] = bh_correct(
                                df_pair_tests.loc[mask, "p"].values
                            )
                        st.dataframe(df_pair_tests, use_container_width=True, hide_index=True)

                pair_levels = df_pair["PairID"].astype(str).drop_duplicates().tolist()
                group_levels_pair = group_levels

                df_pair["PairID"] = pd.Categorical(df_pair["PairID"], categories=pair_levels, ordered=True)
                df_pair["Group"] = pd.Categorical(df_pair["Group"], categories=group_levels_pair, ordered=True)

                if plot_type == "Dots":
                    fig_pair = go.Figure()

                    for gname in group_levels_pair:
                        sub = df_pair[df_pair["Group"] == gname].copy()
                        fig_pair.add_trace(go.Scatter(
                            x=sub["PairID"].astype(str),
                            y=sub["Value"],
                            mode="markers",
                            name=gname,
                            marker=dict(size=6, color=group_color_map.get(gname)),
                            text=sub["SampleID"],
                            hovertemplate="PairID: %{x}<br>Value: %{y}<br>SampleID: %{text}<extra></extra>"
                        ))

                    fig_pair.update_layout(
                        title=f"{y_title} by PairID and Group",
                        xaxis_title="PairID",
                        yaxis_title=y_title,
                        xaxis=dict(
                            type="category",
                            categoryorder="array",
                            categoryarray=pair_levels,
                            tickmode="array",
                            tickvals=pair_levels,
                            ticktext=pair_levels
                        )
                    )

                    fig_pair = style_clean_axes(fig_pair, width=ip_w, height=ip_h)

                    st.plotly_chart(
                        fig_pair,
                        use_container_width=False,
                        config=PLOTLY_CONFIG,
                        key=f"paired_dots_{feat_choice}"
                    )

                elif plot_type == "Bar (mean ± SEM)":
                    g_pair = df_pair.groupby(["PairID", "Group"], observed=False)["Value"]
                    df_pair_sum = pd.DataFrame({
                        "Mean": g_pair.mean(),
                        "SEM": g_pair.std(ddof=1) / np.sqrt(g_pair.count())
                    }).reset_index()

                    df_pair_sum["PairID"] = df_pair_sum["PairID"].astype(str).str.strip()
                    df_pair_sum["Group"] = df_pair_sum["Group"].astype(str).str.strip()

                    df_pair_sum["PairID"] = pd.Categorical(df_pair_sum["PairID"], categories=pair_levels, ordered=True)
                    df_pair_sum["Group"] = pd.Categorical(df_pair_sum["Group"], categories=group_levels_pair,
                                                          ordered=True)
                    df_pair_sum = df_pair_sum.sort_values(["PairID", "Group"])

                    fig_pair = go.Figure()

                    for gname in group_levels_pair:
                        sub = df_pair_sum[df_pair_sum["Group"] == gname]
                        fig_pair.add_trace(go.Bar(
                            x=sub["PairID"].astype(str),
                            y=sub["Mean"],
                            name=gname,
                            marker=dict(
                                color=hex_to_rgba(group_color_map.get(gname), alpha=0.75),
                                line=dict(color="black", width=1)
                            ),
                            error_y=dict(
                                type="data",
                                array=sub["SEM"].fillna(0),
                                visible=True
                            )
                        ))

                    fig_pair.update_layout(
                        title=f"{y_title} by PairID and Group",
                        xaxis_title="PairID",
                        yaxis_title=y_title,
                        barmode="group",
                        xaxis=dict(
                            type="category",
                            categoryorder="array",
                            categoryarray=pair_levels,
                            tickmode="array",
                            tickvals=pair_levels,
                            ticktext=pair_levels
                        )
                    )

                    fig_pair = style_clean_axes(fig_pair, width=ip_w, height=ip_h)

                    st.plotly_chart(
                        fig_pair,
                        use_container_width=False,
                        config=PLOTLY_CONFIG,
                        key=f"paired_bar_{feat_choice}"
                    )

                else:  # Violin
                    fig_pair = px.violin(
                        df_pair,
                        x="PairID",
                        y="Value",
                        color="Group",
                        box=True,
                        points="all",
                        hover_name="SampleID",
                        color_discrete_map=group_color_map,
                        category_orders={"PairID": pair_levels, "Group": group_levels_pair},
                        title=f"{y_title} by PairID and Group"
                    )

                    fig_pair.update_layout(
                        xaxis=dict(
                            type="category",
                            categoryorder="array",
                            categoryarray=pair_levels,
                            tickmode="array",
                            tickvals=pair_levels,
                            ticktext=pair_levels
                        )
                    )

                    fig_pair.update_xaxes(title="PairID")
                    fig_pair.update_yaxes(title=y_title)
                    fig_pair = style_clean_axes(fig_pair, width=ip_w, height=ip_h)

                    st.plotly_chart(
                        fig_pair,
                        use_container_width=False,
                        config=PLOTLY_CONFIG,
                        key=f"paired_violin_{feat_choice}"
                    )
# -----------------------------
# Page: Heatmap (z-scored feature matrix)
# -----------------------------
elif page == "Heatmap (z-scored)":
    st.subheader("Z-scored Feature Matrix heatmap")
    hm_h = sidebar_fig_height("Heatmap height", prefix="hm", default_h=650)
    hm_w = 1400
    # --- clustering toggles ---
    c1, c2 = st.columns(2)
    with c1:
        cluster_rows = st.checkbox("Cluster samples (cols / x-axis)", value=True, key="hm_cluster_rows")
    with c2:
        cluster_cols = st.checkbox("Cluster features (rows / y-axis)", value=True, key="hm_cluster_cols")

    #option: samples / features
    options_samples = [s for s in allowed_samples if s in Xz.index]

    chosen_samples = st.multiselect(
        "Select samples",
        options=options_samples,
        default=options_samples[:min(50, len(options_samples))],
        key="heatmap_sample_select"
    )

    chosen_features = st.multiselect(
        "Select features",
        options=list(X.columns),
        default=list(X.columns)[:min(30, len(X.columns))],
        format_func=lambda c: pretty_map.get(c, c),
        key="heatmap_feature_select"
    )

    if len(chosen_samples) == 0 or len(chosen_features) == 0:
        st.warning("Please select at least 1 sample and 1 feature.")
        st.stop()

    # matrix
    Z = Xz.loc[chosen_samples, chosen_features].T.copy()
    Z = Z.replace({pd.NA: np.nan}).astype("float64")

    n_feat_before = Z.shape[0]
    n_samp_before = Z.shape[1]

    Z = Z.dropna(axis=0)  # drop features (rows) that contain any NaN

    n_feat_after = Z.shape[0]
    dropped_feat = n_feat_before - n_feat_after

    st.caption(f"Dropped {dropped_feat} features with any NA.")

    try:
        # --- cluster COLUMNS: samples (x-axis) ---
        if cluster_rows and Z.shape[1] >= 2:
            C = Z.corr(method="pearson", min_periods=2)
            D = make_safe_distance(C)

            from scipy.spatial.distance import squareform

            Zlink = linkage(squareform(D.values, checks=False), method="average")
            col_order = leaves_list(Zlink)
            Z = Z.iloc[:, col_order]

        # --- cluster ROWS: features (y-axis) ---
        if cluster_cols and Z.shape[0] >= 2:
            Cf = Z.T.corr(method="pearson", min_periods=2)
            Df = make_safe_distance(Cf)

            from scipy.spatial.distance import squareform

            Zlink_f = linkage(squareform(Df.values, checks=False), method="average")
            row_order = leaves_list(Zlink_f)
            Z = Z.iloc[row_order, :]

        # pretty column names for display (after ordering!)
        y_labels = [pretty_map.get(c, c) for c in Z.index]

        fig = go.Figure(data=go.Heatmap(
            z=Z.values,
            x=Z.columns.tolist(),
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0
        ))
        fig.update_layout(
            title="Z-scored Feature Matrix" + (" (clustered)" if (cluster_rows or cluster_cols) else ""),
            xaxis_title="Samples",
            yaxis_title="Features",
            width=hm_w,
            height=hm_h,
            margin=dict(l=180, r=40, t=60, b=180),
            font=dict(size=18, color="black")
        )

        fig.update_xaxes(
            tickfont=dict(size=18, color="black"),
            title_font=dict(size=22, color="black")
        )

        fig.update_yaxes(
            tickfont=dict(size=18, color="black"),
            title_font=dict(size=22, color="black")
        )

        fig.update_traces(
            colorbar=dict(
                tickfont=dict(size=18, color="black"),
                title_font=dict(size=20, color="black")
            )
        )
        st.plotly_chart(fig, use_container_width=False, config=PLOTLY_CONFIG)
        svg_download_button(fig, "PhenoView_heatmap.svg")
    except Exception as e:
        st.error("Unable to generate heatmap. Please check that valid features and samples are selected.")
        st.caption(f"Technical details: {type(e).__name__}: {e}")
# -----------------------------
# Page: Correlation heatmaps (sample-sample / feature-feature)
# -----------------------------
elif page == "Correlation heatmaps":
    st.subheader("Correlation heatmaps")
    corr_h = sidebar_fig_height("Correlation heatmap height", prefix="corr", default_h=650)

    MAX_SAMPLES = st.sidebar.number_input("Max samples for correlation heatmap", 10, 300, 80, 10)
    MAX_FEATURES = st.sidebar.number_input("Max features for correlation heatmap", 10, 300, 80, 10)

    corr_mode = st.radio(
        "Choose correlation type",
        ["Sample–Sample (Pearson)", "Feature–Feature (Pearson)"],
        horizontal=True,
        key="corr_mode_radio"
    )

    # Guard
    if Xz.shape[0] < 2:
        st.warning("Need at least 2 samples to compute Sample–Sample correlation.")
        st.stop()
    if Xz.shape[1] < 2:
        st.warning("Need at least 2 features to compute Feature–Feature correlation.")
        st.stop()

    #NAType -> np.nan, float
    Xz_num = Xz.replace({pd.NA: np.nan}).astype("float64")

    # ===============================
    # Sample–Sample (Pearson)
    # ===============================
    if corr_mode.startswith("Sample"):

        options_samples = [s for s in allowed_samples if s in Xz_num.index]

        chosen_samples = st.multiselect(
            "Select samples",
            options=options_samples,
            default=options_samples[:min(len(options_samples), 200)],
            key="corr_sample_select"
        )

        if len(chosen_samples) < 2:
            st.warning("Please select at least 2 samples.")
            st.stop()
        if len(chosen_samples) > MAX_SAMPLES:
            st.warning(f"Too many samples selected ({len(chosen_samples)}). "
                       f"Please select ≤ {MAX_SAMPLES} to avoid freezing.")
            st.stop()

        try:
            # 1)use selected sample
            X_sub = Xz_num.loc[chosen_samples]

            X_sub = X_sub.loc[:, X_sub.notna().any(axis=0)]

            # 2)sample–sample correlation
            C = X_sub.T.corr(method="pearson", min_periods=2)
            D = make_safe_distance(C)

            from scipy.spatial.distance import squareform

            Z = linkage(squareform(D.values, checks=False), method="average")
            order = leaves_list(Z)

            labels_ord = [C.index[i] for i in order]
            C_ord = C.loc[labels_ord, labels_ord]

            # 4) heatmap tree
            fig = go.Figure(data=go.Heatmap(
                z=C_ord.values,
                x=labels_ord,
                y=labels_ord,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                zmid=0
            ))
            corr_w = min(max(1200, 60 * len(labels_ord)), 3000)

            fig.update_layout(
                title="Sample–Sample Correlation (Pearson)",
                width=corr_w,
                height=corr_h,
                margin=dict(l=180, r=40, t=60, b=180),
                font=dict(size=16, color="black")
            )

            fig.update_xaxes(
                tickfont=dict(size=18, color="black"),
                title_font=dict(size=22, color="black")
            )

            fig.update_yaxes(
                tickfont=dict(size=18, color="black"),
                title_font=dict(size=22, color="black")
            )

            fig.update_traces(
                colorbar=dict(
                    tickfont=dict(size=18, color="black"),
                    title_font=dict(size=20, color="black")
                )
            )

            corr_config = {
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "PhenoView_sample_sample_correlation",
                    "scale": 1
                }
            }
            st.plotly_chart(fig, use_container_width=False, config=corr_config)
            svg_download_button(fig, "PhenoView_sample_sample_correlation.svg")
        except Exception as e:
            st.error("Unable to generate sample–sample correlation heatmap. Please check your selections.")
            st.caption(f"Technical details: {type(e).__name__}: {e}")

    # ===============================
    # Feature–Feature (Pearson)
    else:

        #add select features
        chosen_features = st.multiselect(
            "Select features",
            options=list(X.columns),
            default=list(X.columns),
            format_func=lambda c: pretty_map.get(c, c),
            key="corr_feature_select"
        )

        if len(chosen_features) < 2:
            st.warning("Please select at least 2 features.")
            st.stop()
        if len(chosen_features) > MAX_FEATURES:
            st.warning(f"Too many features selected ({len(chosen_features)}). "
                       f"Please select ≤ {MAX_FEATURES} to avoid freezing.")
            st.stop()

        try:
            # 1)use selected feature
            X_sub = Xz_num[chosen_features]

            X_sub = X_sub.loc[X_sub.notna().any(axis=1)]

            # 2) feature–feature correlation
            Cf = X_sub.corr(method="pearson", min_periods=2)
            D = make_safe_distance(Cf)

            from scipy.spatial.distance import squareform

            Z = linkage(squareform(D.values, checks=False), method="average")
            order = leaves_list(Z)

            labels = [pretty_map.get(c, c) for c in Cf.columns]
            labels_ord = [labels[i] for i in order]
            Cf_ord = Cf.iloc[order, order]

            # 4) heatmap tree
            fig = go.Figure(data=go.Heatmap(
                z=Cf_ord.values,
                x=labels_ord,
                y=labels_ord,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                zmid=0
            ))
            corr_w = min(max(1200, 60 * len(labels_ord)), 3000)

            fig.update_layout(
                title="Feature–Feature Correlation (Pearson)",
                width=corr_w,
                height=corr_h,
                margin=dict(l=180, r=40, t=60, b=180),
                font=dict(size=16, color="black")
            )

            fig.update_xaxes(
                tickfont=dict(size=18, color="black"),
                title_font=dict(size=22, color="black")
            )

            fig.update_yaxes(
                tickfont=dict(size=18, color="black"),
                title_font=dict(size=22, color="black")
            )

            fig.update_traces(
                colorbar=dict(
                    tickfont=dict(size=18, color="black"),
                    title_font=dict(size=20, color="black")
                )
            )

            corr_config = {
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "PhenoView_feature_feature_correlation",
                    "scale": 1
                }
            }
            st.plotly_chart(fig, use_container_width=False, config=corr_config)
            svg_download_button(fig, "PhenoView_feature_feature_correlation.svg")
        except Exception as e:
            st.error("Unable to generate feature–feature correlation heatmap. Please check your selections.")
            st.caption(f"Technical details: {type(e).__name__}: {e}")
# Page: PCA / UMAP
elif page == "PCA / UMAP":

    st.subheader("PCA / UMAP")

    # ---------- Figure size (inches x inches) ----------
    st.sidebar.subheader("Figure size")

    size_preset = st.sidebar.selectbox(
        "Size preset (inches)",
        ["4×3", "6×6", "7×7", "7×5", "8×6", "10×7.5", "Custom"],
        index=0,  # 4×3
        key="fig_inch_preset"
    )

    dpi = st.sidebar.slider("DPI (for sizing)", 72, 300, 150, 10, key="fig_dpi")

    if size_preset != "Custom":
        w_in, h_in = size_preset.split("×")
        w_in, h_in = float(w_in), float(h_in)
    else:
        w_in = st.sidebar.number_input("Width (in)", 2.0, 20.0, 6.0, 0.5, key="fig_w_in_custom")
        h_in = st.sidebar.number_input("Height (in)", 2.0, 20.0, 6.0, 0.5, key="fig_h_in_custom")

    # inch -> px
    fig_w = int(w_in * dpi)
    fig_h = int(h_in * dpi)

    # ---------- Select samples ----------
    options_samples = [s for s in allowed_samples if s in meta["SampleID"].values]
    chosen_samples = st.multiselect(
        "Select samples",
        options=options_samples,
        default=options_samples[:min(len(options_samples), 300)],
        key="embed_samples"
    )
    if len(chosen_samples) < 3:
        st.warning("Please select at least 3 samples.")
        st.stop()

    # meta index
    meta_idx = meta.set_index("SampleID")

    # subset X by chosen samples (X index is SampleID already in your pipeline)
    X_sub = X.loc[chosen_samples].copy()

    # numeric + NA handling
    X_sub = (
        X_sub.replace({pd.NA: np.nan})
             .apply(pd.to_numeric, errors="coerce")
             .astype("float64")
    )
    X_sub = X_sub.loc[:, X_sub.notna().any(axis=0)]

    keep_rows = X_sub.notna().all(axis=1)
    dropped = (~keep_rows).sum()
    X_sub = X_sub.loc[keep_rows]

    st.caption(f"Dropped {dropped} samples with any NA.")
    idx = X_sub.index

    if len(idx) < 3:
        st.warning("After dropping NA, less than 3 samples remain. Please select more samples.")
        st.stop()

    if X_sub.shape[1] < 2:
        st.warning("PCA/UMAP requires at least 2 numeric features. This dataset currently has fewer than 2 usable features.")
        st.stop()

    if min(X_sub.shape[0], X_sub.shape[1]) < 2:
        st.warning("PCA/UMAP requires at least 2 samples and 2 numeric features.")
        st.stop()

    # Standardize
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_sub.values)

    method = st.radio("Method", ["PCA", "UMAP"], horizontal=True, key="embed_method")

    # Color by
    all_color_cols = list(color_df.columns)
    color_pick = st.selectbox(
        "Color by",
        options=all_color_cols,
        index=all_color_cols.index("Group") if "Group" in all_color_cols else 0,
        key="embed_color_all"
    )
    color_title = pretty_map.get(color_pick, color_pick)
    color_series_raw = color_df.reindex(idx)[color_pick]

    color_is_numeric = pd.api.types.is_numeric_dtype(color_series_raw)

    color_discrete_map = None
    category_orders = None

    if (not color_is_numeric) and (color_series_raw.nunique(dropna=True) <= 200):
        color_series_clean = (
            color_series_raw.astype("string")
            .fillna("NA")
            .str.strip()
            .astype(str)
        )

        levels = sorted(color_series_clean.unique().tolist())
        pal = PALETTE

        color_discrete_map = {lvl: pal[i % len(pal)] for i, lvl in enumerate(levels)}
        category_orders = {color_title: levels}
        color_series_plot = pd.Categorical(color_series_clean, categories=levels, ordered=True)
    else:
        color_series_plot = color_series_raw

    try:
        if method == "PCA":
            emb = PCA(n_components=2, random_state=0).fit_transform(X_scaled)
            df_emb = pd.DataFrame(emb, index=idx, columns=["PC1", "PC2"])
            df_emb["SampleID"] = df_emb.index
            df_emb[color_title] = color_series_plot

            fig = px.scatter(
                df_emb,
                x="PC1", y="PC2",
                color=color_title,
                hover_name="SampleID",
                title="PCA (2D)",
                color_discrete_map=color_discrete_map,
                category_orders=category_orders
            )

        else:
            n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 15, 1, key="umap_neighbors")
            min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.05, key="umap_mindist")

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=0
            )
            emb = reducer.fit_transform(X_scaled)

            df_emb = pd.DataFrame(emb, index=idx, columns=["UMAP1", "UMAP2"])
            df_emb["SampleID"] = df_emb.index
            df_emb[color_title] = color_series_plot

            fig = px.scatter(
                df_emb,
                x="UMAP1", y="UMAP2",
                color=color_title,
                hover_name="SampleID",
                title="UMAP (2D)",
                color_discrete_map=color_discrete_map,
                category_orders=category_orders
            )

        fig = style_clean_axes(fig, width=fig_w, height=fig_h)

        st.plotly_chart(fig, use_container_width=False, config=PLOTLY_CONFIG)
        svg_download_button(fig, "PhenoView_pca_umap.svg")
    except Exception as e:
        st.error("Unable to generate embedding plot. Please check that valid features and samples are selected.")
        st.caption(f"Technical details: {type(e).__name__}: {e}")