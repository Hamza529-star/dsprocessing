# =============================================================================
# All-in-One Data Science Tool — Streamlit Application
# =============================================================================
# HOW TO RUN:
#   1. Install dependencies:
#      pip install streamlit pandas plotly openpyxl numpy
#      pip install kaleido  # Optional: for PNG chart export
#
#   2. Launch the app:
#      streamlit run app.py
# =============================================================================

import io
import json
import re
import base64
import warnings
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLab Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-family: 'DM Sans', sans-serif;
        color: #888;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .stat-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-number { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #00d2ff; }
    .stat-label  { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .op-success  { color: #00c9a7; font-size: 0.85rem; }
    .op-error    { color: #ff6b6b; font-size: 0.85rem; }
    div[data-testid="stTabs"] button { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session-state bootstrap ───────────────────────────────────────────────────
DEFAULTS = {
    "df_original": None,   # Never modified after upload
    "df_current": None,    # Working copy
    "history": [],         # List of past df snapshots for undo
    "file_name": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def push_history():
    """Save a snapshot of df_current before a mutation."""
    if st.session_state.df_current is not None:
        snap = st.session_state.df_current.copy()
        st.session_state.history.append(snap)
        if len(st.session_state.history) > 30:          # Cap memory at 30 steps
            st.session_state.history.pop(0)


def undo():
    if st.session_state.history:
        st.session_state.df_current = st.session_state.history.pop()
        st.success("↩️ Undo successful.")
    else:
        st.warning("Nothing to undo.")


def reset_to_original():
    if st.session_state.df_original is not None:
        st.session_state.df_current = st.session_state.df_original.copy()
        st.session_state.history = []
        st.success("🔄 Reset to original data.")


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def load_excel(raw: bytes, name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw))


def df_to_download(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    """Return (bytes, mime, extension) for a given format."""
    if fmt == "csv":
        data = df.to_csv(index=False).encode()
        return data, "text/csv", "csv"
    if fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    if fmt == "json":
        data = df.to_json(orient="records", indent=2).encode()
        return data, "application/json", "json"
    if fmt == "html":
        data = df.to_html(index=False).encode()
        return data, "text/html", "html"
    raise ValueError(f"Unknown format: {fmt}")


def safe_eval_expr(expr: str, df: pd.DataFrame):
    """Very restricted eval — only allow column references + basic math."""
    allowed = set(re.findall(r'\b[A-Za-z_]\w*\b', expr))
    forbidden = allowed - set(df.columns) - {"np", "pd", "abs", "round", "len", "str", "int", "float"}
    bad_tokens = ["import", "exec", "eval", "open", "os", "sys", "__"]
    for bt in bad_tokens:
        if bt in expr:
            raise ValueError(f"Forbidden token: {bt}")
    local_ns = {col: df[col] for col in df.columns}
    local_ns.update({"np": np, "pd": pd})
    return eval(expr, {"__builtins__": {}}, local_ns)  # noqa: S307


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — File upload + quick actions
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="main-title">🔬 DataLab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">All-in-one data science tool</p>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        raw = uploaded.read()
        try:
            if uploaded.name.endswith(".csv"):
                df = load_csv(raw, uploaded.name)
            else:
                df = load_excel(raw, uploaded.name)
            # Only reload if it's a new file
            if uploaded.name != st.session_state.file_name:
                st.session_state.df_original = df.copy()
                st.session_state.df_current = df.copy()
                st.session_state.history = []
                st.session_state.file_name = uploaded.name
            st.success(f"✅ Loaded: {uploaded.name}")
        except Exception as exc:
            st.error(f"Error loading file: {exc}")

    st.divider()

    if st.session_state.df_current is not None:
        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("↩️ Undo", use_container_width=True):
                undo()
        with col_r:
            if st.button("🔄 Reset", use_container_width=True):
                reset_to_original()
    else:
        st.info("Upload a file to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# GUARD: No data loaded
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.df_current is None:
    st.markdown('<h1 class="main-title">🔬 DataLab Pro</h1>', unsafe_allow_html=True)
    st.markdown(
        "#### Upload a **CSV** or **Excel** file using the sidebar to get started.\n\n"
        "Capabilities include: full Pandas cleaning · interactive Plotly charts · "
        "Top-N analysis · multi-format export · advanced statistics."
    )
    st.stop()

df: pd.DataFrame = st.session_state.df_current  # Convenience alias (re-read each run)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Preview & Info",
    "🧹 Data Cleaning",
    "🔍 Filtering",
    "📈 Charts",
    "🏆 Top N",
    "💾 Export",
    "🧮 Statistics",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Preview & Info
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Dataset Overview")

    # Stat cards
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    for col_widget, val, label in [
        (c1, n_rows, "Rows"),
        (c2, n_cols, "Columns"),
        (c3, n_missing, "Missing Values"),
        (c4, n_dupes, "Duplicate Rows"),
    ]:
        col_widget.markdown(
            f'<div class="stat-card"><div class="stat-number">{val:,}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**First 10 rows**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Column Types & Missing Values**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values.astype(str),
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values,
        "Unique": df.nunique().values,
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Data Cleaning & Manipulation")

    # ── 2.1 Drop Columns ──────────────────────────────────────────────────────
    with st.expander("🗑️ Drop Columns"):
        cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols")
        if st.button("Apply – Drop Columns"):
            if cols_to_drop:
                push_history()
                st.session_state.df_current = df.drop(columns=cols_to_drop)
                st.success(f"Dropped: {cols_to_drop}")
                st.rerun()
            else:
                st.warning("No columns selected.")

    # ── 2.2 Drop Rows by Index ────────────────────────────────────────────────
    with st.expander("🗑️ Drop Rows by Index"):
        idx_input = st.text_input("Enter indices (e.g. 0,2,5 or 0:10)", key="drop_idx")
        if st.button("Apply – Drop Rows by Index"):
            try:
                indices = []
                for part in idx_input.split(","):
                    part = part.strip()
                    if ":" in part:
                        s, e = part.split(":")
                        indices += list(range(int(s), int(e)))
                    else:
                        indices.append(int(part))
                push_history()
                st.session_state.df_current = df.drop(index=indices).reset_index(drop=True)
                st.success(f"Dropped {len(indices)} row(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── 2.3 Drop Rows by Condition ────────────────────────────────────────────
    with st.expander("🗑️ Drop Rows by Condition"):
        c1, c2, c3 = st.columns(3)
        cond_col = c1.selectbox("Column", df.columns, key="cond_col")
        cond_op  = c2.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="cond_op")
        cond_val = c3.text_input("Value", key="cond_val")
        if st.button("Apply – Drop by Condition"):
            try:
                push_history()
                tmp = df.copy()
                if cond_op == "contains":
                    mask = tmp[cond_col].astype(str).str.contains(cond_val, na=False)
                else:
                    val = pd.to_numeric(cond_val, errors="ignore")
                    mask = tmp[cond_col].apply(lambda x: eval(f"x {cond_op} val", {"x": x, "val": val}))  # noqa: S307
                st.session_state.df_current = tmp[~mask].reset_index(drop=True)
                st.success(f"Dropped {mask.sum()} row(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── 2.4 Duplicate Handling ────────────────────────────────────────────────
    with st.expander("🔁 Duplicate Handling"):
        dup_subset = st.multiselect("Subset columns (blank = all)", df.columns.tolist(), key="dup_subset")
        subset_arg = dup_subset if dup_subset else None
        dupes = df[df.duplicated(subset=subset_arg, keep=False)]
        st.write(f"Duplicate rows found: **{len(dupes)}**")
        if len(dupes):
            st.dataframe(dupes.head(20), use_container_width=True)
        keep_opt = st.selectbox("Keep", ["first", "last", False], key="dup_keep")
        if st.button("Apply – Drop Duplicates"):
            push_history()
            st.session_state.df_current = df.drop_duplicates(subset=subset_arg, keep=keep_opt).reset_index(drop=True)
            st.success("Duplicates dropped.")
            st.rerun()

    # ── 2.5 Change Data Type ──────────────────────────────────────────────────
    with st.expander("🔄 Change Data Type"):
        c1, c2 = st.columns(2)
        dtype_col = c1.selectbox("Column", df.columns, key="dtype_col")
        dtype_new = c2.selectbox("New Type", ["int", "float", "string", "datetime", "category"], key="dtype_new")
        if st.button("Apply – Change Type"):
            try:
                push_history()
                tmp = st.session_state.df_current.copy()
                if dtype_new == "datetime":
                    tmp[dtype_col] = pd.to_datetime(tmp[dtype_col], errors="coerce")
                elif dtype_new == "string":
                    tmp[dtype_col] = tmp[dtype_col].astype(str)
                elif dtype_new == "category":
                    tmp[dtype_col] = tmp[dtype_col].astype("category")
                elif dtype_new == "int":
                    tmp[dtype_col] = pd.to_numeric(tmp[dtype_col], errors="coerce").astype("Int64")
                elif dtype_new == "float":
                    tmp[dtype_col] = pd.to_numeric(tmp[dtype_col], errors="coerce")
                st.session_state.df_current = tmp
                st.success(f"Column '{dtype_col}' converted to {dtype_new}.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── 2.6 String Operations ─────────────────────────────────────────────────
    with st.expander("🔤 String Operations"):
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            st.info("No text columns found.")
        else:
            c1, c2 = st.columns(2)
            str_col = c1.selectbox("Column", str_cols, key="str_col")
            str_op  = c2.selectbox("Operation", ["uppercase", "lowercase", "strip", "replace"], key="str_op")
            old_val = new_val = ""
            if str_op == "replace":
                old_val = st.text_input("Find", key="str_find")
                new_val = st.text_input("Replace with", key="str_replace")
            if st.button("Apply – String Op"):
                push_history()
                tmp = st.session_state.df_current.copy()
                s = tmp[str_col].astype(str)
                if str_op == "uppercase":   tmp[str_col] = s.str.upper()
                elif str_op == "lowercase": tmp[str_col] = s.str.lower()
                elif str_op == "strip":     tmp[str_col] = s.str.strip()
                elif str_op == "replace":   tmp[str_col] = s.str.replace(old_val, new_val, regex=False)
                st.session_state.df_current = tmp
                st.success(f"String operation '{str_op}' applied to '{str_col}'.")
                st.rerun()

    # ── 2.7 Date Extraction ───────────────────────────────────────────────────
    with st.expander("📅 Date Operations"):
        date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        if not date_cols:
            # Try converting object cols
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if not date_cols:
            st.info("No datetime columns detected. Convert a column's type first.")
        else:
            c1, c2 = st.columns(2)
            date_col = c1.selectbox("Column", date_cols, key="date_col")
            date_part = c2.selectbox("Extract", ["year", "month", "day", "hour", "minute", "weekday"], key="date_part")
            new_col_name = st.text_input("New column name", value=f"{date_col}_{date_part}", key="date_newcol")
            if st.button("Apply – Extract Date Part"):
                try:
                    push_history()
                    tmp = st.session_state.df_current.copy()
                    series = pd.to_datetime(tmp[date_col], errors="coerce")
                    tmp[new_col_name] = getattr(series.dt, date_part)
                    st.session_state.df_current = tmp
                    st.success(f"Column '{new_col_name}' created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── 2.8 Number Formatting ─────────────────────────────────────────────────
    with st.expander("🔢 Number Formatting"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            num_col   = c1.selectbox("Column", num_cols, key="num_col")
            decimals  = c2.number_input("Decimal places", 0, 10, 2, key="num_dec")
            pct_mode  = c3.checkbox("Convert to %", key="num_pct")
            if st.button("Apply – Number Format"):
                push_history()
                tmp = st.session_state.df_current.copy()
                s = tmp[num_col]
                if pct_mode:
                    tmp[num_col] = (s * 100).round(decimals).astype(str) + "%"
                else:
                    tmp[num_col] = s.round(decimals)
                st.session_state.df_current = tmp
                st.success(f"Formatted '{num_col}'.")
                st.rerun()

    # ── 2.9 Rename Columns ────────────────────────────────────────────────────
    with st.expander("✏️ Rename Columns"):
        rename_map = {}
        for col in df.columns:
            new_name = st.text_input(f"  '{col}'  →", value=col, key=f"ren_{col}")
            if new_name != col:
                rename_map[col] = new_name
        if st.button("Apply – Rename") and rename_map:
            push_history()
            st.session_state.df_current = df.rename(columns=rename_map)
            st.success(f"Renamed {len(rename_map)} column(s).")
            st.rerun()

    # ── 2.10 Create New Column ────────────────────────────────────────────────
    with st.expander("➕ Create New Column"):
        new_col_name = st.text_input("New column name", key="new_col_name")
        expr_input   = st.text_area(
            "Expression (use column names, np, basic math)",
            placeholder="e.g.  col_a + col_b * 2   or   np.log(col_a)",
            key="new_col_expr",
        )
        if st.button("Apply – Create Column"):
            if not new_col_name:
                st.error("Provide a column name.")
            elif not expr_input:
                st.error("Provide an expression.")
            else:
                try:
                    result = safe_eval_expr(expr_input, df)
                    push_history()
                    tmp = st.session_state.df_current.copy()
                    tmp[new_col_name] = result
                    st.session_state.df_current = tmp
                    st.success(f"Column '{new_col_name}' created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Expression error: {e}")

    # ── 2.11 Reorder Columns ──────────────────────────────────────────────────
    with st.expander("↕️ Reorder Columns"):
        new_order = st.multiselect(
            "Pick columns in desired order (select all to reorder)",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key="reorder_cols",
        )
        if st.button("Apply – Reorder") and new_order:
            push_history()
            remaining = [c for c in df.columns if c not in new_order]
            st.session_state.df_current = df[new_order + remaining]
            st.success("Columns reordered.")
            st.rerun()

    # ── 2.12 Fill Missing Values ──────────────────────────────────────────────
    with st.expander("🩹 Fill Missing Values"):
        c1, c2 = st.columns(2)
        fill_col    = c1.selectbox("Column", ["-- All --"] + df.columns.tolist(), key="fill_col")
        fill_method = c2.selectbox("Method", ["value", "mean", "median", "mode", "ffill", "bfill"], key="fill_method")
        fill_value  = ""
        if fill_method == "value":
            fill_value = st.text_input("Fill value", key="fill_val")
        if st.button("Apply – Fill Missing"):
            push_history()
            tmp = st.session_state.df_current.copy()
            target_cols = df.columns.tolist() if fill_col == "-- All --" else [fill_col]
            for c in target_cols:
                try:
                    if fill_method == "value":
                        v = pd.to_numeric(fill_value, errors="ignore")
                        tmp[c] = tmp[c].fillna(v)
                    elif fill_method == "mean":
                        tmp[c] = tmp[c].fillna(tmp[c].mean())
                    elif fill_method == "median":
                        tmp[c] = tmp[c].fillna(tmp[c].median())
                    elif fill_method == "mode":
                        tmp[c] = tmp[c].fillna(tmp[c].mode().iloc[0])
                    elif fill_method == "ffill":
                        tmp[c] = tmp[c].ffill()
                    elif fill_method == "bfill":
                        tmp[c] = tmp[c].bfill()
                except Exception:
                    pass
            st.session_state.df_current = tmp
            st.success("Missing values filled.")
            st.rerun()

    # ── 2.13 Drop Missing ─────────────────────────────────────────────────────
    with st.expander("🗑️ Drop Rows with Missing Values"):
        dm_subset = st.multiselect("Columns (blank = any column)", df.columns.tolist(), key="dm_subset")
        if st.button("Apply – Drop Missing"):
            push_history()
            sub = dm_subset if dm_subset else None
            st.session_state.df_current = df.dropna(subset=sub).reset_index(drop=True)
            st.success("Rows with missing values dropped.")
            st.rerun()

    # ── 2.14 Sort ─────────────────────────────────────────────────────────────
    with st.expander("⬆️ Sort"):
        c1, c2 = st.columns(2)
        sort_cols = c1.multiselect("Sort by", df.columns.tolist(), key="sort_cols")
        ascending = c2.selectbox("Order", ["Ascending", "Descending"], key="sort_asc") == "Ascending"
        if st.button("Apply – Sort") and sort_cols:
            push_history()
            st.session_state.df_current = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            st.success("Data sorted.")
            st.rerun()

    # ── 2.15 Rank ─────────────────────────────────────────────────────────────
    with st.expander("🏅 Add Rank Column"):
        num_rank_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_rank_cols:
            st.info("No numeric columns available.")
        else:
            c1, c2, c3 = st.columns(3)
            rank_col    = c1.selectbox("Column to rank", num_rank_cols, key="rank_col")
            rank_method = c2.selectbox("Method", ["average", "min", "max", "first", "dense"], key="rank_method")
            rank_name   = c3.text_input("New column name", value=f"{rank_col}_rank", key="rank_name")
            if st.button("Apply – Rank"):
                push_history()
                tmp = st.session_state.df_current.copy()
                tmp[rank_name] = tmp[rank_col].rank(method=rank_method)
                st.session_state.df_current = tmp
                st.success(f"Rank column '{rank_name}' added.")
                st.rerun()

    # ── 2.16 Custom Apply ─────────────────────────────────────────────────────
    with st.expander("⚗️ Custom Apply Function"):
        c1, c2 = st.columns(2)
        apply_col  = c1.selectbox("Column", df.columns.tolist(), key="apply_col")
        apply_axis = c2.selectbox("Axis", ["element-wise (column)", "row-wise"], key="apply_axis")
        apply_expr = st.text_area(
            "Lambda expression  (use `x` for element / row)",
            placeholder="e.g.  x * 2   or   x['colA'] + x['colB']",
            key="apply_expr",
        )
        new_apply_col = st.text_input("Result column name", value=f"{apply_col}_applied", key="apply_newcol")
        if st.button("Apply – Custom Function"):
            try:
                push_history()
                tmp = st.session_state.df_current.copy()
                safe_globals = {"__builtins__": {}, "np": np, "pd": pd, "abs": abs, "round": round, "len": len}
                fn = eval(f"lambda x: {apply_expr}", safe_globals)  # noqa: S307
                if "row-wise" in apply_axis:
                    tmp[new_apply_col] = tmp.apply(fn, axis=1)
                else:
                    tmp[new_apply_col] = tmp[apply_col].apply(fn)
                st.session_state.df_current = tmp
                st.success(f"Applied to '{new_apply_col}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("**Current dataframe preview (first 10 rows)**")
    st.dataframe(st.session_state.df_current.head(10), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Filtering
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Dynamic Data Filtering")

    if "filters" not in st.session_state:
        st.session_state.filters = []

    with st.form("add_filter_form"):
        fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 2, 1])
        f_col  = fc1.selectbox("Column", df.columns)
        f_op   = fc2.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains", "between"])
        f_val  = fc3.text_input("Value (for between: use val1,val2)")
        f_logic = fc4.selectbox("Logic", ["AND", "OR"])
        submitted = st.form_submit_button("➕ Add Filter")
        if submitted:
            st.session_state.filters.append({
                "col": f_col, "op": f_op, "val": f_val, "logic": f_logic
            })

    if st.button("🗑️ Clear All Filters"):
        st.session_state.filters = []

    # Show current filters
    if st.session_state.filters:
        st.markdown("**Active Filters:**")
        for i, f in enumerate(st.session_state.filters):
            cols = st.columns([4, 1])
            cols[0].markdown(f"`{f['logic']}` &nbsp; `{f['col']}` &nbsp; `{f['op']}` &nbsp; `{f['val']}`")
            if cols[1].button("✕", key=f"del_filter_{i}"):
                st.session_state.filters.pop(i)
                st.rerun()

    # Apply filters
    filtered_df = st.session_state.df_current.copy()
    for i, f in enumerate(st.session_state.filters):
        try:
            col, op, val, logic = f["col"], f["op"], f["val"], f["logic"]
            if op == "contains":
                mask = filtered_df[col].astype(str).str.contains(val, na=False)
            elif op == "between":
                lo, hi = val.split(",")
                mask = filtered_df[col].between(
                    pd.to_numeric(lo.strip(), errors="coerce"),
                    pd.to_numeric(hi.strip(), errors="coerce"),
                )
            else:
                num_val = pd.to_numeric(val, errors="ignore")
                mask = filtered_df[col].apply(lambda x: eval(f"x {op} num_val", {"x": x, "num_val": num_val}))  # noqa: S307
            if i == 0 or logic == "AND":
                filtered_df = filtered_df[mask]
            else:
                filtered_df = pd.concat([filtered_df, st.session_state.df_current[mask]]).drop_duplicates()
        except Exception as e:
            st.warning(f"Filter {i+1} error: {e}")

    st.markdown(f"**Showing {len(filtered_df):,} / {len(st.session_state.df_current):,} rows**")
    st.dataframe(filtered_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Charts
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Interactive Visualizations")

    chart_type = st.selectbox("Chart Type", [
        "Line Chart", "Bar Chart (Vertical)", "Bar Chart (Horizontal)",
        "Scatter Plot", "Histogram", "Box Plot", "Violin Plot",
        "Pie Chart", "Correlation Heatmap", "Area Chart",
    ])

    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols  = df.columns.tolist()

    c1, c2, c3 = st.columns(3)
    chart_title   = st.text_input("Chart title (optional)", key="chart_title")
    x_label       = st.text_input("X-axis label (optional)", key="x_label")
    y_label       = st.text_input("Y-axis label (optional)", key="y_label")

    fig = None
    try:
        if chart_type == "Correlation Heatmap":
            if num_cols:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr, text_auto=True, color_continuous_scale="RdBu_r",
                    title=chart_title or "Correlation Matrix",
                )
            else:
                st.warning("No numeric columns.")

        elif chart_type == "Histogram":
            hist_col = c1.selectbox("Column", num_cols or all_cols, key="hist_col")
            nbins = c2.slider("Bins", 5, 100, 20, key="hist_bins")
            color_col = c3.selectbox("Color (optional)", ["None"] + cat_cols, key="hist_color")
            fig = px.histogram(
                df, x=hist_col,
                nbins=nbins,
                color=None if color_col == "None" else color_col,
                title=chart_title or f"Histogram of {hist_col}",
                labels={"x": x_label or hist_col},
            )

        elif chart_type == "Pie Chart":
            pie_col   = c1.selectbox("Category column", cat_cols or all_cols, key="pie_cat")
            pie_val   = c2.selectbox("Values (optional)", ["count"] + num_cols, key="pie_val")
            if pie_val == "count":
                pie_df = df[pie_col].value_counts().reset_index()
                pie_df.columns = [pie_col, "count"]
                fig = px.pie(pie_df, names=pie_col, values="count", title=chart_title or f"{pie_col} Distribution")
            else:
                fig = px.pie(df, names=pie_col, values=pie_val, title=chart_title or f"{pie_val} by {pie_col}")

        elif chart_type in ("Box Plot", "Violin Plot"):
            y_col     = c1.selectbox("Y column (numeric)", num_cols, key="box_y")
            x_col     = c2.selectbox("X column (category, optional)", ["None"] + cat_cols, key="box_x")
            color_col = c3.selectbox("Color (optional)", ["None"] + cat_cols, key="box_color")
            x_arg     = None if x_col == "None" else x_col
            c_arg     = None if color_col == "None" else color_col
            if chart_type == "Box Plot":
                fig = px.box(df, x=x_arg, y=y_col, color=c_arg, title=chart_title or f"Box – {y_col}")
            else:
                fig = px.violin(df, x=x_arg, y=y_col, color=c_arg, box=True, title=chart_title or f"Violin – {y_col}")

        else:  # Line, Bar, Scatter, Area
            x_col     = c1.selectbox("X column", all_cols, key="chart_x")
            y_cols    = c2.multiselect("Y column(s)", num_cols or all_cols, default=num_cols[:1] if num_cols else [], key="chart_y")
            color_col = c3.selectbox("Color (optional)", ["None"] + cat_cols, key="chart_color")
            c_arg     = None if color_col == "None" else color_col
            if y_cols:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_cols[0], color=c_arg,
                                     title=chart_title, labels={"x": x_label, "y": y_label})
                elif chart_type == "Bar Chart (Vertical)":
                    fig = px.bar(df, x=x_col, y=y_cols, color=c_arg,
                                 title=chart_title, labels={"x": x_label, "y": y_label})
                elif chart_type == "Bar Chart (Horizontal)":
                    fig = px.bar(df, y=x_col, x=y_cols, orientation="h", color=c_arg,
                                 title=chart_title, labels={"x": x_label, "y": y_label})
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_cols, color=c_arg,
                                  title=chart_title, labels={"x": x_label, "y": y_label})
                elif chart_type == "Area Chart":
                    fig = px.area(df, x=x_col, y=y_cols, color=c_arg,
                                  title=chart_title, labels={"x": x_label, "y": y_label})
            else:
                st.info("Select at least one Y column.")

    except Exception as e:
        st.error(f"Chart error: {e}\n{traceback.format_exc()}")

    if fig:
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # PNG / HTML export
        col_png, col_html = st.columns(2)
        try:
            import kaleido  # noqa: F401
            img_bytes = fig.to_image(format="png", width=1400, height=700)
            col_png.download_button("📥 Download PNG", img_bytes, "chart.png", "image/png")
        except Exception:
            col_png.caption("ℹ️ Install kaleido for PNG export.")
        html_str = fig.to_html(include_plotlyjs="cdn")
        col_html.download_button("📥 Download HTML", html_str.encode(), "chart.html", "text/html")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Top N Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Top / Bottom N Analysis")

    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols or not cat_cols:
        st.warning("Need at least one numeric and one categorical column.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        top_num_col = c1.selectbox("Numeric column", num_cols, key="top_num")
        top_cat_col = c2.selectbox("Category column", cat_cols, key="top_cat")
        top_n       = c3.number_input("N", min_value=1, max_value=50, value=10, key="top_n")
        top_dir     = c4.selectbox("Direction", ["Top", "Bottom"], key="top_dir")

        agg_df = df.groupby(top_cat_col)[top_num_col].sum().reset_index()
        agg_df = agg_df.sort_values(top_num_col, ascending=(top_dir == "Bottom")).head(int(top_n))

        fig_top = px.bar(
            agg_df, x=top_cat_col, y=top_num_col,
            title=f"{top_dir} {int(top_n)}: {top_num_col} by {top_cat_col}",
            template="plotly_dark",
            color=top_num_col,
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(agg_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Export
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Export Data")

    export_source = st.radio(
        "Export which dataframe?",
        ["Current working dataframe", "Original uploaded dataframe"],
        horizontal=True,
    )
    export_df = (
        st.session_state.df_current
        if export_source.startswith("Current")
        else st.session_state.df_original
    )

    st.dataframe(export_df.head(10), use_container_width=True)
    st.markdown(f"**{len(export_df):,} rows · {export_df.shape[1]} columns**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    for col_w, fmt, label, mime in [
        (c1, "csv",   "📄 CSV",   "text/csv"),
        (c2, "excel", "📊 Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (c3, "json",  "🗂️ JSON",  "application/json"),
        (c4, "html",  "🌐 HTML",  "text/html"),
    ]:
        try:
            data, mime_type, ext = df_to_download(export_df, fmt)
            col_w.download_button(label, data, f"export.{ext}", mime_type, use_container_width=True)
        except Exception as e:
            col_w.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Statistics
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Data Science Summary")

    # Summary Statistics
    with st.expander("📋 Summary Statistics (.describe())", expanded=True):
        transpose_desc = st.checkbox("Transpose", key="desc_transpose")
        desc = df.describe(include="all")
        if transpose_desc:
            desc = desc.T
        st.dataframe(desc, use_container_width=True)

    # Missing Value Analysis
    with st.expander("🔍 Missing Value Analysis"):
        miss = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
        }).sort_values("Missing Count", ascending=False)
        st.dataframe(miss, use_container_width=True, hide_index=True)
        fig_miss = px.bar(
            miss[miss["Missing Count"] > 0],
            x="Column", y="Missing %",
            title="Missing Value %",
            template="plotly_dark",
            color="Missing %",
            color_continuous_scale="Reds",
        )
        if len(miss[miss["Missing Count"] > 0]):
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values 🎉")

    # Correlation Heatmap
    with st.expander("🌡️ Correlation Matrix"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation",
                template="plotly_dark",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns.")

    # Value Counts
    with st.expander("🔢 Value Counts"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            vc_col = st.selectbox("Column", cat_cols, key="vc_col")
            vc = df[vc_col].value_counts().head(10).reset_index()
            vc.columns = [vc_col, "Count"]
            fig_vc = px.bar(
                vc, x=vc_col, y="Count",
                title=f"Top 10 Values – {vc_col}",
                template="plotly_dark",
                color="Count",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_vc, use_container_width=True)
            st.dataframe(vc, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns.")

    # Column-wise Histograms
    with st.expander("📊 Column-wise Distributions"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            ncols_plot = min(3, len(num_cols))
            nrows_plot = (len(num_cols) + ncols_plot - 1) // ncols_plot
            fig_hist = make_subplots(
                rows=nrows_plot, cols=ncols_plot,
                subplot_titles=num_cols,
            )
            for i, col in enumerate(num_cols):
                row = i // ncols_plot + 1
                col_idx = i % ncols_plot + 1
                fig_hist.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                    row=row, col=col_idx,
                )
            fig_hist.update_layout(
                template="plotly_dark",
                title_text="Numeric Column Distributions",
                height=300 * nrows_plot,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numeric columns.")
