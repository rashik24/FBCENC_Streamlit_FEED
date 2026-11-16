"""FBCENC weekly receiving-volume forecasting dashboard.

Train a fixed StandardScaler + 4-cluster KMeans through June 2020.
At each post-training week use the previous 10 *completed* weeks to
assign agencies and fit cluster-specific KDEs, then simulate a tract's
weekly receiving quantities. This is a receiving-volume proxy, NOT a
verified measure of physical receiving capacity.

Read-only SQL Server access: FBCENC_DEV / EC2AMAZ-BQQ4KPL.
Run: streamlit run FBCENC_Streamlit.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pyodbc
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_NAME = "FBCENC_DEV"
INSTANCE = "EC2AMAZ-BQQ4KPL"
STORAGES = ["DRY", "REFRIG", "PRODUCE", "FROZEN"]
STATIC = ["No_programs", "POPULATION", "RISK_SCORE"]
WINDOW = 10
TRAIN_START = pd.Timestamp("2018-07-01")
TRAIN_CUTOFF = pd.Timestamp("2020-06-30")
INFERENCE_START = pd.Timestamp("2020-07-06")  # first full July week
N_CLUSTERS = 4

st.set_page_config(page_title="FBCENC | Weekly Receiving Forecast", layout="wide")
st.title("FBCENC · Weekly Receiving Forecast")
st.caption("Fixed clustering model (2018–June 2020) · 10-week rolling KDE · Walk-forward validation · Read-only FBCENC_DEV")


def connection():
    server = os.getenv("FBCENC_DB_SERVER", "").strip()
    pwd = os.getenv("FBCENC_DB_PWD", "")
    if not server or not pwd:
        raise RuntimeError("Set FBCENC_DB_SERVER and FBCENC_DB_PWD before launching Streamlit.")
    trust = os.getenv("FBCENC_TRUST_SERVER_CERTIFICATE", "no").lower()
    if trust not in ("yes", "no"):
        raise RuntimeError("FBCENC_TRUST_SERVER_CERTIFICATE must be 'yes' or 'no'.")
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server},1433;DATABASE={DB_NAME};UID=fbcenc_datawriter;PWD={pwd};"
        f"Encrypt=yes;TrustServerCertificate={trust};Connection Timeout=30;",
        autocommit=False,
    )


def to_frame(cur):
    return pd.DataFrame.from_records(cur.fetchall(), columns=[c[0] for c in cur.description])


@st.cache_data(show_spinner="Loading weekly ledger and agency features from SQL Server…", ttl=3600)
def load_sql_data(server, user):
    """Reads SQL; credentials are taken only from environment, never cached as args."""
    conn = connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DB_NAME(), @@SERVERNAME, ORIGINAL_LOGIN()")
        db, instance, login = cur.fetchone()
        if (db, instance) != (DB_NAME, INSTANCE):
            raise RuntimeError(f"Unsafe connection: {db} / {instance}; expected {DB_NAME} / {INSTANCE}.")
        cur.execute("""
            SELECT Code, City, Census_Tract_Name, county, FIPS,
                   No_programs, POPULATION, RISK_SCORE
            FROM dbo.Agency_Clustering_Features
        """)
        features = to_frame(cur)
        cur.execute("SELECT MAX([Posting Date]) FROM dbo.Item_Ledger WHERE [Entry Type]='Sale'")
        latest_transaction_date = pd.Timestamp(cur.fetchone()[0]).normalize()
        cur.execute("""
            SELECT LEFT(LTRIM(RTRIM([Source No.])),5) AS Code,
                   DATEADD(day,-((DATEDIFF(day,'19000101',CAST([Posting Date] AS date)) % 7 + 7) % 7),
                           CAST([Posting Date] AS date)) AS week_start,
                   [FBC Storage Requirement Code] AS storage,
                   SUM(-[Invoiced Quantity]) AS pounds
            FROM dbo.Item_Ledger
            WHERE [Entry Type]='Sale' AND [Posting Date] IS NOT NULL
              AND [Source No.] IS NOT NULL AND LEN(LTRIM(RTRIM([Source No.])))>=5
              AND [FBC Storage Requirement Code] IN ('DRY','REFRIG','PRODUCE','FROZEN')
              AND [Invoiced Quantity] IS NOT NULL
            GROUP BY LEFT(LTRIM(RTRIM([Source No.])),5),
                     DATEADD(day,-((DATEDIFF(day,'19000101',CAST([Posting Date] AS date)) % 7 + 7) % 7),
                             CAST([Posting Date] AS date)),
                     [FBC Storage Requirement Code]
        """)
        ledger = to_frame(cur)
        return features, ledger, latest_transaction_date, login
    finally:
        conn.rollback()
        conn.close()


def normalize_fips(x):
    x = x.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return x.where(x.str.fullmatch(r"\d{1,11}"), pd.NA).str.zfill(11)


@st.cache_resource(show_spinner="Training fixed K-means on 2018–2020 windows…")
def prepare_model(features, ledger, latest_date):
    features = features.copy()
    ledger = ledger.copy()
    features["Code"] = features["Code"].astype(str).str.strip()
    features["FIPS"] = normalize_fips(features["FIPS"])
    for c in STATIC:
        features[c] = pd.to_numeric(features[c], errors="coerce")
    if features["Code"].duplicated().any():
        raise ValueError("Duplicate Code in Agency_Clustering_Features")
    valid = features[STATIC].notna().all(axis=1) & (features["POPULATION"] > 0)
    excluded = features.loc[~valid, "Code"].tolist()
    features = features.loc[valid & features["FIPS"].notna()].copy()
    if len(features) < 4:
        raise ValueError("Fewer than four agencies with complete features and FIPS")

    ledger["Code"] = ledger["Code"].astype(str).str.strip()
    ledger["week_start"] = pd.to_datetime(ledger["week_start"])
    ledger["pounds"] = pd.to_numeric(ledger["pounds"], errors="coerce")
    if not np.isfinite(ledger["pounds"]).all():
        raise ValueError("Non-finite transaction totals")
    valid_codes = sorted(set(features["Code"]) & set(ledger["Code"]))
    features = features.set_index("Code").loc[valid_codes]
    if len(features) < 4:
        raise ValueError("Insufficient matched agency codes between tables")
    # Exclude the final partial week: Monday + 6 days must be <= last posting date.
    last_complete_monday = latest_date - pd.Timedelta(days=latest_date.weekday()+7)
    # If latest_date is a Sunday, the latest week that ended that Sunday is complete.
    if latest_date.weekday() == 6:
        last_complete_monday = latest_date - pd.Timedelta(days=6)
    first_monday = pd.Timestamp("2018-07-02")
    if last_complete_monday < INFERENCE_START:
        raise ValueError("No complete post-training weeks")
    weeks = pd.date_range(first_monday, last_complete_monday, freq="W-MON")
    codes = pd.Index(valid_codes)
    subset = ledger[ledger["Code"].isin(codes) & ledger["week_start"].isin(weeks)]
    weekly = subset.pivot_table(index=["Code", "week_start"], columns="storage", values="pounds", aggfunc="sum", fill_value=0)
    weekly = weekly.reindex(columns=STORAGES, fill_value=0)
    panel_index = pd.MultiIndex.from_product([codes, weeks], names=["Code", "week_start"])
    weekly = weekly.reindex(panel_index, fill_value=0)
    raw_values = weekly.to_numpy(dtype=float).reshape(len(codes), len(weeks), len(STORAGES))
    # Negative NET sales can occur from returns/corrections. Zero-floor here only,
    # retaining signed values for audit and reporting count.
    n_negative = int((raw_values < 0).sum())
    positive = np.maximum(raw_values, 0)
    denom = features.loc[codes, "POPULATION"].to_numpy(dtype=float) * 0.13
    scaled = positive / denom[:, None, None]
    # Fit caps *only* on fully completed weeks ending by June 30, 2020.
    train_week_mask = (weeks + pd.Timedelta(days=6) <= TRAIN_CUTOFF)
    if train_week_mask.sum() < WINDOW:
        raise ValueError("Not enough complete training weeks")
    caps = np.quantile(scaled[:, train_week_mask, :].reshape(-1, len(STORAGES)), .99, axis=0)
    scaled = np.minimum(scaled, caps)
    # Historical (capped) pounds used to fit rolling KDEs.
    capped_pounds = scaled * denom[:, None, None]
    static = features.loc[codes, STATIC].to_numpy(dtype=float)
    dyn = np.lib.stride_tricks.sliding_window_view(scaled, window_shape=WINDOW, axis=1)
    means = dyn.mean(axis=-1)
    stdevs = dyn.std(axis=-1, ddof=0)
    # Window at position j ends at weeks[j+9]. Train on all completed training windows.
    end_weeks = weeks[WINDOW-1:]
    train_positions = np.flatnonzero(end_weeks + pd.Timedelta(days=6) <= TRAIN_CUTOFF)
    feature_blocks = [np.column_stack([static, means[:, j, :], stdevs[:, j, :]]) for j in train_positions]
    train_x = np.vstack(feature_blocks)
    scaler = StandardScaler().fit(train_x)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(scaler.transform(train_x))
    # Stable names: A1 highest historical receiving, A4 lowest, consistent with existing script.
    train_labels = kmeans.labels_
    mean_pounds = np.concatenate([positive[:, j:j+WINDOW, :].mean(axis=1).sum(axis=1) for j in train_positions])
    rank = sorted(range(N_CLUSTERS), key=lambda label: -float(mean_pounds[train_labels == label].mean()))
    raw_to_label = {raw: f"A{r+1}" for r, raw in enumerate(rank)}
    return {
        "features": features, "codes": codes, "weeks": weeks, "positive": positive,
        "capped_pounds": capped_pounds, "static": static, "means": means,
        "stdevs": stdevs, "scaler": scaler, "kmeans": kmeans,
        "raw_to_label": raw_to_label, "caps": caps, "excluded": excluded,
        "negative_count": n_negative, "last_complete_monday": last_complete_monday,
        "train_windows": len(train_positions), "training_rows": len(train_x),
    }


def week_prediction(model, target, fips, draws, outer_reps, seed):
    weeks = model["weeks"]
    loc = weeks.get_indexer([pd.Timestamp(target)])[0]
    if loc < WINDOW or target < INFERENCE_START:
        raise ValueError("Not enough historical weeks or target before inference period")
    # Use the 10 weeks ending with the PREVIOUS completed Monday's week.
    prev_window_position = loc - WINDOW  # feature window ending at weeks[loc-1]
    x = np.column_stack([model["static"], model["means"][:, prev_window_position, :],
                         model["stdevs"][:, prev_window_position, :]])
    raw_labels = model["kmeans"].predict(model["scaler"].transform(x))
    labels = np.asarray([model["raw_to_label"][x] for x in raw_labels])
    region_mask = (model["features"].loc[model["codes"], "FIPS"].to_numpy() == fips)
    actual = float(model["positive"][region_mask, loc, :].sum())
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws, dtype=float)
    kde_info = {}
    for cluster in ("A1", "A2", "A3", "A4"):
        n = int(np.sum(region_mask & (labels == cluster)))
        if n == 0:
            continue
        # Collect all prior 10-week *weekly totals* from agencies predicted into cluster.
        historical = model["capped_pounds"][labels == cluster, loc-WINDOW:loc, :].sum(axis=2).ravel()
        historical = historical[np.isfinite(historical)]
        if historical.size < 2:
            raise ValueError(f"Not enough KDE samples for {cluster} ({target:%Y-%m-%d})")
        kde_info[cluster] = n
        if np.std(historical) <= 0:
            samples = np.full(draws*n, historical[0], dtype=float)
        else:
            kde = gaussian_kde(historical)
            samples = kde.resample(draws*n, seed=rng).reshape(-1)
        totals += np.maximum(samples.reshape(draws, n), 0).sum(axis=1)
    if not kde_info:
        raise ValueError("No agencies in selected tract that match the training model")
    p95 = float(np.percentile(totals, 95))
    # Original PDF: replicate P95 calculations to quantify sampling-estimator uncertainty.
    # Separate from the predictive distribution used to compute coverage/CRPS.
    p95_reps = np.array([np.percentile(ix, 95) for ix in np.array_split(np.random.default_rng(seed+1).choice(totals, draws*outer_reps, replace=True), outer_reps)]) if outer_reps else np.array([p95])
    # CRPS unbiased empirical estimator of E|X-y| - .5 E|X-X'|;
    # sorted O(n log n) calculation rather than O(n^2) pairwise matrix.
    ordered = np.sort(totals)
    m = len(ordered)
    mean_abs_pair = (2.0 / (m*m)) * np.sum((2*np.arange(1,m+1)-m-1)*ordered)
    crps = float(np.mean(np.abs(totals-actual)) - .5*mean_abs_pair)
    return {
        "week_start": pd.Timestamp(target), "actual_lbs": actual,
        "pred_mean_lbs": float(totals.mean()), "pred_p05_lbs": float(np.percentile(totals, 5)),
        "pred_p50_lbs": float(np.percentile(totals, 50)), "pred_p95_lbs": p95,
        "pred_p95_estimator_sd": float(p95_reps.std(ddof=1)) if len(p95_reps)>1 else np.nan,
        "n_agencies": int(region_mask.sum()), "cluster_counts": ", ".join(f"{k}:{v}" for k,v in kde_info.items()),
        "absolute_error_lbs": abs(actual-float(totals.mean())), "crps_lbs": crps,
        "below_p95": bool(actual<=p95), "inside_90pct_interval": bool(np.percentile(totals,5)<=actual<=p95),
        "draws": totals,
    }


try:
    if not os.getenv("FBCENC_DB_SERVER") or not os.getenv("FBCENC_DB_PWD"):
        st.error("Database credentials are missing. Set FBCENC_DB_SERVER and FBCENC_DB_PWD in the terminal before launching Streamlit.")
        st.stop()
    f, l, last_date, login = load_sql_data(os.getenv("FBCENC_DB_SERVER"), "fbcenc_datawriter")
    model = prepare_model(f, l, last_date)
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

month_range = pd.period_range(INFERENCE_START, model["last_complete_monday"], freq="M")
if len(month_range) == 0:
    st.warning("No complete inference weeks available.")
    st.stop()

with st.sidebar:
    st.header("Forecast settings")
    month = st.selectbox("Select prediction month", list(month_range), format_func=lambda p:p.strftime("%B %Y"), index=len(month_range)-1)
    county_list = sorted(model["features"]["county"].dropna().astype(str).unique().tolist())
    county = st.selectbox("County", county_list)
    county_features = model["features"][model["features"]["county"].astype(str)==county]
    tracts = county_features[["FIPS", "Census_Tract_Name"]].drop_duplicates().sort_values("FIPS")
    tract_choices = [(str(row.FIPS), str(row.Census_Tract_Name)) for row in tracts.itertuples()]
    region = st.selectbox("Census tract", tract_choices, format_func=lambda x: f"{x[1]} · {x[0]}")
    draws = st.number_input("Simulation draws per week", min_value=200, max_value=20000, value=2000, step=200)
    outer = st.number_input("P95 stability repetitions", min_value=0, max_value=500, value=100, step=25)
    run = st.button("Estimate selected month", type="primary", use_container_width=True)

st.info("This estimates **realized weekly receiving volume**, used as a proxy for receiving capacity. It does not establish an agency's physical maximum capacity. The July 2018–June 2020 model is fixed; the KDE changes for each forecast week.")
summary_cols = st.columns(4)
summary_cols[0].metric("Training windows", f"{model['train_windows']:,}")
summary_cols[1].metric("Training agency-weeks", f"{model['training_rows']:,}")
summary_cols[2].metric("Matched agencies", f"{len(model['codes']):,}")
summary_cols[3].metric("Last complete week", model['last_complete_monday'].strftime("%b %d, %Y"))
with st.expander("Data and model notes"):
    st.write(f"Database: {DB_NAME} ({INSTANCE}) · Login: {login}")
    st.write(f"Source sales latest posting date: {last_date.date()}. Final partial week excluded.")
    st.write(f"Excluded agencies with missing/nonpositive features: {len(model['excluded'])}. Negative storage-week quantities floored at zero: {model['negative_count']:,}.")
    st.write("FEMA risk, population, and programs are sourced from the uploaded feature table. Historical values may not reflect the state of those features in 2020; retrospective validation is therefore conditional on that limitation.")
    st.write("99th-percentile caps and StandardScaler/K-means are fit strictly before July 2020. Weekly KDEs use previous 10 completed weeks, capped using the training-only thresholds.")
    st.write("July 2020 begins with the first complete Monday-starting week on July 6. Select a month to include Monday-starting weeks within it.")

if run:
    weeks = [w for w in model["weeks"] if w >= INFERENCE_START and w.to_period("M")==month]
    if not weeks:
        st.warning("No complete Monday-starting weeks are available for this month.")
        st.stop()
    try:
        progress = st.progress(0, text="Estimating weekly receiving distributions…")
        rows = []
        for idx,w in enumerate(weeks):
            rows.append(week_prediction(model, w, region[0], int(draws), int(outer), seed=2020+idx))
            progress.progress((idx+1)/len(weeks), text=f"Completed {idx+1}/{len(weeks)} weeks")
        progress.empty()
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()
    result = pd.DataFrame([{k:v for k,v in row.items() if k!="draws"} for row in rows])
    actual_total = float(result.actual_lbs.sum())
    pred_total = float(result.pred_mean_lbs.sum())
    st.subheader(f"{month.strftime('%B %Y')} · {county} · {region[1]}")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Predicted month total", f"{pred_total:,.0f} lbs")
    k2.metric("Actual month total", f"{actual_total:,.0f} lbs")
    k3.metric("Absolute month error", f"{abs(actual_total-pred_total):,.0f} lbs")
    k4.metric("Weekly P95 coverage", f"{result.below_p95.mean():.0%}")
    display = result[["week_start", "n_agencies", "actual_lbs", "pred_mean_lbs", "pred_p50_lbs", "pred_p95_lbs", "absolute_error_lbs", "crps_lbs", "cluster_counts"]]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.line_chart(result.set_index("week_start")[["actual_lbs", "pred_mean_lbs", "pred_p95_lbs"]])
    st.markdown(f"**Weekly MAE:** {result.absolute_error_lbs.mean():,.0f} lbs  ·  **Weekly RMSE:** {np.sqrt(np.mean((result.pred_mean_lbs-result.actual_lbs)**2)):,.0f} lbs  ·  **Mean CRPS:** {result.crps_lbs.mean():,.0f} lbs")
    # Sum simulated draws per week for an approximate independent-week monthly distribution.
    monthly_draws = np.sum(np.vstack([r["draws"] for r in rows]), axis=0)
    st.caption(f"Monthly predictive 5th–95th percentile (assuming independent week draws): {np.percentile(monthly_draws,5):,.0f}–{np.percentile(monthly_draws,95):,.0f} lbs. Monthly coverage should be evaluated across many months, not one.")
    st.download_button("Download weekly results CSV", data=result.to_csv(index=False).encode("utf-8"), file_name=f"receiving_forecast_{region[0]}_{month}.csv", mime="text/csv")
else:
    st.write("Choose a month, county, and census tract in the sidebar, then select **Estimate selected month**.")
