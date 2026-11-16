"""FBCENC weekly receiving capacity estimator: original FEEED2 PDF methodology.

Input: dbo.Agency_Clustering_Features (geography), plus weekly cluster assignments
and KDE samples. For the fixed-training/rolling-inference experiment requested:
train StandardScaler + KMeans on completed training windows up to June 2020,
then calculate each target week's cluster membership and KDE samples using
only the preceding 10 completed weeks.

Capacity algorithm (unchanged from PDF / app_capacity.py):
 agency counts by region/cluster/week -> cluster-week Gaussian KDE -> for each
 outer repetition, simulate M regional totals (one KDE sample per agency), floor
 negative simulated draws at zero, take the desired percentile; report mean,
 standard deviation, minimum and maximum across K percentile estimates.

The month selector merely chooses weeks; no monthly capacity formula is invented.
Read-only against SQL Server. Run: streamlit run FBCENC_Capacity_Streamlit.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyodbc
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_NAME = "FBCENC_DEV"
EXPECTED_INSTANCE = "EC2AMAZ-BQQ4KPL"
STORAGES = ["DRY", "REFRIG", "PRODUCE", "FROZEN"]
STATIC = ["No_programs", "POPULATION", "RISK_SCORE"]
N_CLUSTERS = 4
WINDOW = 10
TRAIN_START = pd.Timestamp("2018-07-01")
TRAIN_END = pd.Timestamp("2020-06-30")
FIRST_INFERENCE_WEEK = pd.Timestamp("2020-07-06")

st.set_page_config(page_title="FBCENC | Weekly Capacity Estimator", layout="wide")
st.title("Weekly Receiving Capacity Estimator")
st.caption("FEEED2 methodology: cluster-week KDEs → nested Monte Carlo → regional percentile-capacity summaries")


def setting(key, default=None):
    # Streamlit Cloud secrets, with local environment fallback.
    try:
        value = st.secrets.get(key)
        if value is not None:
            return str(value)
    except (FileNotFoundError, KeyError):
        pass
    return os.getenv(key, default)


def connection():
    server = setting("FBCENC_DB_SERVER")
    password = setting("FBCENC_DB_PWD")
    username = setting("FBCENC_DB_USER", "fbcenc_datawriter")
    trust = setting("FBCENC_TRUST_SERVER_CERTIFICATE", "yes").lower()
    if not server or not password:
        raise RuntimeError("Configure FBCENC_DB_SERVER and FBCENC_DB_PWD in Streamlit secrets or environment.")
    if trust not in ("yes", "no"):
        raise RuntimeError("FBCENC_TRUST_SERVER_CERTIFICATE must be yes or no")
    available = pyodbc.drivers()
    driver = next((d for d in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server") if d in available), None)
    if driver is None:
        raise RuntimeError(f"Microsoft SQL Server ODBC driver missing. Available: {available}")
    return pyodbc.connect(
        f"DRIVER={{{driver}}};SERVER={server},1433;DATABASE={DB_NAME};"
        f"UID={username};PWD={password};Encrypt=yes;"
        f"TrustServerCertificate={trust};Connection Timeout=30;",
        autocommit=False,
    )


def as_dataframe(cursor):
    return pd.DataFrame.from_records(cursor.fetchall(), columns=[x[0] for x in cursor.description])


def normalize_fips(s):
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.where(out.str.fullmatch(r"\d{1,11}"), pd.NA).str.zfill(11)


@st.cache_data(ttl=3600, show_spinner="Reading FBCENC_DEV ledger and geography...")
def load_data():
    conn = connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DB_NAME(), @@SERVERNAME")
        database, instance = cursor.fetchone()
        if (database, instance) != (DB_NAME, EXPECTED_INSTANCE):
            raise RuntimeError(f"Unexpected SQL destination: {database} / {instance}")
        cursor.execute("""
            SELECT Code, City, Census_Tract_Name, county, FIPS,
                   No_programs, POPULATION, RISK_SCORE
            FROM dbo.Agency_Clustering_Features
        """)
        geo = as_dataframe(cursor)
        cursor.execute("SELECT MAX([Posting Date]) FROM dbo.Item_Ledger WHERE [Entry Type]='Sale'")
        max_date = pd.Timestamp(cursor.fetchone()[0]).normalize()
        cursor.execute("""
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
        ledger = as_dataframe(cursor)
        return geo, ledger, max_date
    finally:
        conn.rollback()
        conn.close()


@st.cache_resource(show_spinner="Fitting fixed clustering model on 2018–June 2020 data...")
def prepare(geo_input, ledger_input, max_date):
    geo = geo_input.copy()
    ledger = ledger_input.copy()
    geo["Code"] = geo["Code"].astype(str).str.strip()
    geo["FIPS"] = normalize_fips(geo["FIPS"])
    for col in STATIC:
        geo[col] = pd.to_numeric(geo[col], errors="coerce")
    if geo["Code"].duplicated().any():
        raise ValueError("Duplicate agency IDs in dbo.Agency_Clustering_Features")
    excluded = geo[STATIC].isna().any(axis=1) | geo["POPULATION"].le(0) | geo["FIPS"].isna()
    excluded_count = int(excluded.sum())
    geo = geo.loc[~excluded].set_index("Code")
    ledger["Code"] = ledger["Code"].astype(str).str.strip()
    ledger["week_start"] = pd.to_datetime(ledger["week_start"])
    ledger["pounds"] = pd.to_numeric(ledger["pounds"], errors="coerce")
    if not np.isfinite(ledger["pounds"]).all():
        raise ValueError("Nonfinite weekly receiving quantities")
    codes = pd.Index(sorted(geo.index.intersection(pd.Index(ledger["Code"].unique()))))
    if len(codes) < N_CLUSTERS:
        raise ValueError("Insufficient agencies matching geographic and transaction records")
    geo = geo.loc[codes]
    # Latest Monday whose Sunday is no later than the last posted date.
    last_complete = max_date - pd.Timedelta(days=(max_date.weekday()+1) % 7 + 6) if max_date.weekday()!=6 else max_date-pd.Timedelta(days=6)
    weeks = pd.date_range("2018-07-02", last_complete, freq="W-MON")
    subset = ledger[ledger["Code"].isin(codes) & ledger["week_start"].isin(weeks)]
    weekly = subset.pivot_table(index=["Code", "week_start"], columns="storage", values="pounds", aggfunc="sum", fill_value=0)
    weekly = weekly.reindex(columns=STORAGES, fill_value=0)
    weekly = weekly.reindex(pd.MultiIndex.from_product([codes, weeks], names=["Code", "week_start"]), fill_value=0)
    raw = weekly.to_numpy(dtype=float).reshape(len(codes), len(weeks), len(STORAGES))
    negative_count = int((raw < 0).sum())
    # This follows the prior FBCENC implementation's handling of signed net ledger values.
    positive = np.maximum(raw, 0.0)
    denominator = geo["POPULATION"].to_numpy(dtype=float) * 0.13
    scaled = positive / denominator[:, None, None]
    completed_train = (weeks + pd.Timedelta(days=6)) <= TRAIN_END
    if completed_train.sum() < WINDOW:
        raise ValueError("Not enough historical training weeks")
    caps = np.quantile(scaled[:, completed_train, :].reshape(-1, len(STORAGES)), 0.99, axis=0)
    scaled = np.minimum(scaled, caps)
    capped_pounds = scaled * denominator[:, None, None]
    static = geo[STATIC].to_numpy(dtype=float)
    rolling = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW, axis=1)
    rolling_means = rolling.mean(axis=-1)
    rolling_stds = rolling.std(axis=-1, ddof=0)
    end_weeks = weeks[WINDOW-1:]
    train_positions = np.flatnonzero((end_weeks + pd.Timedelta(days=6)) <= TRAIN_END)
    blocks = [np.column_stack((static, rolling_means[:, j, :], rolling_stds[:, j, :])) for j in train_positions]
    train_X = np.vstack(blocks)
    scaler = StandardScaler().fit(train_X)
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(scaler.transform(train_X))
    # Preserve previous FBCENC naming: A1 highest receiving, A4 lowest.
    means_for_ranking = np.concatenate([positive[:, j:j+WINDOW, :].mean(axis=1).sum(axis=1) for j in train_positions])
    rank = sorted(range(N_CLUSTERS), key=lambda k: -float(means_for_ranking[kmeans.labels_==k].mean()))
    names = {cluster: f"A{i+1}" for i, cluster in enumerate(rank)}
    return dict(geo=geo, codes=codes, weeks=weeks, static=static, means=rolling_means,
                stds=rolling_stds, capped_pounds=capped_pounds, scaler=scaler,
                kmeans=kmeans, names=names, excluded=excluded_count, negative_count=negative_count,
                train_rows=len(train_X), latest_week=last_complete)


@st.cache_resource(show_spinner=False)
def week_inputs(model, week_stamp):
    """Construct cluster-week membership and KDE from previous 10 completed weeks."""
    week = pd.Timestamp(week_stamp)
    pos = model["weeks"].get_indexer([week])[0]
    if pos < WINDOW or week < FIRST_INFERENCE_WEEK:
        raise ValueError("Selected week does not have the required ten historical weeks")
    j = pos - WINDOW  # rolling feature window ending at week pos-1
    X = np.column_stack((model["static"], model["means"][:, j, :], model["stds"][:, j, :]))
    raw_labels = model["kmeans"].predict(model["scaler"].transform(X))
    labels = np.array([model["names"][int(label)] for label in raw_labels])
    kdes = {}
    sample_sizes = {}
    for cluster in ("A1", "A2", "A3", "A4"):
        # Same agency membership for the week, same trailing 10 weeks as its KDE samples.
        values = model["capped_pounds"][labels==cluster, pos-WINDOW:pos, :].sum(axis=2).ravel()
        sample_sizes[cluster] = len(values)
        if len(values) < 2 or not np.isfinite(values).all() or np.unique(values).size < 2:
            # Do not fabricate a KDE for a degenerate group.
            kdes[cluster] = None
        else:
            try:
                kdes[cluster] = gaussian_kde(values)
            except np.linalg.LinAlgError:
                kdes[cluster] = None
    return labels, kdes, sample_sizes


def simulate_pdf(counts, kdes, inner_runs, outer_runs, percentile, seed):
    """Exact PDF nesting: each outer run estimates a percentile from fresh inner draws."""
    if not 0 < percentile < 100 or inner_runs < 2 or outer_runs < 2:
        raise ValueError("Invalid Monte Carlo settings")
    missing = [c for c,n in counts.items() if n>0 and kdes.get(c) is None]
    if missing:
        raise ValueError(f"Cannot fit KDE for active clusters: {missing}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(outer_runs, dtype=float)
    # Batch outer repetitions to avoid excessive memory use.
    for start in range(0,outer_runs,20):
        n_batch = min(20, outer_runs-start)
        inner_totals = np.zeros((n_batch,inner_runs), dtype=float)
        for c,n_agencies in counts.items():
            if n_agencies <= 0:
                continue
            draws = kdes[c].resample(n_batch*inner_runs*n_agencies,seed=rng)
            draws = draws.reshape(n_batch,inner_runs,n_agencies)
            inner_totals += np.clip(draws,0,None).sum(axis=2)
        estimates[start:start+n_batch] = np.percentile(inner_totals,percentile,axis=1)
    return estimates


def render_histogram(estimates,title):
    mean = float(np.mean(estimates))
    sd = float(np.std(estimates,ddof=1)) if len(estimates)>1 else 0.0
    fig = go.Figure()
    fig.add_histogram(x=estimates, histnorm="probability density", nbinsx=30,
                      name="Outer-loop estimates", opacity=.65)
    if sd > 0:
        x = np.linspace(float(np.min(estimates)),float(np.max(estimates)),300)
        fig.add_scatter(x=x,y=norm.pdf(x,mean,sd), mode="lines", name="Normal reference fit")
    fig.update_layout(title=title,xaxis_title="Estimated weekly capacity (lbs)",
                      yaxis_title="Density",height=420,barmode="overlay")
    return fig


def make_result(model, week, fips, inner, outer, percentile, seed):
    labels, kdes, sample_sizes = week_inputs(model,week)
    regional = model["geo"].loc[model["codes"],"FIPS"].to_numpy()==fips
    counts = {c:int(np.sum(regional & (labels==c))) for c in ("A1","A2","A3","A4")}
    if sum(counts.values()) == 0:
        raise ValueError("No agencies with valid trained features in selected tract")
    result = simulate_pdf(counts,kdes,inner,outer,percentile,seed)
    return result,counts,sample_sizes


try:
    geo_source,ledger_source,last_posted = load_data()
    model = prepare(geo_source,ledger_source,last_posted)
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

valid_weeks = [w for w in model["weeks"] if w>=FIRST_INFERENCE_WEEK]
if not valid_weeks:
    st.error("No complete weeks available after the training cutoff")
    st.stop()
months = sorted({w.strftime("%Y-%m") for w in valid_weeks})
region = model["geo"].copy()
region["FIPS"] = normalize_fips(region["FIPS"])

with st.sidebar:
    st.header("Select capacity estimate")
    selected_month = st.selectbox("Month",months,index=0,format_func=lambda s:pd.Timestamp(s+"-01").strftime("%B %Y"))
    in_month = [w for w in valid_weeks if w.strftime("%Y-%m")==selected_month]
    week = st.selectbox("Week beginning (Monday)",in_month,format_func=lambda w:w.strftime("%b %d, %Y"))
    counties = sorted(region["county"].dropna().astype(str).unique())
    county = st.selectbox("County",counties)
    tracts = region.loc[region["county"].astype(str)==county,["FIPS","Census_Tract_Name"]].drop_duplicates()
    labels_for_tracts = {r.FIPS:f"{r.Census_Tract_Name} · {r.FIPS}" for r in tracts.itertuples(index=False)}
    fips = st.selectbox("Service region / tract",sorted(labels_for_tracts),format_func=lambda x:labels_for_tracts[x])
    st.divider()
    st.subheader("Monte Carlo settings")
    percentile = st.number_input("Inner-loop percentile",min_value=1.0,max_value=99.9,value=95.0,step=1.0)
    inner = st.number_input("Inner simulations (M)",min_value=2,max_value=5000,value=100,step=50)
    outer = st.number_input("Outer repetitions (K)",min_value=2,max_value=10000,value=1000,step=100)
    seed = st.number_input("Random seed",min_value=0,value=42)
    run = st.button("Estimate weekly capacity",type="primary")
    st.caption("Uses 10 completed weeks preceding the selected week. No database writes.")

st.markdown(f"### {pd.Timestamp(selected_month+'-01').strftime('%B %Y')} · {county} · {labels_for_tracts[fips]}")
st.caption(f"Selected week: {week:%B %d, %Y} · Fixed clustering trained through June 2020 · 10-week rolling KDE inputs")

if run:
    try:
        with st.spinner("Running nested Monte Carlo capacity estimation..."):
            values,counts,sample_sizes = make_result(model,week,fips,int(inner),int(outer),float(percentile),int(seed))
        st.session_state["capacity_result"] = {"values":values,"counts":counts,"sizes":sample_sizes,
             "key":(str(week),fips,int(inner),int(outer),float(percentile),int(seed))}
    except Exception as exc:
        st.error(str(exc))

key = (str(week),fips,int(inner),int(outer),float(percentile),int(seed))
saved = st.session_state.get("capacity_result")
if saved and saved["key"]==key:
    values = saved["values"]
    cols = st.columns(4)
    for col,title,value in zip(cols,["Mean capacity", "Standard deviation", "Minimum", "Maximum"],
                               [np.mean(values),np.std(values,ddof=1),np.min(values),np.max(values)]):
        col.metric(title,f"{value:,.0f} lbs/week")
    st.plotly_chart(render_histogram(values,f"Outer-loop {percentile:g}th-percentile estimates"),use_container_width=True)
    st.write("**Agencies by cluster in selected tract**")
    st.dataframe(pd.DataFrame([{"cluster":c,"agency_count":n,"KDE_sample_count":saved["sizes"][c]} for c,n in saved["counts"].items()]),hide_index=True)
    st.download_button("Download outer-loop capacity estimates (CSV)",
       pd.DataFrame({"week_stamp":str(week.date()),"FIPS":fips,"estimate_lbs":values}).to_csv(index=False),
       file_name=f"capacity_{fips}_{week:%Y%m%d}.csv",mime="text/csv")
    st.info("The summary is the distribution of repeated inner-loop percentile estimates, as specified in the FEEED2 PDF—not a prediction-error calculation.")
else:
    st.write("Select a month, week and service region, then click **Estimate weekly capacity**.")

st.divider()
st.subheader("Compare two regions and/or weeks")
st.caption("Runs the identical nested Monte Carlo procedure for each selection, as in the original app_capacity.py.")
a,b=st.columns(2)
with a:
    left_week = st.selectbox("Left week",in_month,key="left_week")
    left_fips = st.selectbox("Left tract",sorted(labels_for_tracts),format_func=lambda x:labels_for_tracts[x],key="left_fips")
with b:
    right_week = st.selectbox("Right week",in_month,index=min(1,len(in_month)-1),key="right_week")
    right_fips = st.selectbox("Right tract",sorted(labels_for_tracts),index=min(1,len(labels_for_tracts)-1),format_func=lambda x:labels_for_tracts[x],key="right_fips")
if st.button("Run comparison"):
    try:
        with st.spinner("Comparing capacity distributions..."):
            left,_,_=make_result(model,left_week,left_fips,int(inner),int(outer),float(percentile),int(seed))
            right,_,_=make_result(model,right_week,right_fips,int(inner),int(outer),float(percentile),int(seed)+1)
        m1,m2,m3=st.columns(3)
        m1.metric("Left mean capacity",f"{left.mean():,.0f} lbs/week")
        m2.metric("Right mean capacity",f"{right.mean():,.0f} lbs/week")
        m3.metric("Difference (right − left)",f"{right.mean()-left.mean():,.0f} lbs/week")
        fig=go.Figure()
        for draws,name in [(left,"Left"),(right,"Right")]:
            fig.add_histogram(x=draws,histnorm="probability density",name=name,opacity=.40,nbinsx=30)
            sd=np.std(draws,ddof=1)
            if sd>0:
                xs=np.linspace(draws.min(),draws.max(),300)
                fig.add_scatter(x=xs,y=norm.pdf(xs,draws.mean(),sd),mode="lines",name=f"{name} normal fit")
        fig.update_layout(barmode="overlay",xaxis_title="Estimated weekly capacity (lbs)",yaxis_title="Density")
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

with st.expander("Data and model diagnostics"):
    st.write({"database":DB_NAME,"instance":EXPECTED_INSTANCE,"ledger_last_posting_date":str(last_posted.date()),
              "usable_last_complete_week":str(model["latest_week"].date()),"training_feature_rows":model["train_rows"],
              "excluded_agencies_missing_features":model["excluded"],"negative_ledger_cells_zero_floored":model["negative_count"],
              "window_weeks":WINDOW})
    st.caption("The fixed-clustering training/inference split is a selected experimental setup; the capacity calculation follows the PDF. The input static features may postdate the historical evaluation period.")
