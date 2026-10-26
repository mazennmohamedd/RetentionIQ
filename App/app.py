import re
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
st.set_page_config(page_title="RetainAI", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
 
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "Data" / "processed" / "telco_clean.csv"
 
missing = [p for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error("RetainAI could not find the files it needs:")
    for p in missing:
        st.code(str(p), language=None)
    st.stop()
 
 
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
 
 
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
 
 
model, df = load_model(), load_data()
FEATURES = [c for c in df.columns if c not in {"customerID", "Churn"}]
NUMERIC = {"tenure", "MonthlyCharges", "TotalCharges"}
 
# Palette (all text colors meet WCAG AA on white)
INK, MUTED, PRIMARY = "#14223B", "#52627A", "#2F5BEA"
STAY, LEFT, WARN, NAVY = "#12A594", "#D95762", "#F5A524", "#0F2744"
BINS, LABELS = [-1, 12, 24, 48, 60, 72], ["0–12", "13–24", "25–48", "49–60", "61–72"]
 
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], button, input, textarea {{ font-family:'Inter',sans-serif; }}
.stApp {{ background:#F3F6FB; }}
.block-container {{ max-width:1280px; padding:2rem 2.5rem 4rem; }}
header[data-testid="stHeader"] {{ background:transparent; }}
 
/* readable text everywhere in the main area */
.stMarkdown p, .stMarkdown li, [data-testid="stCaptionContainer"] p {{ color:{INK}; font-size:15px; line-height:1.55; }}
[data-testid="stCaptionContainer"] p {{ color:{MUTED}; font-size:13.5px; }}
[data-testid="stWidgetLabel"] p {{ color:{INK}; font-size:14.5px; font-weight:600; }}
/* Keep controls readable when Streamlit's theme styles their internals. */
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label,
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label p,
[data-testid="stAppViewContainer"] [data-testid="stRadio"] label span {{ color:{INK}; }}
[data-testid="stMultiSelect"] [data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] > span {{ background:{PRIMARY} !important; background-color:{PRIMARY} !important; color:#fff !important; border-color:{PRIMARY} !important; }}
[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{ color:#fff !important; fill:#fff !important; }}
[data-testid="stAlert"] {{ color:{INK} !important; }}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {{ color:{INK} !important; }}
h1,h2,h3,h4 {{ color:{INK}; letter-spacing:-0.02em; }}
 
/* sidebar */
[data-testid="stSidebar"] {{ background:{NAVY}; }}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span {{ color:#DCE6F5 !important; font-size:15px; font-weight:500; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:10px 14px; border-radius:10px; width:100%; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:rgba(255,255,255,.08); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background:{PRIMARY}; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span {{ color:#fff !important; font-weight:700; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {{ display:none; }}
.brand {{ color:#fff; font-size:24px; font-weight:800; letter-spacing:-0.03em; margin:4px 0 2px; }}
.brand-sub {{ color:#9DB2D1; font-size:13px; margin-bottom:22px; }}
.side-foot {{ color:#9DB2D1; font-size:12.5px; }}
 
/* page header */
.page-title {{ color:{INK}; font-size:30px; font-weight:800; letter-spacing:-0.03em; margin:0; }}
.page-sub {{ color:{MUTED}; font-size:15.5px; margin:6px 0 24px; }}
 
/* KPI cards */
.kpi {{ background:#fff; border:1px solid #E1E8F2; border-radius:14px; padding:18px 20px; box-shadow:0 2px 8px rgba(20,34,59,.05); }}
.kpi-l {{ color:{MUTED}; font-size:14px; font-weight:600; }}
.kpi-v {{ color:{INK}; font-size:30px; font-weight:800; letter-spacing:-0.03em; margin:4px 0 2px; }}
.kpi-n {{ color:{MUTED}; font-size:13px; }}
 
/* charts, forms, callouts */
[data-testid="stPlotlyChart"] {{ background:#fff; border:1px solid #E1E8F2; border-radius:14px; padding:8px; box-shadow:0 2px 8px rgba(20,34,59,.05); }}
div[data-testid="stForm"] {{ background:#fff; border:1px solid #E1E8F2; border-radius:14px; padding:22px; }}
.callout {{ background:#EAF0FF; border-left:5px solid {PRIMARY}; color:{INK}; border-radius:10px; padding:14px 18px; font-size:15px; margin:18px 0; }}
.badge {{ display:inline-block; border-radius:999px; padding:8px 16px; font-size:15px; font-weight:700; }}
.low {{ background:#DDF5F1; color:#0B6B5E; }} .medium {{ background:#FFF0D1; color:#855400; }} .high {{ background:#FDE3E4; color:#A12328; }}
.step {{ background:#fff; border:1px solid #E1E8F2; border-left:5px solid {PRIMARY}; border-radius:10px; padding:13px 16px; margin:8px 0; color:{INK}; font-size:15px; }}
.card {{ background:#fff; border:1px solid #E1E8F2; border-radius:14px; padding:18px 20px; color:{INK}; font-size:15px; height:100%; }}
.card b {{ font-size:16px; }}
.stFormSubmitButton > button {{ background:{PRIMARY}; color:#fff; border:0; border-radius:10px; min-height:48px; font-size:16px; font-weight:700; }}
.stFormSubmitButton > button:hover {{ background:#2548BF; color:#fff; }}
.stTabs [data-baseweb="tab"] p {{ font-size:15px; font-weight:600; }}
</style>
""",
    unsafe_allow_html=True,
)
 
 
# ---------------------------- helpers ----------------------------
 
def churn_rate(d):
    return float(d["Churn"].eq("Yes").mean() * 100) if len(d) else 0.0
 
 
def pretty(c):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", str(c)).capitalize()
 
 
def header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)
 
 
def kpi(col, label, value, note="", color=PRIMARY):
    col.markdown(
        f'<div class="kpi" style="border-top:4px solid {color}"><div class="kpi-l">{label}</div>'
        f'<div class="kpi-v">{value}</div><div class="kpi-n">{note}&nbsp;</div></div>',
        unsafe_allow_html=True,
    )
 
 
def style(fig, title, h=340):
    fig.update_layout(
        template="plotly_white", height=h, margin=dict(t=60, b=20, l=15, r=15),
        title=dict(text=f"<b>{title}</b>", x=0.02, font=dict(size=17, color=INK)),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Inter, sans-serif", size=14, color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor="#C9D3E1", tickfont=dict(size=13, color=INK), title_font=dict(color=INK))
    fig.update_yaxes(gridcolor="#E8EDF5", tickfont=dict(size=13, color=INK), title_font=dict(color=INK))
    fig.update_layout(legend=dict(font=dict(color=INK)))
    return fig
 
 
def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
 
 
def tenure_group(d):
    d = d.copy()
    d["Tenure Group"] = pd.cut(d["tenure"], bins=BINS, labels=LABELS)
    return d
 
 
def rate_by(d, col):
    return d.groupby(col, observed=True)["Churn"].apply(lambda s: s.eq("Yes").mean() * 100).reset_index(name="Churn Rate")
 
 
def predict_proba(row):
    probs = model.predict_proba(row)[0]
    classes = list(getattr(model, "classes_", []))
    idx = classes.index("Yes") if "Yes" in classes else classes.index(1) if 1 in classes else 1
    return float(probs[idx])
 
 
# ---------------------------- sidebar ----------------------------
 
PAGES = {"📊  Overview": "Overview", "🎯  Predict churn": "Predict", "🧩  Customer segments": "Segments", "⚙️  Model details": "Model"}
 
with st.sidebar:
    st.markdown('<div class="brand">🧭 RetainAI</div><div class="brand-sub">Customer retention analytics</div>', unsafe_allow_html=True)
    page = PAGES[st.radio("Navigation", list(PAGES), label_visibility="collapsed")]
    st.markdown(f'<br><div class="side-foot">Telco dataset · {len(df):,} customers</div>', unsafe_allow_html=True)
 
# ---------------------------- overview ----------------------------
 
if page == "Overview":
    header("Customer overview", "See how many customers leave, who they are, and where to act first.")
 
    contracts, internets = sorted(df["Contract"].unique()), sorted(df["InternetService"].unique())
    f1, f2 = st.columns(2)
    sel_c = f1.multiselect("Contract type", contracts, default=contracts)
    sel_i = f2.multiselect("Internet service", internets, default=internets)
 
    fdf = df[df["Contract"].isin(sel_c) & df["InternetService"].isin(sel_i)]
    if fdf.empty:
        st.warning("No customers match these filters. Pick at least one option in each filter.")
        st.stop()
 
    lost = fdf[fdf["Churn"] == "Yes"]
    rate, overall = churn_rate(fdf), churn_rate(df)
    st.write("")
    k = st.columns(4)
    kpi(k[0], "Customers", f"{len(fdf):,}", "in current selection")
    kpi(k[1], "Churn rate", f"{rate:.1f}%", f"{rate - overall:+.1f} pts vs. all customers", LEFT)
    kpi(k[2], "Customers lost", f"{len(lost):,}", "have already left", WARN)
    kpi(k[3], "Monthly revenue lost", f"${lost['MonthlyCharges'].sum():,.0f}", "from churned customers", STAY)
 
    by_contract = rate_by(fdf, "Contract").sort_values("Churn Rate", ascending=False)
    top = by_contract.iloc[0]
    st.markdown(
        f'<div class="callout"><b>Where to look first:</b> customers on <b>{top["Contract"]}</b> contracts '
        f'churn the most, at <b>{top["Churn Rate"]:.1f}%</b>.</div>',
        unsafe_allow_html=True,
    )
 
    a, b = st.columns(2)
    with a:
        fig = px.pie(pd.DataFrame({"Status": ["Stayed", "Left"], "Customers": [len(fdf) - len(lost), len(lost)]}),
                     names="Status", values="Customers", hole=0.65, color="Status",
                     color_discrete_map={"Stayed": STAY, "Left": LEFT})
        fig.update_traces(textinfo="percent", textfont_size=15)
        fig.add_annotation(text=f"<b>{rate:.0f}%</b><br>churn", showarrow=False, font=dict(size=20, color=INK))
        show(style(fig, "Stayed vs. left", 340))
    with b:
        fig = px.bar(by_contract, x="Contract", y="Churn Rate", text_auto=".1f", color_discrete_sequence=[PRIMARY])
        fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=14)
        fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)", yaxis_range=[0, max(20, by_contract["Churn Rate"].max() * 1.25)])
        show(style(fig, "Churn rate by contract", 340))
 
    a, b = st.columns(2)
    with a:
        fig = px.line(rate_by(tenure_group(fdf), "Tenure Group"), x="Tenure Group", y="Churn Rate", markers=True)
        fig.update_traces(line=dict(color=LEFT, width=3), marker=dict(size=10))
        fig.update_layout(xaxis_title="Months as a customer", yaxis_title="Churn rate (%)")
        show(style(fig, "Churn rate by tenure", 340))
    with b:
        fig = px.box(fdf.assign(Status=fdf["Churn"].map({"Yes": "Left", "No": "Stayed"})), x="Status", y="MonthlyCharges",
                     color="Status", color_discrete_map={"Stayed": STAY, "Left": LEFT})
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monthly charges ($)")
        show(style(fig, "Monthly charges: stayed vs. left", 340))
 
    st.caption("These charts describe the data. They do not show what causes churn.")
 
# ---------------------------- predict ----------------------------
 
elif page == "Predict":
    header("Predict churn risk", "Enter a customer's details. Fields start with typical values, so change only what differs.")
 
    SECTIONS = {
        "Profile": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"],
        "Services": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
        "Billing": ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
    }
 
    with st.form("customer_form"):
        customer = {}
        for tab, cols in zip(st.tabs(list(SECTIONS)), SECTIONS.values()):
            with tab:
                st.write("")
                grid = st.columns(3)
                for i, c in enumerate([c for c in cols if c in FEATURES]):
                    with grid[i % 3]:
                        if c == "tenure":
                            customer[c] = st.number_input("Tenure (months)", 0, 72, 12, 1, help="How long they've been a customer.")
                        elif c in NUMERIC:
                            lo, hi = float(df[c].min()), float(df[c].max())
                            customer[c] = st.number_input(pretty(c), lo, hi, min(max(float(df[c].median()), lo), hi))
                        else:
                            opts = sorted(df[c].dropna().unique().tolist(), key=str)
                            customer[c] = st.selectbox(pretty(c), opts, index=opts.index(df[c].mode().iloc[0]))
        go_btn = st.form_submit_button("Calculate churn risk", use_container_width=True)
 
    if go_btn:
        prob = predict_proba(pd.DataFrame([[customer[c] for c in FEATURES]], columns=FEATURES))
        label, cls, color = (("High risk", "high", LEFT) if prob >= 0.7 else ("Medium risk", "medium", WARN) if prob >= 0.4 else ("Low risk", "low", STAY))
 
        st.markdown("### Result")
        g, r = st.columns([1.1, 1])
        with g:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number=dict(suffix="%", font=dict(size=46, color=INK)),
                gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=color, thickness=0.3), borderwidth=0,
                           steps=[dict(range=[0, 40], color="#DDF5F1"), dict(range=[40, 70], color="#FFF0D1"), dict(range=[70, 100], color="#FDE3E4")]),
            ))
            show(style(gauge, "Estimated churn probability", 320))
        with r:
            st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
            st.write("")
            k = st.columns(2)
            kpi(k[0], "This customer", f"{prob:.0%}", "estimated", color)
            kpi(k[1], "Dataset average", f"{churn_rate(df):.0f}%", "all customers", PRIMARY)
            sim = df[(df["Contract"] == customer.get("Contract")) & (df["InternetService"] == customer.get("InternetService"))]
            if len(sim):
                st.caption(f"Customers with the same contract and internet service churn at {churn_rate(sim):.1f}% ({len(sim):,} customers).")
            st.caption(f"Model classification at the 0.50 threshold: **{'likely to churn' if prob >= 0.5 else 'likely to stay'}**.")
 
        st.markdown("### Suggested next steps")
        actions = []
        if customer.get("tenure", 0) <= 12:
            actions.append("Offer onboarding help or an early loyalty reward.")
        if customer.get("Contract") == "Month-to-month":
            actions.append("Offer a longer plan with a clear benefit.")
        if customer.get("TechSupport") == "No":
            actions.append("Check whether a tech-support add-on would help.")
        if customer.get("OnlineSecurity") == "No":
            actions.append("Check whether an online-security package fits.")
        if prob < 0.4:
            actions = ["No action needed now. Keep regular contact and re-check if their plan changes."]
        elif not actions:
            actions = ["Review their recent service experience and billing questions."]
        for a in actions:
            st.markdown(f'<div class="step">{a}</div>', unsafe_allow_html=True)
        st.caption("Risk bands (40% and 70%) are illustrative. Predictions are estimates, not guarantees.")
 
# ---------------------------- segments ----------------------------
 
elif page == "Segments":
    header("Customer segments", "Compare churn across groups to find where to focus.")
 
    options = [c for c in ["Contract", "InternetService", "PaymentMethod", "tenure", "TechSupport", "OnlineSecurity", "PaperlessBilling"] if c in df.columns]
    a, b = st.columns(2)
    feature = a.selectbox("Group customers by", options, format_func=lambda c: "Tenure" if c == "tenure" else pretty(c))
    view = b.radio("Show", ["Churn rate", "Customer count"], horizontal=True)
 
    data = tenure_group(df) if feature == "tenure" else df
    col = "Tenure Group" if feature == "tenure" else feature
 
    if view == "Churn rate":
        summary, y = rate_by(data, col).sort_values("Churn Rate", ascending=False), "Churn Rate"
    else:
        summary, y = data.groupby(col, observed=True).size().reset_index(name="Customer Count").sort_values("Customer Count", ascending=False), "Customer Count"
 
    fig = px.bar(summary, x=col, y=y, text_auto=".1f" if view == "Churn rate" else True, color_discrete_sequence=[PRIMARY])
    fig.update_traces(textposition="outside", cliponaxis=False, textfont_size=14)
    fig.update_layout(xaxis_title="", yaxis_title=view)
    show(style(fig, f"{view} by {'tenure' if feature == 'tenure' else pretty(feature).lower()}", 400))
 
    with st.expander("View the numbers"):
        st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Differences between groups are descriptive and do not prove cause.")
 
# ---------------------------- model ----------------------------
 
else:
    header("Model details", "What powers the predictions and how to read them.")
    est = model.steps[-1][1] if hasattr(model, "steps") else model
 
    k = st.columns(3)
    kpi(k[0], "Algorithm", type(est).__name__)
    kpi(k[1], "Customer inputs", str(len(FEATURES)), "fields per prediction", STAY)
    kpi(k[2], "Decision threshold", "0.50", "at or above = likely to churn", WARN)
 
    st.markdown("### How it works")
    s = st.columns(3)
    s[0].markdown('<div class="card"><b>1. Customer details</b><br>Profile, services, contract and billing information.</div>', unsafe_allow_html=True)
    s[1].markdown('<div class="card"><b>2. Data preparation</b><br>The saved pipeline fills gaps, scales numbers and encodes categories.</div>', unsafe_allow_html=True)
    s[2].markdown('<div class="card"><b>3. Churn estimate</b><br>The model returns a probability and a likely-to-stay or likely-to-churn label.</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout">The Low, Medium and High risk bands are separate from the model\'s 0.50 classification threshold.</div>', unsafe_allow_html=True)
 
    try:
        names = [n.split("__", 1)[-1] for n in model[:-1].get_feature_names_out()]
        vals = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_).ravel()
        imp = pd.DataFrame({"Feature": names, "Importance": vals}).nlargest(12, "Importance").sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_layout(xaxis_title="Relative importance", yaxis_title="")
        show(style(fig, "What the model relies on most", 440))
        st.caption("Importance shows what the model uses, not what causes churn.")
    except Exception:
        st.caption("Feature importance is not available for this model pipeline.")
 
    with st.expander("Held-out test evaluation", expanded=True):
        st.caption(
            "These results come from the held-out test set in the evaluation notebook, "
            "using a 0.50 classification threshold. They are estimates of performance "
            "on data similar to this dataset, not a guarantee for future customers."
        )
        metrics = st.columns(5)
        kpi(metrics[0], "ROC-AUC", "0.832", "test set")
        kpi(metrics[1], "PR-AUC", "0.624", "test set", STAY)
        kpi(metrics[2], "Churn precision", "0.51", "of flagged customers churned", WARN)
        kpi(metrics[3], "Churn recall", "0.81", "of churners were identified", LEFT)
        kpi(metrics[4], "Churn F1", "0.62", "precision and recall balance", PRIMARY)
        st.caption(
            "At this threshold, churn recall is higher than precision: the model catches "
            "many churners, while some customers it flags will stay. Choose an outreach "
            "threshold based on retention costs, capacity, and campaign results."
        )
