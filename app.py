import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Brazil Institutional Fund Simulator", layout="wide")
st.title("🏦 Brazil Institutional Rotation Fund Simulator")

# ==========================================================
# UNIVERSE (B3 + BDR ONLY)
# ==========================================================
UNIVERSE = {
    "B3 COMMODITIES": ["VALE3.SA", "PETR4.SA"],
    "B3 FINANCIALS": ["ITUB4.SA", "BBDC4.SA", "SANB11.SA"],
    "B3 RETAIL": ["MGLU3.SA", "LREN3.SA"],
    "BDR TECH": ["AAPL34.SA", "MSFT34.SA", "NVDC34.SA", "AMZO34.SA"]
}

# ==========================================================
# RISK ENGINE
# ==========================================================
gain_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 8.0, 3.0)
loss_atr = st.sidebar.slider("Stop Loss (ATR)", 0.5, 5.0, 1.5)

# ==========================================================
# DATA
# ==========================================================
@st.cache_data(ttl=3600)
def load(symbol):

    df = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df

# ==========================================================
# INDICATORS
# ==========================================================
def ema(s, p):

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    return s.ewm(span=p, adjust=False).mean()


def rsi(s, p=14):

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = pd.Series(s).astype(float)

    d = s.diff()

    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)

    avg_gain = gain.rolling(p).mean()
    avg_loss = loss.rolling(p).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(df, p=14):

    high = pd.Series(df["High"]).astype(float)
    low = pd.Series(df["Low"]).astype(float)
    close = pd.Series(df["Close"]).astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(p).mean()

# ==========================================================
# CORE ENGINE
# ==========================================================
def analyze(asset, sector):

    df = load(asset)

    if df.empty or len(df) < 200:
        return None

    df["EMA9"] = ema(df["Close"], 9)
    df["EMA29"] = ema(df["Close"], 29)
    df["EMA69"] = ema(df["Close"], 69)
    df["EMA169"] = ema(df["Close"], 169)

    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)

    df = df.dropna()

    if len(df) < 50:
        return None

    price = float(df["Close"].iloc[-1])
    ema169 = float(df["EMA169"].iloc[-1])
    rsi_now = float(df["RSI"].iloc[-1])

    trend_ok = price > ema169

    ribbon = 1 if (
        df["EMA9"].iloc[-1]
        > df["EMA29"].iloc[-1]
        > df["EMA69"].iloc[-1]
        > ema169
    ) else 0

    score = (
        (60 if trend_ok else 0)
        + np.clip((40 - rsi_now) * 1.5, 0, 25)
        + (15 if rsi_now < 45 else 5)
        + (ribbon * 15)
    )

    valid = df.iloc[-200:]

    wins = 0
    samples = 80

    for _ in range(samples):

        idx = np.random.randint(50, len(valid) - 10)

        p = float(valid["Close"].iloc[idx])
        a = float(valid["ATR"].iloc[idx])

        future = valid["Close"].iloc[idx + 1: idx + 20]

        if (future >= p + gain_atr * a).any():
            wins += 1

    prob = wins / samples

    final_score = (score * 0.7) + (prob * 100 * 0.3)

    return {
        "asset": asset,
        "sector": sector,
        "score": score,
        "prob": prob,
        "final_score": final_score
    }

# ==========================================================
# RUN ENGINE
# ==========================================================
results = []

for sector, assets in UNIVERSE.items():
    for a in assets:

        try:
            r = analyze(a, sector)

            if r:
                results.append(r)

        except Exception as e:
            st.warning(f"Erro em {a}: {e}")

if len(results) == 0:
    st.error("Nenhum ativo retornou dados.")
    st.stop()

df = pd.DataFrame(results)

# ==========================================================
# SECTOR VIEW
# ==========================================================
sector_df = df.groupby("sector").agg({
    "final_score": "mean",
    "prob": "mean"
}).reset_index()

sector_df["fund_strength"] = (
    sector_df["final_score"] * 0.7
    + sector_df["prob"] * 100 * 0.3
)

sector_df = sector_df.sort_values(
    "fund_strength",
    ascending=False
)

# ==========================================================
# FUND CONSTRUCTION
# ==========================================================
df["weight"] = df["final_score"] / df["final_score"].sum()

portfolio = df.sort_values(
    "weight",
    ascending=False
)

# ==========================================================
# UI
# ==========================================================
st.subheader("🌍 Brazil Institutional Allocation (Sector View)")
st.dataframe(sector_df)

st.subheader("💼 Simulated Fund Portfolio")
st.dataframe(
    portfolio[
        ["asset", "sector", "final_score", "weight"]
    ]
)

st.subheader("🔥 Top Position")

top = portfolio.iloc[0]

st.success(
    f"{top['asset']} | Weight: {top['weight']:.2%}"
)

# ==========================================================
# GRÁFICO DO MELHOR ATIVO
# ==========================================================
st.subheader("📈 Top Position Chart")

chart_df = load(top["asset"])

if not chart_df.empty:

    chart_df["EMA9"] = ema(chart_df["Close"], 9)
    chart_df["EMA29"] = ema(chart_df["Close"], 29)
    chart_df["EMA69"] = ema(chart_df["Close"], 69)
    chart_df["EMA169"] = ema(chart_df["Close"], 169)

    chart_plot = chart_df.tail(250).copy()

    chart_plot = chart_plot.set_index("Date")

    st.line_chart(
        chart_plot[
            [
                "Close",
                "EMA9",
                "EMA29",
                "EMA69",
                "EMA169"
            ]
        ]
    )

# ==========================================================
# SUMMARY
# ==========================================================
st.subheader("🏦 Fund Summary")

st.write({
    "Leading Sector": sector_df.iloc[0]["sector"],
    "Top Asset": top["asset"],
    "Fund Strength": sector_df.iloc[0]["fund_strength"]
})
