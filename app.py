import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="B3 + BDR Rotation Engine", layout="wide")
st.title("🇧🇷 B3 + BDR Global Rotation Engine v1")

# ==========================================================
# UNIVERSE (ONLY B3 + BDR)
# ==========================================================
UNIVERSE = {
    "B3 COMMODITIES": ["VALE3.SA", "PETR4.SA"],
    "B3 FINANCIALS": ["ITUB4.SA", "BBDC4.SA", "SANB11.SA"],
    "B3 RETAIL/GROWTH": ["MGLU3.SA", "LREN3.SA"],
    "BDR TECH": ["AAPL34.SA", "MSFT34.SA", "NVDC34.SA", "AMZO34.SA"]
}

# ==========================================================
# RISK ENGINE
# ==========================================================
st.sidebar.header("🎯 Risk Engine")
gain_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 8.0, 3.0, 0.5)
loss_atr = st.sidebar.slider("Stop Loss (ATR)", 0.5, 5.0, 1.5, 0.5)

# ==========================================================
# DATA
# ==========================================================
@st.cache_data(ttl=3600)
def load_data(symbol):
    df = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.reset_index()

# ==========================================================
# INDICATORS
# ==========================================================
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    ag = pd.Series(gain).rolling(p).mean()
    al = pd.Series(loss).rolling(p).mean()

    rs = ag / al
    return 100 - (100 / (1 + rs))

def atr(df, p=14):
    h, l, c = df["High"], df["Low"], df["Close"]

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(p).mean()

# ==========================================================
# CORE ENGINE
# ==========================================================
def analyze(asset):

    df = load_data(asset)
    if df is None or df.empty or len(df) < 200:
        return None

    df["EMA9"] = ema(df["Close"], 9)
    df["EMA29"] = ema(df["Close"], 29)
    df["EMA69"] = ema(df["Close"], 69)
    df["EMA169"] = ema(df["Close"], 169)

    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)

    df = df.dropna()

    price = float(df["Close"].iloc[-1])
    ema9 = float(df["EMA9"].iloc[-1])
    ema29 = float(df["EMA29"].iloc[-1])
    ema69 = float(df["EMA69"].iloc[-1])
    ema169 = float(df["EMA169"].iloc[-1])
    rsi_now = float(df["RSI"].iloc[-1])

    trend_ok = price > ema169

    # RIBBON
    ema_max = max(ema9, ema29, ema69, ema169)
    ema_min = min(ema9, ema29, ema69, ema169)
    compression = (ema_max - ema_min) / ema69

    if ema9 > ema29 > ema69 > ema169:
        ribbon = 1
    elif ema9 < ema29 < ema69 < ema169:
        ribbon = -1
    elif compression < 0.08:
        ribbon = 0.5
    else:
        ribbon = 0

    score = (
        (60 if trend_ok else 0) +
        np.clip((40 - rsi_now) * 1.5, 0, 25) +
        (15 if rsi_now < 45 else 5) +
        (ribbon * 15)
    )

    # ATR PROB (simplificado estável)
    valid = df.iloc[-250:]

    wins = 0
    samples = 100

    for _ in range(samples):
        idx = np.random.randint(50, len(valid) - 10)
        entry = valid.iloc[idx]

        p = entry["Close"]
        a = entry["ATR"]

        if np.isnan(a) or a == 0:
            continue

        future = valid.iloc[idx+1:idx+20]["Close"]

        if (future >= p + gain_atr * a).any():
            wins += 1

    prob = wins / samples

    final_score = (score * 0.7) + (prob * 100 * 0.3)

    return {
        "asset": asset,
        "price": price,
        "score": score,
        "prob": prob,
        "final_score": final_score
    }

# ==========================================================
# RUN
# ==========================================================
results = []

for sector, assets in UNIVERSE.items():
    for asset in assets:
        r = analyze(asset)
        if r:
            r["sector"] = sector
            results.append(r)

df = pd.DataFrame(results)

if df.empty:
    st.error("Sem dados.")
    st.stop()

# ==========================================================
# SECTOR ROTATION
# ==========================================================
sector = df.groupby("sector")[["score", "prob", "final_score"]].mean()
sector["rotation_score"] = sector["final_score"] * 0.7 + sector["prob"] * 100 * 0.3
sector = sector.sort_values("rotation_score", ascending=False)

# ==========================================================
# UI
# ==========================================================
st.subheader("🌍 Sector Rotation (B3 + BDR ONLY)")
st.dataframe(sector)

top_sector = sector.index[0]
st.success(f"🔥 LEADING SECTOR: {top_sector}")

st.subheader("🥇 Asset Ranking")
df = df.sort_values("final_score", ascending=False)
st.dataframe(df)

top = df.iloc[0]
st.success(f"🔥 TOP ASSET: {top['asset']}")

# ==========================================================
# SUMMARY
# ==========================================================
st.subheader("Resumo")

st.write({
    "Leading Sector": top_sector,
    "Top Asset": top["asset"],
    "Rotation Score": float(sector.iloc[0]["rotation_score"])
})
