import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================================
# CONFIGURAÇÃO
# ==========================================================
st.set_page_config(
    page_title="Simulador de Fundo Institucional Brasileiro",
    layout="wide"
)

st.title("🏦 Simulador de Rotação Institucional da Bolsa Brasileira")

# ==========================================================
# UNIVERSO (APENAS B3 E BDRs)
# ==========================================================
UNIVERSE = {
    "Commodities (Mineração e Petróleo)": ["VALE3.SA", "PETR4.SA"],
    "Setor Financeiro": ["ITUB4.SA", "BBDC4.SA", "SANB11.SA"],
    "Varejo": ["MGLU3.SA", "LREN3.SA"],
    "Tecnologia (BDRs)": ["AAPL34.SA", "MSFT34.SA", "NVDC34.SA", "AMZO34.SA"]
}

# ==========================================================
# MOTOR DE RISCO
# ==========================================================
gain_atr = st.sidebar.slider(
    "Alvo de Lucro (ATR)",
    1.0,
    8.0,
    3.0
)

loss_atr = st.sidebar.slider(
    "Stop Loss (ATR)",
    0.5,
    5.0,
    1.5
)

# ==========================================================
# DADOS
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

    # Corrige retornos MultiIndex do yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df

# ==========================================================
# INDICADORES
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
# MOTOR PRINCIPAL
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
        "Ativo": asset,
        "Setor": sector,
        "Pontuação": score,
        "Probabilidade": prob,
        "Pontuação Final": final_score
    }

# ==========================================================
# EXECUÇÃO
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
# VISÃO SETORIAL
# ==========================================================
sector_df = df.groupby("Setor").agg({
    "Pontuação Final": "mean",
    "Probabilidade": "mean"
}).reset_index()

sector_df["Força do Fundo"] = (
    sector_df["Pontuação Final"] * 0.7
    + sector_df["Probabilidade"] * 100 * 0.3
)

sector_df = sector_df.sort_values(
    "Força do Fundo",
    ascending=False
)

# ==========================================================
# CONSTRUÇÃO DA CARTEIRA
# ==========================================================
df["Peso"] = (
    df["Pontuação Final"]
    / df["Pontuação Final"].sum()
)

portfolio = df.sort_values(
    "Peso",
    ascending=False
)

# ==========================================================
# INTERFACE
# ==========================================================
st.subheader("🌍 Alocação Institucional por Setor")

st.dataframe(
    sector_df,
    use_container_width=True
)

st.subheader("💼 Carteira Simulada do Fundo")

st.dataframe(
    portfolio[
        [
            "Ativo",
            "Setor",
            "Pontuação Final",
            "Peso"
        ]
    ],
    use_container_width=True
)

st.subheader("🔥 Maior Posição da Carteira")

top = portfolio.iloc[0]

st.success(
    f"{top['Ativo']} | Peso: {top['Peso']:.2%}"
)

# ==========================================================
# RESUMO
# ==========================================================
st.subheader("🏦 Resumo do Fundo")

st.write({
    "Setor Líder": sector_df.iloc[0]["Setor"],
    "Melhor Ativo": top["Ativo"],
    "Força do Fundo": round(
        sector_df.iloc[0]["Força do Fundo"],
        2
    )
})
