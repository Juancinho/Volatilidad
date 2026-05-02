"""
Superficie de Volatilidad Implícita — Dashboard Educativo
Fórmula cerrada IV basada en arxiv:2604.24480
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import invgauss, norm
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Superficie de Volatilidad Implícita",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* =====================================================================
   NEOMORFISMO ELEGANTE — PALETA OSCURA
   Inspirado en neumorphism.io (sombras dual-tone sobre superficie suave)
   ===================================================================== */
:root {
    --neu-bg:           #1c1f2e;
    --neu-surface:      #1f2336;
    --neu-surface-2:    #232843;
    --neu-pressed:      #181b29;
    --neu-shadow-dark:  rgba(0, 0, 0, 0.55);
    --neu-shadow-light: rgba(70, 78, 110, 0.28);
    --neu-text:         #e6e8f0;
    --neu-text-muted:   #98a0b8;
    --neu-accent:       #7dd3fc;
    --neu-accent-warm:  #f5b454;
    --neu-radius:       18px;
    --neu-radius-sm:    12px;

    --neu-elevated:
        7px 7px 16px var(--neu-shadow-dark),
        -5px -5px 12px var(--neu-shadow-light);
    --neu-elevated-sm:
        4px 4px 10px var(--neu-shadow-dark),
        -3px -3px 8px var(--neu-shadow-light);
    --neu-pressed-shadow:
        inset 5px 5px 12px var(--neu-shadow-dark),
        inset -3px -3px 8px var(--neu-shadow-light);
    --neu-pressed-shadow-sm:
        inset 3px 3px 7px var(--neu-shadow-dark),
        inset -2px -2px 5px var(--neu-shadow-light);
}

html, body, .stApp {
    background: var(--neu-bg) !important;
    color: var(--neu-text);
}
.main .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

/* ── SIDEBAR ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: var(--neu-bg);
    border-right: none;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--neu-text); }

/* ── HEADINGS / LINKS ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5 { color: var(--neu-text); letter-spacing: -0.01em; }
.stMarkdown a { color: var(--neu-accent); text-decoration: none; }
.stMarkdown a:hover { text-decoration: underline; }
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 22px 0;
}

/* ── MÉTRICAS — TARJETAS ELEVADAS ──────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--neu-surface);
    border: none;
    border-radius: var(--neu-radius);
    padding: 18px 20px;
    box-shadow: var(--neu-elevated);
    transition: transform 0.18s ease;
}
[data-testid="metric-container"]:hover { transform: translateY(-1px); }
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: var(--neu-text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--neu-text);
    font-weight: 600;
}

/* ── CAJAS INFORMATIVAS ────────────────────────────────────────────── */
.info-box, .concept-box, .doc-card {
    background: var(--neu-surface);
    border: none;
    border-radius: var(--neu-radius);
    padding: 22px 24px;
    margin: 14px 0;
    color: var(--neu-text);
    font-size: 0.92rem;
    line-height: 1.72;
    box-shadow: var(--neu-elevated);
}
.warn-box {
    background: var(--neu-surface);
    border: none;
    border-left: 3px solid var(--neu-accent-warm);
    border-radius: var(--neu-radius);
    padding: 22px 24px;
    margin: 14px 0;
    color: #ead7b3;
    font-size: 0.92rem;
    line-height: 1.72;
    box-shadow: var(--neu-elevated);
}
.doc-card h4 { color: var(--neu-text); margin: 0 0 10px 0; font-weight: 600; }
.muted { color: var(--neu-text-muted); font-size: 0.92rem; }

/* ── BANNER DESTACADO (paper) ──────────────────────────────────────── */
.paper-banner {
    background: linear-gradient(135deg, var(--neu-surface), var(--neu-surface-2));
    border: none;
    border-radius: var(--neu-radius);
    padding: 24px 28px;
    margin: 16px 0 24px 0;
    color: var(--neu-text);
    box-shadow:
        9px 9px 20px var(--neu-shadow-dark),
        -6px -6px 14px var(--neu-shadow-light);
}
.paper-banner strong { color: #f8fafc; }
.paper-banner code {
    background: var(--neu-pressed);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.86em;
    color: var(--neu-accent);
    box-shadow: var(--neu-pressed-shadow-sm);
}

/* ── HERO HEADER ───────────────────────────────────────────────────── */
.neu-hero {
    background: var(--neu-surface);
    border-radius: var(--neu-radius);
    padding: 28px 32px;
    margin: 6px 0 18px 0;
    text-align: center;
    box-shadow: var(--neu-elevated);
}
.neu-hero h1 {
    color: var(--neu-text);
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.02em;
}
.neu-hero p { color: var(--neu-text-muted); margin: 6px 0 0 0; }
.neu-hero a { color: var(--neu-accent); }

/* ── TABS — PISTA HUNDIDA + CHIPS ELEVADOS ─────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--neu-bg);
    gap: 10px;
    padding: 10px;
    border-radius: var(--neu-radius);
    box-shadow: var(--neu-pressed-shadow);
}
.stTabs [data-baseweb="tab"] {
    background: var(--neu-surface);
    border-radius: var(--neu-radius-sm);
    padding: 10px 18px !important;
    color: var(--neu-text-muted);
    border: none !important;
    box-shadow: var(--neu-elevated-sm);
    transition: all 0.18s ease;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--neu-text); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--neu-accent);
    box-shadow: var(--neu-pressed-shadow-sm);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── EXPANDERS ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--neu-surface);
    border: none !important;
    border-radius: var(--neu-radius);
    box-shadow: var(--neu-elevated);
    margin: 14px 0;
    overflow: hidden;
}
[data-testid="stExpander"] details summary { padding: 14px 22px; }
[data-testid="stExpander"] details summary p {
    color: var(--neu-text);
    font-weight: 500;
}
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

/* ── BOTONES ───────────────────────────────────────────────────────── */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {
    background: var(--neu-surface);
    color: var(--neu-text);
    border: none;
    border-radius: var(--neu-radius-sm);
    padding: 9px 20px;
    box-shadow: var(--neu-elevated-sm);
    transition: all 0.15s ease;
    font-weight: 500;
}
.stButton button:hover, .stDownloadButton button:hover,
[data-testid="stPopover"] button:hover { color: var(--neu-accent); }
.stButton button:active, .stDownloadButton button:active {
    box-shadow: var(--neu-pressed-shadow-sm);
}

/* ── INPUTS HUNDIDOS ───────────────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"],
.stTextInput input,
.stNumberInput input,
.stNumberInput [data-baseweb="input"],
[data-testid="stTextInput"] input {
    background: var(--neu-pressed) !important;
    border: none !important;
    border-radius: var(--neu-radius-sm) !important;
    box-shadow: var(--neu-pressed-shadow-sm) !important;
    color: var(--neu-text) !important;
}
[data-baseweb="select"] > div { min-height: 40px; padding: 2px 8px; }
[data-baseweb="popover"] [role="listbox"] {
    background: var(--neu-surface) !important;
    border-radius: var(--neu-radius-sm) !important;
    box-shadow: var(--neu-elevated) !important;
    border: none !important;
}

/* ── SLIDERS ───────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] > div:first-child {
    background: var(--neu-pressed);
    box-shadow: var(--neu-pressed-shadow-sm);
    border-radius: 12px;
    height: 8px;
}
.stSlider [role="slider"] {
    background: var(--neu-surface) !important;
    border: 1px solid rgba(125, 211, 252, 0.45) !important;
    box-shadow: var(--neu-elevated-sm) !important;
}
.stSlider [data-testid="stTickBar"] { color: var(--neu-text-muted); }

/* ── DATAFRAMES ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: var(--neu-radius);
    overflow: hidden;
    box-shadow: var(--neu-elevated);
    border: none !important;
}

/* ── PLOTLY CARDS ──────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--neu-surface);
    border-radius: var(--neu-radius);
    padding: 8px;
    box-shadow: var(--neu-elevated);
    margin: 8px 0;
}

/* ── ALERTAS ───────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--neu-radius);
    border: none !important;
    box-shadow: var(--neu-elevated);
}

/* ── CHECKBOX / RADIO ──────────────────────────────────────────────── */
.stCheckbox label, .stRadio label { color: var(--neu-text); }

/* ── SCROLLBAR ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--neu-bg); }
::-webkit-scrollbar-thumb {
    background: var(--neu-surface);
    border-radius: 6px;
    box-shadow: var(--neu-elevated-sm);
}
::-webkit-scrollbar-thumb:hover { background: var(--neu-surface-2); }

/* ── SPINNER ───────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: var(--neu-accent) !important; }

/* ── CÓDIGO INLINE ─────────────────────────────────────────────────── */
.stMarkdown code {
    background: var(--neu-pressed);
    color: var(--neu-accent);
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 0.88em;
    box-shadow: var(--neu-pressed-shadow-sm);
}
.stMarkdown pre { box-shadow: var(--neu-pressed-shadow-sm); border-radius: var(--neu-radius-sm); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FÓRMULA CERRADA IV  (arxiv:2604.24480)
# ─────────────────────────────────────────────────────────────────────────────

def iv_closed_form(C: float, K: float, F: float, D: float, T: float) -> float:
    if T <= 0 or C <= 0 or K <= 0 or F <= 0 or D <= 0:
        return np.nan
    c = C / (D * F)
    k = np.log(K / F)
    if abs(k) < 1e-7:
        arg = np.clip((c + 1.0) / 2.0, 1e-10, 1 - 1e-10)
        v = 2.0 * norm.ppf(arg)
        return (v / np.sqrt(T)) if v > 0 else np.nan
    m = 1.0 if k > 0 else np.exp(k)
    prob = np.clip((1.0 - c) / m, 1e-10, 1 - 1e-10)
    try:
        x = invgauss.ppf(prob, mu=2.0 / abs(k))
        s = (2.0 / np.sqrt(x)) / np.sqrt(T)
        return s if np.isfinite(s) and s > 0 else np.nan
    except Exception:
        return np.nan


def iv_from_put(P: float, K: float, F: float, D: float, T: float) -> float:
    C = P + D * (F - K)
    return iv_closed_form(C, K, F, D, T) if C > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_data(ticker: str, risk_free_rate: float, max_T: float):
    tkr = yf.Ticker(ticker)
    spot = tkr.fast_info.get('lastPrice', None)
    if not spot:
        hist = tkr.history(period='5d')
        spot = float(hist['Close'].iloc[-1]) if not hist.empty else None
    if not spot:
        raise ValueError(f"No se pudo obtener precio para {ticker}")

    today = date.today()
    rows = []

    for exp_str in tkr.options:
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        except ValueError:
            continue
        T = (exp_date - today).days / 365.0
        if T < 1 / 365 or T > max_T:
            continue
        D = np.exp(-risk_free_rate * T)
        F = spot / D

        try:
            puts = tkr.option_chain(exp_str).puts.copy()
        except Exception:
            continue
        if puts.empty:
            continue

        puts.columns = [c.lower().replace(' ', '_') for c in puts.columns]

        for _, row in puts.iterrows():
            K = float(row.get('strike', 0) or 0)
            if K <= 0:
                continue
            bid  = float(row.get('bid',  0) or 0)
            ask  = float(row.get('ask',  0) or 0)
            last = float(row.get('lastprice', 0) or 0)
            P = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            if P <= 0:
                continue

            def safe_int(v):
                return 0 if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)

            if safe_int(row.get('volume', 0)) == 0 and safe_int(row.get('openinterest', 0)) == 0:
                continue

            log_m = np.log(K / F)
            if log_m > 0.02:
                continue

            iv = iv_from_put(P, K, F, D, T)
            if not np.isfinite(iv) or not (0.01 <= iv <= 3.0):
                continue

            sq = iv * np.sqrt(T)
            d1 = np.log(F / K) / sq + sq / 2
            delta = norm.cdf(d1) - 1.0
            gamma = norm.pdf(d1) / (F * sq + 1e-12)
            vega  = F * D * norm.pdf(d1) * np.sqrt(T)

            rows.append({
                'expiry':        exp_str,
                'T':             round(T, 4),
                'K':             K,
                'F':             round(F, 2),
                'D':             round(D, 6),
                'log_moneyness': round(log_m, 5),
                'iv':            round(iv, 5),
                'iv_pct':        round(iv * 100, 2),
                'delta':         round(delta, 4),
                'gamma':         round(gamma, 6),
                'vega':          round(vega, 4),
                'bid':           bid,
                'ask':           ask,
                'mid_price':     round(P, 3),
                'volume':        safe_int(row.get('volume', 0)),
                'open_interest': safe_int(row.get('openinterest', 0)),
            })

    if not rows:
        raise ValueError(f"Sin datos válidos de opciones para {ticker}.")
    return pd.DataFrame(rows), float(spot)


def compute_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    def interp_at(xs, ys, x0):
        if len(xs) < 1:
            return np.nan
        idx = np.argsort(xs)
        xs, ys = xs[idx], ys[idx]
        if x0 <= xs[0]:  return float(ys[0])
        if x0 >= xs[-1]: return float(ys[-1])
        i = int(np.searchsorted(xs, x0)) - 1
        t = (x0 - xs[i]) / (xs[i+1] - xs[i] + 1e-12)
        return float(ys[i] * (1-t) + ys[i+1] * t)

    records = []
    for exp, g in df.groupby('expiry'):
        g = g.sort_values('K')
        T    = g['T'].iloc[0]
        atm  = interp_at(g['log_moneyness'].values, g['iv'].values, 0.0)
        d25  = interp_at(g['delta'].values, g['iv'].values, -0.25)
        d10  = interp_at(g['delta'].values, g['iv'].values, -0.10)
        if np.isfinite(atm) and np.isfinite(d25):
            records.append({
                'expiry':  exp,
                'T':       T,
                'atm_iv':  atm,
                'iv_25d':  d25,
                'iv_10d':  d10 if np.isfinite(d10) else np.nan,
                'skew':    d25 - atm,
                'skew_10': (d10 - atm) if np.isfinite(d10) else np.nan,
            })
    return pd.DataFrame(records).sort_values('T').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

PANEL_BG  = '#1f2336'
TMPL      = 'plotly_dark'


def fig_3d_surface(df: pd.DataFrame) -> go.Figure:
    """Scatter de IV observada + superficie interpolada linealmente."""
    sd = df[(df['log_moneyness'] >= -0.55) & (df['T'] <= 3.5)].copy()
    xs = sd['log_moneyness'].values * 100
    ys = sd['T'].values
    zs = sd['iv_pct'].values
    vmin, vmax = np.percentile(zs, 2), np.percentile(zs, 98)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='markers',
        marker=dict(size=2.5, color=zs, colorscale='plasma',
                    cmin=vmin, cmax=vmax, opacity=0.85),
        name='Datos reales',
        hovertemplate='Moneyness: %{x:.1f}%<br>Vencimiento: %{y:.2f}a<br>IV: %{z:.1f}%<extra></extra>',
    ))
    try:
        xi = np.linspace(xs.min(), xs.max(), 60)
        yi = np.linspace(ys.min(), ys.max(), 50)
        XI, YI = np.meshgrid(xi, yi)
        # Interpolacion lineal sobre puntos irregulares: no calibra un modelo SVI/SABR.
        ZI = griddata((xs, ys), zs, (XI, YI), method='linear')
        fig.add_trace(go.Surface(
            x=XI, y=YI, z=ZI, colorscale='plasma',
            cmin=vmin, cmax=vmax, opacity=0.40,
            showscale=True,
            colorbar=dict(title='IV (%)', x=1.01, thickness=14,
                          tickfont=dict(color='#aaaacc')),
            hovertemplate='Moneyness: %{x:.1f}%<br>Venc: %{y:.2f}a<br>IV interpolada: %{z:.1f}%<extra></extra>',
        ))
    except Exception:
        pass

    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, showlegend=False,
        scene=dict(
            bgcolor=PANEL_BG,
            xaxis=dict(title='Log-Moneyness ln(K/F) (%)', color='#aaaacc'),
            yaxis=dict(title='Tiempo al vencimiento (años)', color='#aaaacc'),
            zaxis=dict(title='Vol Implícita (%)', color='#aaaacc'),
            camera=dict(eye=dict(x=-1.5, y=-1.8, z=0.85)),
        ),
        margin=dict(l=0, r=0, t=30, b=0), height=580,
    )
    return fig


def fig_term_structure(term_df: pd.DataFrame) -> go.Figure:
    labels = [e[2:] for e in term_df['expiry']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=term_df['atm_iv'] * 100, mode='lines+markers',
        name='ATM IV', line=dict(color='#4488ff', width=2), marker=dict(size=5),
        hovertemplate='%{x}<br>ATM IV: %{y:.1f}%<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=term_df['iv_25d'] * 100, mode='lines+markers',
        name='25Δ Put IV', line=dict(color='#ff4444', width=2, dash='dash'),
        marker=dict(size=5),
        hovertemplate='%{x}<br>25Δ IV: %{y:.1f}%<extra></extra>',
    ))
    if term_df['iv_10d'].notna().any():
        fig.add_trace(go.Scatter(
            x=labels, y=term_df['iv_10d'] * 100, mode='lines+markers',
            name='10Δ Put IV', line=dict(color='#ff8800', width=1.5, dash='dot'),
            marker=dict(size=4),
            hovertemplate='%{x}<br>10Δ IV: %{y:.1f}%<extra></extra>',
        ))
    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='ATM IV — Estructura Temporal',
        xaxis=dict(title='Vencimiento', tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(title='IV (%)', ticksuffix='%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=320, margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def fig_skew_index(term_df: pd.DataFrame) -> go.Figure:
    labels   = [e[2:] for e in term_df['expiry']]
    skew_pt  = term_df['skew'] * 100
    colors   = ['#cc3333' if s > 5 else '#22aa44' for s in skew_pt]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=skew_pt, marker_color=colors,
        hovertemplate='%{x}<br>Skew: %{y:.1f} pt<extra></extra>',
    ))
    fig.add_hline(y=5, line_dash='dot', line_color='#ffaa00',
                  annotation_text='Umbral 5pt', annotation_font_color='#ffaa00')
    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='Índice de Skew  (25Δ IV − ATM IV)',
        xaxis=dict(title='Vencimiento', tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(title='Skew (puntos vol)', ticksuffix=' pt'),
        showlegend=False,
        height=320, margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def fig_iv_smile_moneyness(df: pd.DataFrame, selected: list) -> go.Figure:
    COLORS = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, exp in enumerate(selected):
        g = df[df['expiry'] == exp].sort_values('log_moneyness')
        if g.empty:
            continue
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=g['log_moneyness'] * 100, y=g['iv_pct'],
            mode='markers+lines', name=f"{exp} (T={g['T'].iloc[0]:.2f}a)",
            line=dict(color=c, width=1.5), marker=dict(size=5),
            hovertemplate='Moneyness: %{x:.1f}%<br>IV: %{y:.1f}%<extra></extra>',
        ))
    fig.add_vline(x=0, line_dash='dot', line_color='#888899',
                  annotation_text='ATM', annotation_font_color='#888899')
    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='Sonrisa de Volatilidad — espacio log-moneyness',
        xaxis=dict(title='Log-Moneyness ln(K/F) (%)', ticksuffix='%',
                   zeroline=True, zerolinecolor='#333355'),
        yaxis=dict(title='IV (%)', ticksuffix='%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=8)),
        height=380, margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def fig_iv_smile_delta(df: pd.DataFrame, selected: list) -> go.Figure:
    COLORS = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, exp in enumerate(selected):
        g = df[df['expiry'] == exp].sort_values('delta')
        if g.empty:
            continue
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=g['delta'] * 100, y=g['iv_pct'],
            mode='markers+lines', name=exp,
            line=dict(color=c, width=1.5), marker=dict(size=5),
            hovertemplate='Delta: %{x:.0f}%<br>IV: %{y:.1f}%<extra></extra>',
        ))
    fig.add_vline(x=-50, line_dash='dot', line_color='#888899',
                  annotation_text='ATM (Δ≈−50%)', annotation_font_color='#888899')
    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='Sonrisa de Volatilidad — espacio delta',
        xaxis=dict(title='Delta del put (%)', ticksuffix='%'),
        yaxis=dict(title='IV (%)', ticksuffix='%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=8)),
        height=380, margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def fig_iv_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap 2D de la superficie de volatilidad interpolada."""
    sd = df[(df['log_moneyness'] >= -0.55) & (df['T'] <= 3.5)].copy()
    try:
        xi = np.linspace(sd['log_moneyness'].min() * 100,
                         sd['log_moneyness'].max() * 100, 80)
        yi = np.linspace(sd['T'].min(), sd['T'].max(), 60)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata(
            (sd['log_moneyness'].values * 100, sd['T'].values),
            sd['iv_pct'].values,
            (XI, YI), method='linear'
        )
    except Exception:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        x=xi, y=yi, z=ZI,
        colorscale='plasma', zmin=np.nanpercentile(ZI, 2), zmax=np.nanpercentile(ZI, 98),
        colorbar=dict(title='IV (%)', tickfont=dict(color='#aaaacc')),
        hovertemplate='Moneyness: %{x:.1f}%<br>T: %{y:.2f}a<br>IV: %{z:.1f}%<extra></extra>',
    ))
    fig.add_vline(x=0, line_dash='dot', line_color='white',
                  annotation_text='ATM', annotation_font_color='white')
    fig.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='Mapa de Calor — Superficie de Volatilidad (vista superior)',
        xaxis=dict(title='Log-Moneyness ln(K/F) (%)', ticksuffix='%'),
        yaxis=dict(title='Tiempo al vencimiento (años)'),
        height=400, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Configuración")
    ticker_presets = {
        "SPY — S&P 500 ETF": "SPY",
        "QQQ — Nasdaq 100 ETF": "QQQ",
        "IWM — Russell 2000 ETF": "IWM",
        "GLD — Oro ETF": "GLD",
        "TLT — Bonos largo plazo": "TLT",
        "FXE — Euro/USD ETF": "FXE",
        "FXY — Yen japonés ETF": "FXY",
        "FXB — Libra esterlina ETF": "FXB",
        "FXA — Dólar australiano ETF": "FXA",
        "UUP — Dólar USA Index ETF": "UUP",
        "Personalizado": "",
    }
    ticker_choice = st.selectbox(
        "Activo rápido",
        options=list(ticker_presets.keys()),
        help=(
            "Para FX, Yahoo Finance no suele ofrecer opciones OTC sobre el par spot. "
            "Estos tickers son ETFs de divisas o del dólar que sí pueden tener cadena de opciones."
        ),
    )
    if ticker_choice == "Personalizado":
        ticker = st.text_input(
            "Ticker manual (Yahoo Finance)",
            value="SPY",
            help="Ejemplos: SPY, QQQ, IWM, AAPL, FXE, FXY, FXB, FXA, UUP."
        ).upper().strip()
    else:
        ticker = ticker_presets[ticker_choice]
        st.caption(f"Ticker usado: `{ticker}`")
    risk_free_rate = st.slider(
        "Tasa libre de riesgo (%)", 0.0, 8.0, 4.3, 0.1, format="%.1f%%"
    ) / 100
    max_T = st.slider("Máximo vencimiento (años)", 0.5, 5.0, 3.0, 0.5)

    st.markdown("---")
    st.markdown("""
**Tickers recomendados:**
- `SPY` — S&P 500 ETF (más líquido)
- `QQQ` — Nasdaq 100 ETF
- `IWM` — Russell 2000 ETF
- `GLD` — Oro ETF
- `TLT` — Bonos largo plazo

**Proxies FX con opciones listadas:**
- `FXE` — Euro CurrencyShares ETF
- `FXY` — Japanese Yen ETF
- `FXB` — British Pound ETF
- `FXA` — Australian Dollar ETF
- `UUP` — US Dollar Index Bullish Fund

> En Yahoo/yfinance normalmente no hay cadenas gratuitas de opciones OTC sobre pares
> spot tipo `EURUSD=X`. Para FX usamos ETFs de divisas con opciones listadas.

> **Nota:** NK225 no está disponible gratuitamente. Requiere Bloomberg/IBKR. Usa SPY como equivalente de demostración.

---
**Fuente:** Yahoo Finance via yfinance
**Fórmula IV:** arxiv:2604.24480
**Caché:** 5 minutos
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="neu-hero">
  <h1>Superficie de Volatilidad Implícita</h1>
  <p>
    Dashboard educativo · Fórmula cerrada sin iteración basada en
    <a href="https://arxiv.org/abs/2604.24480">arxiv:2604.24480</a>
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner(f"Descargando opciones de **{ticker}** desde Yahoo Finance..."):
    try:
        df, spot = fetch_options_data(ticker, risk_free_rate, max_T)
        term_df  = compute_term_structure(df)
        load_err = None
    except Exception as e:
        df = term_df = None
        spot = 0.0
        load_err = str(e)

if load_err:
    st.error(load_err)
    st.info("Prueba con un ticker diferente (SPY, QQQ, IWM) o revisa tu conexión.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS CLAVE
# ─────────────────────────────────────────────────────────────────────────────

atm_short = term_df['atm_iv'].iloc[0]  * 100 if len(term_df) > 0 else 0
atm_long  = term_df['atm_iv'].iloc[-1] * 100 if len(term_df) > 0 else 0
avg_skew  = term_df['skew'].mean()     * 100 if len(term_df) > 0 else 0
n_exp     = len(term_df)
iv_min    = df['iv_pct'].min()
iv_max    = df['iv_pct'].max()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Spot", f"{spot:,.1f}")
    with st.popover("Definición"):
        st.markdown(
            "Último precio disponible del activo subyacente descargado desde Yahoo Finance. "
            "Se usa para construir el forward de cada vencimiento."
        )
        st.latex(r"F=S e^{rT}")
with c2:
    st.metric("Vencimientos", n_exp)
    with st.popover("Definición"):
        st.markdown(
            "Número de fechas de expiración que quedan después de filtrar vencimientos "
            "sin datos válidos o fuera del horizonte seleccionado."
        )
with c3:
    st.metric("ATM IV corto", f"{atm_short:.1f}%")
    with st.popover("Definición"):
        st.markdown(
            "Volatilidad implícita interpolada en `k = 0` para el primer vencimiento "
            "disponible. Resume el nivel de volatilidad de corto plazo."
        )
        st.latex(r"k=\ln(K/F)=0")
with c4:
    st.metric("ATM IV largo", f"{atm_long:.1f}%")
    with st.popover("Definición"):
        st.markdown(
            "Volatilidad implícita interpolada en `k = 0` para el último vencimiento "
            "incluido. Permite comparar el nivel de largo plazo con el corto plazo."
        )
with c5:
    st.metric("Skew promedio", f"{avg_skew:+.1f} pt")
    with st.popover("Definición"):
        st.markdown(
            "Promedio, entre vencimientos, de la diferencia entre la IV del put 25-delta "
            "y la IV ATM. Se expresa en puntos de volatilidad."
        )
        st.latex(r"Skew=\sigma_{25\Delta}-\sigma_{ATM}")
with c6:
    st.metric("Rango IV", f"{iv_min:.0f}%–{iv_max:.0f}%")
    with st.popover("Definición"):
        st.markdown(
            "Mínimo y máximo de volatilidad implícita observados en los contratos que "
            "superan los filtros de liquidez, vencimiento y moneyness."
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Banner: fórmula en uso ────────────────────────────────────────────────────
st.markdown("""
<div class="paper-banner">
  <div style="font-size:1.02rem; font-weight:700;">
    Todos los valores de IV se calculan con la fórmula cerrada de
    <a href="https://arxiv.org/abs/2604.24480">arxiv:2604.24480</a>
  </div>
  <div class="muted" style="margin-top:6px;">
    Cada punto en cada gráfico proviene de <code>iv_from_put(P, K, F, D, T)</code>,
    continúa con <code>iv_closed_form(C, K, F, D, T)</code> y evalúa
    <code>invgauss.ppf(prob, mu=2/|k|)</code>.
    No hay búsqueda iterativa de raíces: primero se calcula una IV por contrato y después
    se visualiza la nube de IVs como superficie.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Panel principal",
    "Metodología",
    "Interpretación",
    "Sonrisa",
    "Cadena",
    "Guía completa",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUPERFICIE 3D + TERM STRUCTURE + SKEW + HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── Superficie 3D ────────────────────────────────────────────────────────
    st.markdown("### Superficie de volatilidad implícita")

    with st.expander("Construcción de la superficie", expanded=True):
        st.markdown("""
<div class="concept-box">

La figura combina dos objetos distintos: observaciones de mercado y una interpolación
visual. Conviene separarlos porque no tienen el mismo estatuto.

1. **Puntos observados.** Para cada put con precio y liquidez suficientes, se toma el
precio medio `mid = (bid + ask)/2`, se convierte el put en una call sintética mediante
paridad put-call y se calcula la volatilidad implícita con la fórmula explícita de
arxiv:2604.24480. Cada contrato genera un punto:

`(log-moneyness, tiempo al vencimiento, volatilidad implicita)`.

2. **Superficie interpolada.** Las opciones reales no cotizan en una rejilla completa
de strikes y vencimientos. Para representar una lámina continua se usa
`scipy.interpolate.griddata` con `method="linear"`. Esta interpolación estima valores
intermedios entre contratos cercanos; no calibra un modelo paramétrico ni impone
restricciones de no arbitraje.

Por tanto, la información primaria son los puntos. La malla semitransparente sirve para
leer la geometría global de la superficie: pendiente por strike, estructura temporal y
zonas de mayor o menor prima de volatilidad.

</div>
        """, unsafe_allow_html=True)

    with st.expander("Lectura del gráfico", expanded=False):
        st.markdown("""
El gráfico representa la función empírica:

$$
\\sigma_{imp}=\\sigma_{imp}(\\ln(K/F),T)
$$

donde cada punto corresponde a una opción observada en mercado.

---

#### Ejes

**Eje X: log-moneyness**

El eje horizontal muestra:

$$
k=\\ln(K/F)
$$

El punto `0%` corresponde a `K = F`, es decir, al strike at-the-forward. Los valores
negativos corresponden a strikes por debajo del forward:

| Log-Moneyness | Significado concreto (aprox.) |
|--------------|-------------------------------|
| 0% | Strike igual al forward |
| -10% | Strike aproximadamente 9.5% por debajo del forward |
| -20% | Strike aproximadamente 18% por debajo del forward |
| -40% | Strike aproximadamente 33% por debajo del forward |

En puts, la zona negativa representa principalmente opciones fuera del dinero, usadas
como protección frente a caídas.

**Eje Y: tiempo al vencimiento**

Mide el plazo restante hasta el vencimiento, expresado en años.

**Eje Z: volatilidad implícita**

La altura de cada punto es la volatilidad implícita anualizada que reproduce el precio
observado de la opción bajo Black-Scholes.

---

#### Forma esperada

En el modelo Black-Scholes con volatilidad constante, la superficie sería plana:

$$
\\sigma_{imp}(k,T)=\\sigma_0
$$

En datos reales de índices de renta variable suele observarse una pendiente por strike:
los puts OTM, situados a la izquierda, tienen mayor IV que las opciones próximas al ATM.
Esto refleja la prima que el mercado asigna a la protección contra caídas severas.

En activos de tipo FX puede aparecer una sonrisa más simétrica, porque el mercado puede
pagar prima por movimientos extremos en ambas direcciones.

---

#### Color

El color codifica la misma variable que la altura: la IV. Los tonos oscuros indican
menor volatilidad implícita; los tonos claros indican mayor volatilidad implícita.

El gráfico es interactivo: se puede rotar, ampliar y restablecer la vista.

---

#### Observación frente a interpolación

Cada punto es una opción real filtrada por liquidez, vencimiento y moneyness. La malla
transparente no es un segundo modelo financiero: es una interpolación lineal. Donde hay
pocos contratos cercanos, la lectura de la malla debe hacerse con más cautela.
        """)

    st.plotly_chart(fig_3d_surface(df), use_container_width=True)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown("### Mapa de calor de la superficie")

    with st.expander("Lectura del mapa de calor"):
        st.markdown("""
<div class="info-box">

El mapa de calor muestra la misma superficie desde una vista superior. El eje horizontal
es el log-moneyness, el eje vertical es el vencimiento y el color representa la IV.

La ventaja de esta vista es que permite comparar zonas de la superficie sin depender
del ángulo de cámara del gráfico 3D.

| Color | IV | Interpretación |
|-------|----|----------------|
| Tonos claros | Alta | Mayor prima de volatilidad |
| Tonos intermedios | Media | Zona de transición |
| Tonos oscuros | Baja | Menor prima de volatilidad |

La línea vertical blanca marca `k = 0`, es decir, `K = F`. A la izquierda están los
strikes por debajo del forward; en puts, esa región corresponde a protección OTM.

Patrones relevantes:

- Una zona clara en el extremo izquierdo indica mayor IV en puts OTM profundos.
- Un degradado suave de izquierda a derecha indica skew estable.
- Bandas horizontales sugieren cambios por vencimiento, posiblemente asociados a eventos
  concentrados en determinadas fechas.
- IV elevada en vencimientos cortos puede indicar riesgo inmediato o estrés de mercado.

#### Qué significa "interpolado" aquí

El mapa se calcula sobre una rejilla regular de 80 × 60 puntos. Para cada celda, Python
busca los contratos reales cercanos en el plano `(log-moneyness, T)` y estima la IV por
interpolación lineal. Por eso el mapa es útil para ver patrones, pero la tabla de la
cadena de opciones sigue siendo la fuente exacta contrato por contrato.

</div>
        """, unsafe_allow_html=True)

    st.plotly_chart(fig_iv_heatmap(df), use_container_width=True)

    # ── Term Structure + Skew ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Estructura temporal de IV")
        with st.expander("Lectura de la estructura temporal"):
            st.markdown("""
<div class="info-box">

Este gráfico resume cómo cambia la volatilidad implícita al variar el vencimiento.
Cada punto del eje X es una fecha de expiración disponible en la cadena de opciones; el
eje Y muestra la IV anualizada.

---

#### Curvas representadas

**ATM IV:** volatilidad implícita interpolada en `k = 0`, es decir, en el strike
at-the-forward. Es la referencia central de volatilidad para cada vencimiento.

**25-delta put IV:** volatilidad implícita de un put OTM representativo. Se usa como
referencia habitual para medir el coste de protección de cartera.

**10-delta put IV:** volatilidad implícita de un put más alejado del dinero. Captura
mejor la prima asociada a escenarios de cola.

---

#### Pendiente de la curva

**Contango:** la IV aumenta con el vencimiento. Suele aparecer cuando el mercado no está
asignando una prima excepcional al riesgo inmediato.

**Backwardation:** la IV corta es superior a la IV larga. Suele indicar tensión de corto
plazo, eventos próximos o demanda intensa de cobertura inmediata.

**Curva plana:** el mercado no diferencia de forma marcada entre riesgo corto y largo.

---

Si la curva 25-delta está por encima de la ATM, existe una prima de skew: el mercado
paga más volatilidad por protección OTM que por exposición cercana al dinero.

</div>
            """, unsafe_allow_html=True)
        st.plotly_chart(fig_term_structure(term_df), use_container_width=True)

    with col_r:
        st.markdown("### Índice de skew  (25Δ − ATM IV)")
        with st.expander("Lectura del índice de skew"):
            st.markdown("""
El skew mide la diferencia entre la volatilidad de un put OTM representativo y la
volatilidad ATM del mismo vencimiento.

---

#### Definición

$$
Skew(T)=\\sigma_{25\\Delta}(T)-\\sigma_{ATM}(T)
$$

Se expresa en puntos de volatilidad. Si la IV del put 25-delta es 28% y la ATM es 20%,
el skew es +8 puntos.

---

#### Interpretación

**Skew positivo y elevado:** los puts OTM incorporan una prima significativa frente al
ATM. Es habitual en índices de renta variable, especialmente en entornos defensivos.

**Skew bajo:** la diferencia entre protección OTM y volatilidad ATM es reducida.

**Skew negativo:** situación inusual en renta variable; puede indicar datos defectuosos,
iliquidez o una dislocación concreta.

---

#### Patrones a observar

- Skew alto en vencimientos cortos y bajo en largos: riesgo específico de corto plazo.
- Skew alto en toda la curva: demanda estructural de protección.
- Skew decreciente por vencimiento: la prima de cola se concentra en el corto plazo.
- Skew errático: revisar liquidez, spreads y open interest.

---

#### Referencia histórica aproximada

| Evento | Skew típico (S&P) |
|--------|-------------------|
| Mercado alcista tranquilo (2017) | 3–5 pt |
| Mercado normal | 5–8 pt |
| Corrección −10% (2018 Q4) | 10–15 pt |
| Crisis COVID (mar 2020) | 20–30 pt |
| GFC (oct 2008) | 25–40 pt |
            """)
        st.plotly_chart(fig_skew_index(term_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METODOLOGÍA
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    st.markdown("## Metodología: de precios de mercado a superficie")
    st.markdown(
        "Esta pestaña separa el cálculo financiero de la visualización. El paper se usa "
        "para obtener una IV por contrato; la superficie continua aparece después como "
        "una interpolación gráfica de esos puntos."
    )

    st.markdown("### 1. Datos observados")
    st.markdown("""
Para cada vencimiento descargado desde Yahoo Finance se toma cada put con precio válido.
Cuando hay bid y ask positivos, el precio usado es el mid:
    """)
    st.latex(r"P_{mid}=\frac{bid+ask}{2}")
    st.markdown("""
Si no hay mid válido, se usa el último precio disponible. Después se eliminan contratos
sin volumen ni open interest para reducir ruido de iliquidez.
    """)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Forward y coordenadas")
        st.latex(r"D=e^{-rT}")
        st.latex(r"F=\frac{S}{D}=S e^{rT}")
        st.latex(r"k=\ln\left(\frac{K}{F}\right)")
        st.markdown("""
`k` es el log-moneyness. Es la coordenada horizontal de la superficie. El punto ATM
forward está en `k = 0`; los puts OTM suelen estar en `k < 0`.
        """)
    with c_right:
        st.markdown("#### Put a call sintética")
        st.latex(r"C=P+D(F-K)")
        st.latex(r"c=\frac{C}{DF}")
        st.markdown("""
La fórmula del paper se aplica a calls. Por eso convertimos cada put en una call
sintética con paridad put-call antes de calcular la IV.
        """)

    st.markdown("### Fórmula explícita usada")
    st.markdown("""
Para \(k \\neq 0\), el paper conecta Black-Scholes con la CDF de la Gaussiana Inversa:
    """)
    st.latex(r"""
\frac{1-c}{m}
=
\mathcal{F}_{IG}
\left(
    \frac{4}{v^2};
    \frac{2}{|k|},
    1
\right),
\qquad
v=\sigma\sqrt{T},
\qquad
m=\min(1,e^k)
    """)
    st.markdown("Al invertir la CDF:")
    st.latex(r"""
\sigma
=
\frac{2}{\sqrt{T}}
\left[
\mathcal{F}_{IG}^{-1}
\left(
    \frac{1-c}{m};
    \frac{2}{|k|},
    1
\right)
\right]^{-1/2}
    """)

    st.markdown("### Pipeline implementado en el código")
    st.code("""
P = (bid + ask) / 2
D = exp(-r * T)
F = spot / D
k = log(K / F)

C = P + D * (F - K)      # paridad put-call
c = C / (D * F)          # precio normalizado
m = min(1, exp(k))
p = (1 - c) / m

x_star = invgauss.ppf(p, mu=2 / abs(k))
sigma = 2 / (sqrt(x_star) * sqrt(T))

surface_point = (100 * k, T, 100 * sigma)
    """, language="python")

    st.markdown("### De nube de puntos a superficie")
    st.markdown("""
El paper termina en `sigma`: una IV por contrato. Para dibujar una lámina continua,
la app crea una rejilla regular en `(log-moneyness, T)` y usa interpolación lineal:
    """)
    st.code("""
xi = np.linspace(xs.min(), xs.max(), 60)
yi = np.linspace(ys.min(), ys.max(), 50)
XI, YI = np.meshgrid(xi, yi)
ZI = griddata((xs, ys), zs, (XI, YI), method="linear")
    """, language="python")
    st.markdown("""
`griddata` triangula la nube irregular de contratos y estima la IV dentro de cada
triángulo como un plano local:
    """)
    st.latex(r"z(x,y)\approx a+bx+cy")
    st.markdown("""
Por eso la lectura correcta es: **los puntos son datos calculados contrato a contrato;
la malla y el heatmap son una ayuda visual interpolada**. No es una calibración SVI/SABR
ni una superficie libre de arbitraje.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INTERPRETACIÓN
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    st.markdown("## Interpretación: log-moneyness, skew y lectura de la superficie")

    st.markdown("### Por qué hay valores negativos en el eje X")
    st.markdown("""
La app usa \(k=\\ln(K/F)\). Si el strike está por debajo del forward, entonces \(K/F<1\)
y el logaritmo es negativo. Para un put, eso significa que la opción está fuera del dinero:
    """)
    st.latex(r"K<F \Rightarrow \ln(K/F)<0 \Rightarrow \text{put OTM}")

    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        st.markdown("#### Ejemplo numérico")
        st.latex(r"F=600,\qquad K=540")
        st.latex(r"\frac{K}{F}=0.90")
        st.latex(r"k=\ln(0.90)\approx -0.105")
        st.markdown(
            "En el gráfico aparece como `-10.5%`. Significa que el strike está "
            "aproximadamente un 10% por debajo del forward."
        )
    with col_b:
        st.markdown("#### Tabla de lectura")
        st.dataframe(pd.DataFrame([
            {"K/F": "1.00", "k": "0.0%", "Lectura": "ATM"},
            {"K/F": "0.95", "k": "-5.1%", "Lectura": "Put ligeramente OTM"},
            {"K/F": "0.90", "k": "-10.5%", "Lectura": "Put OTM"},
            {"K/F": "0.80", "k": "-22.3%", "Lectura": "Put muy OTM"},
            {"K/F": "0.60", "k": "-51.1%", "Lectura": "Put de crash extremo"},
            {"K/F": "1.05", "k": "+4.9%", "Lectura": "Put ITM"},
        ]), use_container_width=True, hide_index=True)

    st.markdown("### Cómo debería verse el skew")
    st.markdown("""
En índices de renta variable, lo habitual es que la IV suba al movernos hacia valores
más negativos de log-moneyness:
    """)
    st.latex(r"""
\sigma_{\mathrm{imp}}(k=-50\%)
>
\sigma_{\mathrm{imp}}(k=-10\%)
>
\sigma_{\mathrm{imp}}(k=0)
    """)
    st.markdown("""
La razón es que los puts profundos OTM son seguros contra crashes. La demanda estructural
de protección sube su precio y, al invertir Black-Scholes, eso se traduce en una IV mayor.
    """)
    st.dataframe(pd.DataFrame([
        {"Zona": "k ≈ 0", "Contrato": "ATM", "Lectura": "IV base"},
        {"Zona": "k ≈ -10%", "Contrato": "Put OTM moderado", "Lectura": "Más IV que ATM"},
        {"Zona": "k ≈ -25%", "Contrato": "Protección de caída fuerte", "Lectura": "IV claramente superior"},
        {"Zona": "k ≈ -50%", "Contrato": "Protección de crash extremo", "Lectura": "IV alta, datos menos fiables"},
    ]), use_container_width=True, hide_index=True)

    st.markdown("""
<div class="warn-box">
<strong>Cuidado con los extremos.</strong> En strikes muy alejados, como \(k=-50\\%\),
puede haber poco volumen, bid/ask amplio o precios antiguos. Si la IV extrema parece
incoherente, conviene revisar la cadena de opciones antes de interpretar la superficie.
Para lectura robusta, suele ser mejor comparar ATM, 25-delta y 10-delta.
</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SONRISA DE VOLATILIDAD
# ══════════════════════════════════════════════════════════════════════════════

with tab4:

    st.markdown("### Sonrisa y smirk de volatilidad")

    with st.expander("Lectura de la sonrisa de volatilidad", expanded=True):
        st.markdown("""
La sonrisa de volatilidad es un corte de la superficie para un vencimiento fijo:

$$
\\sigma_{imp}(k,T_0)
$$

Muestra cómo cambia la IV al variar el strike, manteniendo constante el plazo.

---

Si Black-Scholes con volatilidad constante describiera completamente el mercado, este
corte sería horizontal. En mercados reales, la IV cambia con el strike porque los precios
de las opciones incorporan colas, asimetría y demanda de cobertura.

---

#### Formas habituales

**Smile simétrico:** frecuente en algunos mercados de divisas. La IV aumenta en ambos
extremos porque el mercado paga por movimientos grandes tanto al alza como a la baja.

```
IV alta    |  *               *
           | * *           * *
           |    * * * * * *
IV baja    |________________
           |     OTM  ATM  OTM
           |     put      call
```

**Smirk o skew de renta variable:** habitual en índices. La IV es mayor en puts OTM,
porque la protección contra caídas tiene demanda estructural.

```
IV alta    | *
           | * *
           |    * *
IV baja    |       * * * * *
           |___________________
           | OTM  ATM  OTM
           | put      call
```

---

#### Coordenadas

El gráfico por log-moneyness permite comparar strikes en términos relativos al forward.
El gráfico por delta permite leer la curva en una escala más habitual de trading:
25-delta y 10-delta son referencias estándar para protección OTM y cola.
        """)

    all_exp  = sorted(df['expiry'].unique().tolist())
    default  = all_exp[:min(4, len(all_exp))]
    selected = st.multiselect(
        "Selecciona vencimientos (puedes elegir varios para comparar):",
        options=all_exp, default=default,
        help="Cada línea representa la curva de IV para ese vencimiento"
    )

    if not selected:
        st.info("Selecciona al menos un vencimiento arriba para ver la sonrisa.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Por Log-Moneyness")
            st.markdown("""
<p style="color:#9090b0; font-size:0.88rem;">
<b>Eje X:</b> ln(K/F) × 100%. El 0% es ATM. Un put a −20% requiere que el activo caiga
~18% para terminar dentro del dinero. Cuanto más negativo es el eje, más alejado está el
strike del forward y más extrema es la protección que representa.
</p>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_iv_smile_moneyness(df, selected), use_container_width=True)

        with col_b:
            st.markdown("#### Por Delta del Put")
            st.markdown("""
<p style="color:#9090b0; font-size:0.88rem;">
<b>Eje X:</b> el delta del put, que va de 0% (muy OTM, improbable) a −100% (muy ITM, casi seguro).
Delta −50% ≈ ATM. Delta −25% = put estándar de cobertura institucional. Delta −10% = seguro
catastrófico. Usar el espacio delta facilita comparar opciones entre activos distintos.
</p>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_iv_smile_delta(df, selected), use_container_width=True)

        st.markdown("#### Resumen de IV clave por vencimiento seleccionado")
        summary_rows = []
        for exp in selected:
            row = term_df[term_df['expiry'] == exp]
            if row.empty:
                continue
            r = row.iloc[0]
            summary_rows.append({
                'Vencimiento': exp,
                'T (años)':    f"{r['T']:.2f}",
                'ATM IV':      f"{r['atm_iv']*100:.1f}%",
                '25Δ IV':      f"{r['iv_25d']*100:.1f}%",
                '10Δ IV':      f"{r['iv_10d']*100:.1f}%" if np.isfinite(r['iv_10d']) else '—',
                'Skew (25Δ)':  f"{r['skew']*100:+.1f} pt",
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CADENA DE OPCIONES
# ══════════════════════════════════════════════════════════════════════════════

with tab5:

    st.markdown("### Cadena de opciones")

    with st.expander("Lectura de la cadena de opciones", expanded=True):
        st.markdown("""
<div class="info-box">

La **cadena de opciones** (options chain) es la tabla maestra de todos los contratos de
opciones disponibles para un activo. Es lo que ves en la pantalla de cualquier broker
cuando quieres operar opciones. Cada fila es un contrato diferente con un strike y
vencimiento específico.

---

#### ¿Qué significa cada columna?

**Strike (K):** El precio pactado en el contrato. Si compras un put con Strike=$530,
tienes el derecho a vender el activo a $530, independientemente de a cuánto cotice.
Los strikes por debajo del precio actual (spot) son puts OTM — requieren que el mercado
caiga para ser rentables.

**Bid y Ask:** El mercado de opciones funciona como una subasta.
- **Bid:** El precio más alto al que alguien está dispuesto a *comprarte* la opción ahora mismo
- **Ask:** El precio más bajo al que alguien está dispuesto a *venderte* la opción ahora mismo
- **Spread (Ask − Bid):** Tu coste de entrada/salida. Un spread de $0.50 en una opción
  que vale $5 = 10% de coste implícito. Siempre busca opciones con spreads pequeños.

**Mid:** (Bid + Ask) / 2. El precio "justo" estimado. En este dashboard usamos el Mid
para calcular la IV, ya que es más representativo que el último precio negociado.

**IV %:** La volatilidad implícita calculada usando la **fórmula cerrada del paper
arxiv:2604.24480** (ver tab Guía Completa). Es el "precio" de la opción expresado
en unidades de volatilidad anual esperada. Permite comparar el "coste" de protección
entre strikes y vencimientos distintos de forma directa.

**Delta:** La sensibilidad del precio de la opción a un movimiento de $1 en el activo.
Para un put siempre es negativo (si el activo sube, el put baja):
- Delta −0.10: opción muy OTM, cambia poco con el precio y representa una zona de baja probabilidad.
- Delta −0.50: opción aproximadamente ATM, con sensibilidad elevada al precio.
- Delta −0.90: opción muy ITM, se mueve casi como el propio activo

**Gamma:** Cómo cambia el delta por cada $1 de movimiento. Alto gamma = el delta
cambia rápido. Las opciones ATM y las de vencimiento cercano tienen el gamma más alto.
Importante para hedgers que ajustan su delta constantemente.

**Vega:** Cuánto sube (o baja) el precio de la opción si la IV sube 1%.
Un vega alto significa que el precio de la opción es muy sensible a cambios en la
volatilidad del mercado. Las opciones a largo plazo tienen vega más alto.

**Volumen y OI (Open Interest):**
- Volumen = contratos negociados durante la sesión.
- OI = total de contratos abiertos existentes.
- Opciones con OI bajo son ilíquidas: el spread es amplio y es difícil salir de la posición

---

#### Código de colores de la IV

- Rojo (IV > 40%): opción con volatilidad implícita elevada. El mercado asigna una
  prima de riesgo alta a ese strike.
  Puede ser interesante *vender* (cobrar la prima), con la gestión de riesgo adecuada.
- Amarillo (IV 20–40%): nivel intermedio.
- Verde (IV < 20%): opción con volatilidad implícita reducida.
  Puede ser interesante *comprar* como seguro de cartera a buen precio.

</div>
        """, unsafe_allow_html=True)

    # Filtros
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sel_exp = st.selectbox(
            "Vencimiento:", ['Todos'] + sorted(df['expiry'].unique().tolist())
        )
    with fc2:
        min_oi = st.number_input("OI mínimo:", 0, value=0, step=10)
    with fc3:
        min_vol = st.number_input("Volumen mínimo:", 0, value=0, step=10)
    with fc4:
        iv_rng = st.slider("Rango IV (%):", 0.0, 200.0, (0.0, 150.0), 1.0)

    filtered = df.copy()
    if sel_exp != 'Todos':
        filtered = filtered[filtered['expiry'] == sel_exp]
    filtered = filtered[
        (filtered['open_interest'] >= min_oi) &
        (filtered['volume'] >= min_vol) &
        (filtered['iv_pct'] >= iv_rng[0]) &
        (filtered['iv_pct'] <= iv_rng[1])
    ].sort_values(['expiry', 'K']).reset_index(drop=True)

    st.markdown(f"**{len(filtered)} contratos** mostrados | Spot actual: **{spot:,.2f}**")

    display = filtered[[
        'expiry', 'K', 'bid', 'ask', 'mid_price', 'iv_pct',
        'delta', 'gamma', 'vega', 'volume', 'open_interest', 'log_moneyness'
    ]].rename(columns={
        'expiry': 'Vencimiento', 'K': 'Strike (K)',
        'bid': 'Bid', 'ask': 'Ask', 'mid_price': 'Mid',
        'iv_pct': 'IV %', 'delta': 'Delta', 'gamma': 'Gamma',
        'vega': 'Vega', 'volume': 'Volumen',
        'open_interest': 'OI', 'log_moneyness': 'Log-Moneyness',
    })

    def color_iv_cell(val):
        if val > 40:   return 'background-color:#3a1010; color:#ff8888'
        elif val > 20: return 'background-color:#2a2510; color:#ffcc66'
        else:          return 'background-color:#102010; color:#88cc88'

    styled = (
        display.style
        .map(color_iv_cell, subset=['IV %'])
        .format({
            'Strike (K)':   '{:.1f}',
            'Bid':          '{:.3f}',
            'Ask':          '{:.3f}',
            'Mid':          '{:.3f}',
            'IV %':         '{:.1f}',
            'Delta':        '{:.3f}',
            'Gamma':        '{:.5f}',
            'Vega':         '{:.4f}',
            'Log-Moneyness':'{:.3f}',
        })
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # Histograma de IV
    st.markdown("#### Distribución de IV en la cadena filtrada")
    fig_hist = go.Figure(go.Histogram(
        x=filtered['iv_pct'], nbinsx=40,
        marker_color='#4488ff', opacity=0.8,
        hovertemplate='IV: %{x:.1f}%<br>Contratos: %{y}<extra></extra>',
    ))
    fig_hist.update_layout(
        template=TMPL, paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        title='Distribución de Volatilidad Implícita',
        xaxis=dict(title='IV (%)', ticksuffix='%'),
        yaxis=dict(title='Número de contratos'),
        height=280, margin=dict(l=50, r=20, t=40, b=50),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GUÍA COMPLETA
# ══════════════════════════════════════════════════════════════════════════════

with tab6:

    st.markdown("## Guía completa: de Black-Scholes a la superficie de volatilidad")
    st.markdown(
        "*Todo lo que necesitas para entender este dashboard: qué son las opciones, "
        "qué es la volatilidad implícita, cómo funciona la fórmula matemática del paper, "
        "y cómo leer cada gráfico. Sin conocimientos previos.*"
    )

    with st.expander("1. Qué es una opción financiera", expanded=True):
        st.markdown("""
#### La idea fundamental

Una **opción financiera** te da el **derecho** (no la obligación) de comprar o vender
un activo a un precio acordado (el *strike*), en o antes de una fecha concreta (el *vencimiento*).
Ese derecho tiene un precio que pagas hoy: la **prima**.

Una analogía sencilla es un contrato de seguro:
- Pagas una prima mensual de 50€
- Si tu casa se incendia, la aseguradora te paga la reparación
- Si no pasa nada malo ese mes, pierdes los 50€ pero recuperas tranquilidad

Las opciones funcionan igual, pero para carteras de inversión.

---

#### Los dos tipos

**Call (opción de compra):** Te da el derecho a *comprar* el activo al strike K.
Ganas cuando el activo sube por encima de K. Un call es una apuesta alcista con pérdida limitada.

**Put (opción de venta):** Te da el derecho a *vender* el activo al strike K.
Ganas cuando el activo cae por debajo de K. Un put es un seguro contra caídas.

En este dashboard nos centramos en **puts**, que son las más usadas como protección de cartera.

---

#### Ejemplo concreto con números reales

Imagina que tienes una cartera de SPY (S&P 500 ETF) a $560 y quieres protegerla.

Compras un put con estos términos:
- **Strike:** $530 (el seguro se activa si SPY cae por debajo de $530)
- **Vencimiento:** 2 meses desde hoy
- **Prima:** $6 (lo que pagas hoy por este seguro)

¿Qué pasa al vencimiento?

| Escenario | Precio SPY | El put vale | Tu ganancia neta |
|-----------|-----------|-------------|-----------------|
| Subida fuerte | $620 | $0 (caduca sin valor) | −$6 (solo la prima) |
| Sin cambios | $558 | $0 (K < spot) | −$6 (solo la prima) |
| Caída moderada | $520 | $10 (= 530−520) | +$4 (10 − 6 prima) |
| Crash fuerte | $460 | $70 (= 530−460) | +$64 (70 − 6 prima) |

El put funciona como un seguro de cartera: el coste máximo es la prima ($6), y la
protección aumenta cuanto más cae el mercado.

---

#### Terminología clave

| Término | Significado |
|---------|-------------|
| **ATM (At The Money)** | El strike es aproximadamente igual al precio actual: K ≈ Spot |
| **OTM (Out of The Money)** | Para un put: K < Spot; requiere una caída para tener valor intrínseco |
| **ITM (In The Money)** | Para un put: K > Spot; ya tiene valor intrínseco |
| **Prima** | El precio de la opción que pagas hoy |
| **Vencimiento** | La fecha en que expira el contrato |
| **Ejercicio** | Hacer uso del derecho que da la opción |
        """)

    with st.expander("2. Qué es la volatilidad implícita"):
        st.markdown("""
#### Primero, ¿qué es la volatilidad?

La **volatilidad** es la medida estadística de cuánto fluctúa un activo. Se expresa como
desviación estándar de los retornos diarios, anualizada en porcentaje.

- **IV 10%:** El activo se mueve poco. Mercado muy tranquilo.
- **IV 20%:** Movimiento "normal" para un índice de renta variable grande.
- **IV 50%:** Alta incertidumbre. El activo puede oscilar mucho en ambas direcciones.
- **IV 80%:** Crisis activa. El mercado está en modo pánico.

Una regla rápida: una IV del X% implica que el activo podría moverse aproximadamente
±X%/√52 en una semana (±X%/√12 en un mes).

---

#### ¿Por qué "implícita"? El problema inverso

La fórmula de Black-Scholes toma la volatilidad como *entrada* para calcular el precio
de una opción. Pero en el mercado real sucede lo contrario: vemos el precio de la opción
y necesitamos encontrar qué volatilidad lo justifica.

Es como si supieras que el seguro de hogar cuesta 80€/mes, y de ahí tuvieras que inferir
qué probabilidad le asigna la aseguradora a que tu casa sufra daños.

La volatilidad implícita es la respuesta a esa pregunta: es la volatilidad que, introducida
en Black-Scholes, reproduce exactamente el precio de mercado de la opción.

---

#### ¿Por qué es la medida más importante del mercado de opciones?

Porque permite comparar el "coste" de distintas opciones en términos homogéneos:

- Un put de SPY a 6 meses puede valer $15. Sin IV, ese precio no es comparable de forma directa.
- Ese mismo put tiene IV del 22%. Históricamente, SPY ha tenido IV de 15–25%.
  Esa lectura lo sitúa en un rango habitual para ese activo.

La IV convierte los precios de opciones en una escala común y comparable.

---

#### El VIX: la IV del mercado entero

El **VIX** (publicado por CBOE) es la IV promedio ponderada de opciones del S&P 500
para los próximos 30 días. Es una referencia agregada del nivel de estrés esperado en el mercado:

| VIX | Estado del mercado |
|-----|-------------------|
| < 12 | Euforia. Inversores complacientes. Posible señal de sobrevaloración. |
| 12–20 | Rango normal. |
| 20–30 | Cautela. Cierta preocupación en el mercado. |
| 30–50 | Miedo. Corrección significativa en curso o temida. |
| > 50 | Pánico. Crisis activa. (COVID: 82, GFC: 89) |
        """)

    with st.expander("3. Black-Scholes: el modelo y el problema inverso"):
        st.markdown("""
Fischer Black, Myron Scholes y Robert Merton publicaron en 1973 la fórmula que
revolucionó las finanzas y les valió el Premio Nobel de Economía en 1997.
Asumiendo que el activo sigue un movimiento browniano geométrico bajo la medida
de riesgo neutral, el precio de no-arbitraje de una **call europea** es:
        """)
        st.latex(r"""
C \;=\; S\,\Phi(d_1) \;-\; K e^{-rT}\,\Phi(d_2)
        """)
        st.markdown("donde la CDF normal aparece en dos argumentos:")
        st.latex(r"""
d_1 = \frac{\ln(S/K) +\bigl(r + \tfrac{1}{2}\sigma^2\bigr)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}
        """)
        st.markdown("""
| Símbolo | Significado | Ejemplo numérico |
|---------|------------|-----------------|
| $S$ | Precio spot del activo hoy | $560 |
| $K$ | Strike del contrato | $530 |
| $r$ | Tasa libre de riesgo continua | $0.043$ (4.3 %) |
| $T$ | Tiempo al vencimiento (años) | $0.167$ (2 meses) |
| $\\sigma$ | Volatilidad anual — **la incógnita** | ? |
| $\\Phi(\\cdot)$ | CDF normal estándar | tabla/función |

Para el **put** se usa la paridad put-call, que es una relación de no-arbitraje exacta:
        """)
        st.latex(r"P = C - S + K e^{-rT}")
        st.markdown("""
---
#### Reformulación en términos del forward: la forma canónica

Definimos el **precio forward** $F = Se^{rT}$ (precio justo del activo entregado en $T$)
y el **factor de descuento** $D = e^{-rT}$. Dividiendo todo entre $D \\cdot F$
obtenemos el precio normalizado:
        """)
        st.latex(r"c \;=\; \frac{C}{D\cdot F}")
        st.markdown("y el **log-moneyness** (posición relativa del strike respecto al forward):")
        st.latex(r"k \;=\; \ln\!\left(\frac{K}{F}\right)")
        st.markdown("""
| Valor de $k$ | Significado | Para un put |
|-------------|------------|------------|
| $k = 0$ | ATM — el strike coincide con el forward | En el dinero |
| $k < 0$ | $K < F$ | Put OTM — necesita que el mercado caiga |
| $k > 0$ | $K > F$ | Put ITM — ya tiene valor intrínseco |

Introduciendo la **volatilidad total** $v = \\sigma\\sqrt{T}$, la fórmula BS se reduce a
solo dos variables $(v, k)$, sin depender de $S$, $K$, $r$, $T$ por separado:
        """)
        st.latex(r"""
c \;=\; \Phi\!\left(\frac{-k}{v}+\frac{v}{2}\right)
      \;-\; e^{k}\,\Phi\!\left(\frac{-k}{v}-\frac{v}{2}\right)
        """)
        st.markdown("""
---
#### El problema de inversión: 52 años sin solución

La función $c(v)$ es estrictamente creciente y diferenciable en $v > 0$, pero su
inversa $v(c)$ no existe en forma algebraica cerrada. Para obtener $\\sigma$ dada
la observación de mercado $C$, durante 52 años la única opción fue resolver numéricamente:

**Newton-Raphson clásico:** partiendo de un $\\sigma_0$ inicial, iterar
        """)
        st.latex(r"""
\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{mercado}}}
                                {\mathcal{V}(\sigma_n)},
\quad
\mathcal{V} = \frac{\partial C}{\partial \sigma} = S\,\phi(d_1)\sqrt{T}
        """)
        st.markdown("""
donde $\\phi$ es la PDF normal. Converge en 5–10 iteraciones, pero es **lento**
para carteras con millones de opciones repriceadas en tiempo real, y puede **divergir**
en opciones muy OTM o con vencimientos muy cortos.
        """)

    with st.expander("4. Teoría matemática: fórmula del paper arxiv:2604.24480"):

        st.markdown("### La distribución Gaussiana Inversa (IG)")
        st.markdown("""
La **distribución Gaussiana Inversa** $\\text{IG}(\\mu, \\lambda)$, también llamada
distribución de Wald, modela el tiempo de primer paso de un movimiento browniano
con drift: el tiempo que tarda en cruzar un nivel fijo por primera vez.
Su PDF para $x > 0$ es:
        """)
        st.latex(r"""
f(x;\,\mu,\,\lambda)
= \sqrt{\frac{\lambda}{2\pi x^3}}\;
  \exp\!\left(-\frac{\lambda\,(x-\mu)^2}{2\,\mu^2\, x}\right)
        """)
        st.markdown("""
Sus propiedades clave:

| Parámetro | Nombre | Rol en la fórmula |
|-----------|--------|------------------|
| $\\mu > 0$ | Media | $\\mu = 2/|k|$, depende del log-moneyness |
| $\\lambda > 0$ | Forma/precisión | $\\lambda = 1$ (fijado en el paper) |
| $\\text{Var}(X) = \\mu^3/\\lambda$ | Varianza | Crece con $\\mu^3$ |

En Python, `scipy.stats.invgauss(mu)` implementa exactamente $\\text{IG}(\\mu, \\lambda=1)$,
de modo que `invgauss.ppf(p, mu=2/|k|)` es $\\mathcal{F}_{\\text{IG}}^{-1}(p;\\,2/|k|,\\,1)$.

---
        """)

        st.markdown("### El teorema fundamental del paper")
        st.markdown("""
**Definiciones previas** (ya introducidas en la Sección 3):

- $c = C/(D\\cdot F)$ — precio normalizado de la call
- $k = \\ln(K/F)$ — log-moneyness
- $v = \\sigma\\sqrt{T}$ — volatilidad total
- $m = \\min(1,\\, e^k)$ — factor normalizador
        """)
        st.latex(r"""
m = \begin{cases} 1 & \text{si } k \geq 0 \;\;(K \geq F,\; \text{put ITM}) \\
                  e^{k} = K/F & \text{si } k < 0 \;\;(K < F,\; \text{put OTM}) \end{cases}
        """)
        st.markdown("""
**Teorema (arxiv:2604.24480):** Para $k \\neq 0$, el precio normalizado de la call
Black-Scholes satisface la identidad exacta:
        """)
        st.latex(r"""
\boxed{
\frac{1 - c}{m}
\;=\;
\mathcal{F}_{\mathrm{IG}}\!\left(\frac{4}{v^2};\;\frac{2}{|k|},\;1\right)
}
        """)
        st.markdown("""
donde $\\mathcal{F}_{\\text{IG}}(\\cdot\\,;\\,\\mu,\\lambda)$ es la **CDF** de la distribución
Gaussiana Inversa. Esto establece una correspondencia uno-a-uno entre el precio
normalizado de la opción $c$ y la CDF de la IG evaluada en $x = 4/v^2$.

La identidad se demuestra a partir de la representación del precio de la call de
Black-Scholes como la probabilidad de cruce de nivel de un movimiento browniano
con drift, y la conocida expresión de esa probabilidad en términos de la CDF de la IG.
La prueba completa aparece en la proposición 1 del paper.

---
        """)

        st.markdown("### La inversión: de precio a volatilidad")
        st.markdown("""
Dado el precio de mercado $C$ (y por tanto $c$), la identidad anterior es una
ecuación en $v$. Despejando:
        """)
        st.latex(r"""
\frac{4}{v^2}
\;=\;
\mathcal{F}_{\mathrm{IG}}^{-1}\!\left(\frac{1-c}{m};\;\frac{2}{|k|},\;1\right)
\;=:\; x^*
        """)
        st.markdown("De $x^* = 4/v^2$ se obtiene $v$ directamente:")
        st.latex(r"v = \frac{2}{\sqrt{x^*}}")
        st.markdown("Y finalmente la volatilidad anualizada $\\sigma = v/\\sqrt{T}$:")
        st.latex(r"""
\boxed{
\sigma
= \frac{2}{\sqrt{T}} \cdot \left[
    \mathcal{F}_{\mathrm{IG}}^{-1}\!\left(\frac{1-c}{m};\;\frac{2}{|k|},\;1\right)
  \right]^{-1/2}
}
        """)
        st.markdown("""
Este es exactamente el resultado del paper. La clave es que **el cuantil de la IG
se evalúa en un único punto**, sin ningún bucle.

---
        """)

        st.markdown("### El caso límite ATM: $k \\to 0$")
        st.markdown("""
Cuando $K = F$ exactamente, $k = 0$ y $\\mu = 2/|k| \\to \\infty$.
Por el teorema central del límite aplicado a los tiempos de primer paso
de un browniano, la distribución $\\text{IG}(\\mu, 1)$ converge a una normal
conforme $\\mu \\to \\infty$. En ese límite, la fórmula IG degenera en:
        """)
        st.latex(r"""
\sigma_{\mathrm{ATM}}
= \frac{2}{\sqrt{T}} \cdot \Phi^{-1}\!\!\left(\frac{c + 1}{2}\right)
        """)
        st.markdown("""
donde $\\Phi^{-1}$ es la inversa de la CDF normal estándar (función `norm.ppf` en scipy).
Esta fórmula ATM ya era conocida antes del paper y sirve como caso de comprobación.

---
        """)

        st.markdown("### Paridad put-call: del put al call")
        st.markdown("""
La fórmula del paper trabaja con precios de **call**. Los datos de mercado más líquidos
son los puts OTM. La relación de no-arbitraje (paridad put-call) conecta ambos:
        """)
        st.latex(r"C = P + D\cdot(F - K) = P + e^{-rT}(F - K)")
        st.markdown("""
Esta igualdad es exacta y se aplica antes de calcular la IV. En el código:
```python
def iv_from_put(P, K, F, D, T):
    C = P + D * (F - K)   # ← paridad put-call exacta
    return iv_closed_form(C, K, F, D, T)
```

---
        """)

        st.markdown("### Cadena completa: del precio de mercado a la IV en el gráfico")
        st.markdown("""
Este es el flujo **exacto** que sigue este dashboard para cada opción que aparece
en cualquier gráfico. La línea final es donde se obtiene la volatilidad implícita.
        """)
        st.code("""
# ── PASO 1: precio de mercado (cadena de opciones) ────────────────────
P   = (bid + ask) / 2          # mid-price del put OTM
K   = row['strike']            # strike del contrato

# ── PASO 2: forward y descuento ──────────────────────────────────────
D   = exp(-r * T)              # D = e^{-rT}
F   = spot / D                 # F = S × e^{rT}

# ── PASO 3: paridad put-call y call sintética ─────────────────────────
C   = P + D * (F - K)          # C = P + D(F−K)   [paridad put-call]

# ── PASO 4: normalización ─────────────────────────────────────────────
c   = C / (D * F)              # c = C/(DF)        [precio normalizado]
k   = log(K / F)               # k = ln(K/F)       [log-moneyness]

# ── PASO 5: parámetro m ───────────────────────────────────────────────
m   = 1.0 if k > 0 else exp(k) # m = min(1, K/F)

# ── PASO 6: argumento de la CDF-IG ────────────────────────────────────
p   = (1 - c) / m              # p = (1−c)/m  ∈ (0,1)   [teorema del paper]

# ── PASO 7: cuantil IG y x* = 4/v²  [LA LÍNEA CLAVE] ─────────────────
mu_ig = 2.0 / abs(k)          # μ = 2/|k|
x_star = invgauss.ppf(p, mu=mu_ig)   # x* = F_IG^{-1}(p; 2/|k|, 1)

# ── PASO 8: σ = 2 / (√x* · √T) ───────────────────────────────────────
sigma = (2.0 / sqrt(x_star)) / sqrt(T)   # volatilidad implícita final

# ── RESULTADO: σ alimenta TODOS los gráficos ──────────────────────────
df['iv']     = sigma           # usado en la superficie 3D
df['iv_pct'] = sigma * 100     # usado en heatmap, term structure, skew y smile
        """, language="python")

        st.markdown("---")
        st.markdown("### Cómo pasamos de IVs individuales a una superficie")
        st.markdown("""
El paper resuelve **un problema puntual**: dada una opción con precio de mercado,
strike y vencimiento, calcula su volatilidad implícita Black-Scholes. Es decir, produce
un punto:

| Variable | En la app | Qué representa |
|----------|-----------|----------------|
| $x$ | `log_moneyness = ln(K/F)` | Distancia relativa del strike al forward |
| $y$ | `T` | Tiempo al vencimiento en años |
| $z$ | `iv_pct` | Volatilidad implícita calculada con el paper |

La superficie visual se construye juntando miles de puntos de este tipo. Como el mercado
solo cotiza algunos strikes y algunas fechas, la nube no cae en una rejilla perfecta.
Por eso usamos una interpolación:
        """)
        st.code("""
xi = np.linspace(xs.min(), xs.max(), 60)      # rejilla regular de moneyness
yi = np.linspace(ys.min(), ys.max(), 50)      # rejilla regular de vencimientos
XI, YI = np.meshgrid(xi, yi)

ZI = griddata(
    points=(xs, ys),     # puntos reales: log-moneyness y T
    values=zs,           # IV real calculada con la formula del paper
    xi=(XI, YI),
    method="linear"      # interpolacion lineal entre contratos cercanos
)
        """, language="python")
        st.markdown("""
`griddata(..., method="linear")` triangula internamente la nube irregular de puntos y,
dentro de cada triángulo, aproxima la IV como un plano. En términos prácticos:

- si dos strikes vecinos tienen IV 20% y 24%, un strike intermedio puede quedar cerca
  de 22%;
- si dos vencimientos vecinos tienen IV 18% y 21%, un vencimiento intermedio se estima
  entre ambos;
- si una zona no tiene contratos suficientes alrededor, la interpolación puede dejar
  huecos o producir una lámina menos fiable.

Esto significa que hay que distinguir dos objetos:

| Objeto | Sale del paper | Es interpolado | Uso correcto |
|--------|----------------|----------------|--------------|
| Puntos de la nube | Sí | No | IV contrato por contrato |
| Superficie transparente 3D | Indirectamente | Sí | Visualizar forma global |
| Mapa de calor | Indirectamente | Sí | Leer patrones por zonas |
| ATM, 25Δ y 10Δ | Sí, y luego interpolación 1D | Sí | Comparar vencimientos |

La app no calibra una superficie libre de arbitraje. Para producción institucional se
usaría normalmente SVI, SABR o un ajuste con restricciones de no arbitraje. Aquí el
objetivo es didáctico: mostrar cómo la fórmula explícita convierte precios reales en
IVs y cómo esas IVs dibujan el skew observado por el mercado.
        """)

        st.markdown("---")
        st.markdown("### Rendimiento: fórmula cerrada vs Newton-Raphson")
        st.markdown("""
Resultados del paper sobre 328 casos de prueba (volatilidades totales 0.01–2.00,
deltas 0.05–0.95):
        """)
        st.markdown("""
| Métrica | Newton-Raphson | **Fórmula cerrada (paper)** |
|---------|:--------------:|:---------------------------:|
| Iteraciones por opción | 5–10 | **0** |
| Tiempo por evaluación | 1.038 μs | **0.305 μs** |
| Factor de velocidad | 1× | **3.4× más rápido** |
| Error medio absoluto | $\\sim 10^{-14}$ | $\\mathbf{2.24 \\times 10^{-16}}$ |
| ¿Puede divergir? | Sí (OTM extremo) | **No — siempre converge** |

Para una cartera con 500,000 opciones repriceadas cada segundo:
        """)
        st.latex(r"""
\underbrace{500{,}000 \times 1.038\,\mu\text{s}}_{\text{Newton-Raphson} \approx 0.52\,\text{s}}
\quad\longrightarrow\quad
\underbrace{500{,}000 \times 0.305\,\mu\text{s}}_{\text{Fórmula cerrada} \approx 0.15\,\text{s}}
        """)

        st.markdown("---")
        st.markdown("### Verificación: consistencia de la fórmula")
        st.markdown("""
Para cualquier opción válida en el dashboard, se puede comprobar que:
        """)
        st.latex(r"""
\mathcal{F}_{\mathrm{IG}}\!\left(\frac{4}{\hat{\sigma}^2\, T};\;\frac{2}{|k|},\;1\right)
\;=\;
\frac{1 - c}{m}
\;\;\pm\;\; 10^{-14}
        """)
        st.markdown("""
donde $\\hat{\\sigma}$ es la IV calculada. La igualdad se cumple hasta el límite
de precisión de punto flotante en doble precisión (`float64`).
        """)

    with st.expander("5. La superficie de volatilidad: por qué no es plana"):
        st.markdown("""
#### El mundo de Black-Scholes vs la realidad

**En el modelo de Black-Scholes:** la volatilidad σ es constante. Esto significa que la
IV debería ser idéntica para todas las opciones de un mismo activo. La superficie sería
un plano completamente horizontal. Todos los puts del mismo activo, a cualquier strike
y vencimiento, tendrían la misma IV.

**En la realidad:** la superficie tiene forma de cuña inclinada. Esto ocurre porque
el modelo Black-Scholes tiene supuestos que no se cumplen:

1. **Los retornos no son normales.** La distribución normal tiene colas muy finas.
   Los mercados reales tienen colas izquierdas "gordas": los crashes ocurren mucho más
   frecuentemente de lo que la normal predice. El Lunes Negro (1987) hubiera tenido
   probabilidad ≈ 10⁻²⁰ bajo una normal. Fue real.

2. **La volatilidad no es constante.** Hay períodos de calma y de tormenta. La
   volatilidad se agrupa (volatility clustering): cuando hay alta volatilidad un día,
   suele haber alta volatilidad al día siguiente también.

3. **La demanda de puts OTM es estructuralmente alta.** Los fondos de pensiones,
   hedge funds y gestores de patrimonios compran puts OTM de forma sistemática para
   proteger sus carteras. Esa presión de demanda constante sube el precio (y la IV)
   de los puts OTM más allá de lo que justifica el modelo.

---

#### Las dos dimensiones de la superficie

**Dimensión horizontal — El Skew (por strike):**
La IV varía según el strike. Los puts OTM tienen IV más alta que los ATM.
Esto crea la forma de "mueca" (smirk) que ves en la sección transversal (Tab: Sonrisa).

**Dimensión en profundidad — La Term Structure (por vencimiento):**
La IV también varía según el tiempo al vencimiento. Lo que se llama "estructura temporal".
Normalmente la IV sube con el vencimiento (contango), pero en crisis se invierte (backwardation).

La superficie de volatilidad captura ambas dimensiones simultáneamente, dando una imagen
completa de cómo el mercado precio el riesgo en todos los escenarios y plazos posibles.

---

#### ¿Qué información "esconde" la superficie?

Los matemáticos han demostrado que de la superficie de volatilidad se puede extraer,
bajo ciertas condiciones, la distribución de probabilidad que el mercado le asigna al
precio futuro del activo. Es decir, la superficie de volatilidad **es** el resumen de
todas las expectativas del mercado sobre el futuro.

Si la superficie tiene la pared izquierda muy pronunciada, el mercado le asigna mayor
probabilidad a caídas bruscas que la que asignaría una distribución normal. Es una
medida de la prima de riesgo de cola incorporada en los precios.
        """)

    with st.expander("6. Greeks: delta, gamma y vega"):
        st.markdown("""
Los **Greeks** son las derivadas parciales del precio de la opción respecto a cada
parámetro. Miden la sensibilidad del precio a cambios en el mercado. Son imprescindibles
para gestionar una cartera de opciones.

---

#### Delta (Δ): la sensibilidad al precio del activo

**Definición:** cambio en el precio de la opción por cada $1 de cambio en el spot.

Para un **put**: siempre negativo (entre −1 y 0). Si el activo sube $1, el put baja.
- Delta = −0.05: opción muy OTM. El precio casi no cambia con el spot.
- Delta = −0.25: opción moderadamente OTM. El "seguro estándar" de la industria.
- Delta = −0.50: opción ATM. Cambia $0.50 por cada $1 del activo.
- Delta = −0.90: opción muy ITM. Se comporta casi como tener el activo en corto.

**Delta ≈ probabilidad de estar ITM al vencimiento** (no es exactamente igual, pero
es una aproximación muy útil para el trading).

**Uso práctico del delta:** Para neutralizar el riesgo de precio de una opción
(delta hedging), necesitas tener en el activo subyacente una posición de tamaño
proporcional al delta. Por ejemplo, un put con delta −0.25 sobre 100 acciones se
cubre comprando 25 acciones del activo.

---

#### Gamma (Γ): la aceleración del delta

**Definición:** cambio en el delta por cada $1 de cambio en el spot.

Gamma siempre positivo para el comprador de opciones. Si compras un put:
- Gamma alto: el delta cambia rápidamente ante movimientos del subyacente.
- Gamma bajo: el delta cambia lentamente

Gamma es máximo en opciones ATM y en opciones de vencimiento muy próximo. Una opción
que vence mañana y está ATM puede tener un gamma enorme: el resultado cambia radicalmente
con movimientos pequeños del precio.

---

#### Vega (ν): la sensibilidad a la volatilidad

**Definición:** cambio en el precio de la opción por cada 1% de cambio en la IV.

Siempre positivo para el comprador de opciones (cualquier tipo: put o call). Si la
volatilidad del mercado sube, el precio de todas las opciones sube.

Vega es mayor en opciones ATM y en opciones de largo vencimiento. Un put a 2 años tiene
mucho más vega que uno a 1 semana: un cambio en la IV tiene mucho mayor impacto en
una opción que tiene mucho tiempo por delante.

**Uso práctico:** Si compras opciones (puts) buscando protección, también estás
automáticamente comprando volatilidad (vega positivo). Si la IV sube, tus opciones
suben de valor aunque el mercado no se mueva. Esto es lo que hacen los fondos que
"compran volatilidad" como activo.

---

#### La tabla de Greeks en la cadena de opciones

En el tab "Cadena de Opciones" puedes ver delta, gamma y vega para cada put.
Los valores han sido calculados con la fórmula de Black-Scholes usando la IV
que acabamos de calcular con la fórmula cerrada del paper:

```python
sq    = iv * sqrt(T)
d1    = log(F/K) / sq + sq/2       # d1 de Black-Scholes
delta = norm.cdf(d1) - 1.0         # Δ del put = N(d1) - 1
gamma = norm.pdf(d1) / (F * sq)    # Γ = N'(d1) / (F × σ√T)
vega  = F * D * norm.pdf(d1) * sqrt(T)  # ν = F × D × N'(d1) × √T
```
        """)

    with st.expander("7. Lectura combinada de las señales"):
        st.markdown("""
Ningún indicador aislado es suficiente. El arte del análisis de opciones está en
combinar todas las señales y construir una imagen coherente del estado del mercado.

---

#### Tabla de diagnóstico rápido

| Indicador | Nivel bajo | Nivel intermedio | Nivel alto |
|-----------|--------------|-----------------|--------------|
| ATM IV corto plazo | < 15% | 15–30% | > 40% |
| Estructura temporal | Contango pronunciado | Plana | Backwardation |
| Skew promedio | < 5 pt | 5–12 pt | > 15 pt |
| Superficie 3D | Suave y gradual | Cuña moderada | Pared vertical izquierda |

---

#### Escenarios típicos y su lectura

**Escenario A — Mercado alcista tranquilo:**
- ATM IV corto: 12–16%
- Curva: contango suave, sube gradualmente con el vencimiento
- Skew: 3–5 puntos, barras verdes en todos los vencimientos
- Superficie: plana con ligera inclinación izquierda, sin picos
- **Lectura:** La protección mediante puts OTM tiene una prima relativa reducida.

**Escenario B — Corrección en curso (−10% a −20%):**
- ATM IV corto: 25–40%
- Curva: tendiendo a plana o backwardation en los primeros vencimientos
- Skew: 8–15 puntos, barras rojas en vencimientos cortos
- Superficie: cuña pronunciada, pared izquierda visible
- **Lectura:** La protección mediante puts incorpora una prima elevada frente al escenario normal.

**Escenario C — Crisis aguda (−30%+):**
- ATM IV corto: 50–80%+
- Curva: backwardation severa (IV a 1 semana > IV a 1 año)
- Skew: 20–35+ puntos, barras rojas gigantes en todos los vencimientos
- Superficie: pared vertical pronunciada en el extremo izquierdo
- **Acción:** Momento de máxima oportunidad para vendedores de volatilidad con tolerancia al riesgo.

---

#### La señal de alarma compuesta (las tres juntas = atención máxima)

1. **ATM IV corto plazo > percentil 80 histórico** (busca comparar con el VIX histórico)
2. **Curva temporal en backwardation** (el gráfico de term structure tiene pendiente negativa)
3. **Skew > 10 puntos** en los 3 primeros vencimientos disponibles

Cuando estas tres condiciones coinciden, el mercado está en modo defensivo. Históricamente
precede o coincide con correcciones significativas.

---

#### Referencia histórica del VIX (≈ ATM IV del S&P 500)

| Fecha | Evento | VIX máximo |
|-------|--------|-----------|
| Oct 2008 | Crisis financiera global | ~89% |
| Mar 2020 | Inicio COVID-19 | ~82% |
| Ago 2015 | Flash crash China | ~53% |
| Feb 2018 | "Volmageddon" | ~50% |
| Dic 2018 | Corrección Fed | ~36% |
| 2017 completo | Año más tranquilo del S&P | 9–12% |
| Media histórica 1990–2025 | — | ~19% |
        """)

    st.markdown("---")
    st.markdown("""
#### Lecturas recomendadas

- **[arxiv:2604.24480](https://arxiv.org/abs/2604.24480)** — El paper: fórmula cerrada para IV (2026)
- **Black & Scholes (1973)** — *The Pricing of Options and Corporate Liabilities* — El paper original
- **Gatheral, J. (2006)** — *The Volatility Surface: A Practitioner's Guide* (Wiley) — La referencia estándar
- **Natenberg, S. (1994)** — *Option Volatility and Pricing* — El libro de cabecera de los traders
- **Hull, J. (2022)** — *Options, Futures and Other Derivatives* — El libro de texto universitario más usado
    """)
