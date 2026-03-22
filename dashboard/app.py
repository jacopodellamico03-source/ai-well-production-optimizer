import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Well Production Optimizer", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

POZZI = ['NO 15/9-F-14 H', 'NO 15/9-F-12 H', 'NO 15/9-F-11 H']
POZZI_LABEL = {'NO 15/9-F-14 H': 'F-14 H', 'NO 15/9-F-12 H': 'F-12 H', 'NO 15/9-F-11 H': 'F-11 H'}
BBL_PER_SM3 = 6.29

# ── CARICAMENTO DATI ──────────────────────────────────────────────────────────
@st.cache_data
def carica_dati():
    """Carica il dataset di produzione Volve dal file Excel.

    Returns:
        pd.DataFrame: Dati grezzi di produzione con tutte le colonne originali.
    """
    return pd.read_excel(os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx'))

@st.cache_data
def carica_brent():
    """Carica i prezzi storici del Brent grezzo dal dataset EIA (CSV).

    Returns:
        pd.DataFrame | None: DataFrame con indice DATE e colonna BRENT (USD/bbl),
            oppure None se il file non è disponibile o contiene errori.
    """
    try:
        path = os.path.join(BASE_DIR, 'data', 'brent_prices.csv')
        df_b = pd.read_csv(path, parse_dates=['observation_date'])
        df_b.columns = ['DATE', 'BRENT']
        df_b['BRENT'] = pd.to_numeric(df_b['BRENT'], errors='coerce')
        df_b = df_b.dropna().set_index('DATE')
        return df_b
    except Exception:
        return None

@st.cache_resource
def carica_modelli_xgb():
    """Carica i modelli XGBoost e i relativi scaler per tutti e tre i pozzi.

    Returns:
        dict[str, tuple]: Dizionario che mappa il nome del pozzo a una coppia
            (XGBRegressor, MinMaxScaler). Se un modello non è trovato su disco,
            la coppia corrispondente è (None, None).
    """
    modelli = {}
    mapping = {
        'NO 15/9-F-14 H': ('xgboost_F14H.pkl', 'scaler_F14H.pkl'),
        'NO 15/9-F-12 H': ('xgboost_F12H.pkl', 'scaler_F12H.pkl'),
        'NO 15/9-F-11 H': ('xgboost_F11H.pkl', 'scaler_F11H.pkl'),
    }
    for well, (xgb_file, scaler_file) in mapping.items():
        try:
            with open(os.path.join(BASE_DIR, 'models', xgb_file), 'rb') as f:
                xgb = pickle.load(f)
            with open(os.path.join(BASE_DIR, 'models', scaler_file), 'rb') as f:
                scaler = pickle.load(f)
            modelli[well] = (xgb, scaler)
        except Exception:
            modelli[well] = (None, None)
    return modelli

def get_prezzo_medio_brent(df_prod, df_brent, fallback=80.0):
    """Calcola il prezzo medio del Brent nel periodo operativo di un pozzo.

    Args:
        df_prod (pd.DataFrame): Dati di produzione del pozzo con colonna DATEPRD.
        df_brent (pd.DataFrame | None): Prezzi Brent con indice di tipo DatetimeIndex
            e colonna BRENT. Può essere None se il dataset non è disponibile.
        fallback (float): Prezzo in USD/bbl da usare se i dati Brent non coprono
            il periodo del pozzo. Default: 80.0.

    Returns:
        tuple[float, bool]: Coppia (prezzo, trovato) dove prezzo è il Brent medio
            in USD/bbl e trovato indica se il prezzo proviene da dati reali (True)
            o dal valore di fallback (False).
    """
    if df_brent is None or len(df_prod) == 0:
        return fallback, False
    try:
        d_min = pd.Timestamp(df_prod['DATEPRD'].min())
        d_max = pd.Timestamp(df_prod['DATEPRD'].max())
        subset = df_brent[(df_brent.index >= d_min) & (df_brent.index <= d_max)]['BRENT'].dropna()
        if len(subset) > 0:
            return round(float(subset.mean()), 2), True
        return fallback, False
    except Exception:
        return fallback, False

df        = carica_dati()
df_brent  = carica_brent()
xgb_modelli = carica_modelli_xgb()

def arps_esponenziale(t, qi, Di):
    return qi * np.exp(-Di * t)

def arps_iperbolica(t, qi, Di, b):
    return qi / (1 + b * Di * t) ** (1 / b)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛢️ Well Production Optimizer")
st.sidebar.markdown("**Campo Volve — Mare del Nord**")
st.sidebar.markdown("---")
sezione = st.sidebar.radio("Navigazione",
    ["🏠 Home", "📈 Production Forecast", "🚨 Anomaly Monitor", "⚙️ Well Optimizer", "🔮 What-if Analysis"])
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Equinor Volve Field (2007–2016)")
if df_brent is not None:
    st.sidebar.caption(f"💹 Brent EIA: {len(df_brent)} giorni caricati")

# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if sezione == "🏠 Home":
    st.title("AI-Powered Well Production Optimizer")
    st.markdown("### Campo Volve — Mare del Nord (Equinor Open Dataset)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pozzi analizzati", "3")
    col2.metric("Anni di dati", "9 (2007–2016)")
    col3.metric("Modelli ML", "XGBoost + LSTM + IF")
    col4.metric("Framework", "Streamlit + Optuna")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 📋 Moduli
        | Modulo | Tecnologia | Pozzi |
        |--------|-----------|-------|
        | 📈 Production Forecast | Arps + XGBoost | F-14 H, F-12 H, F-11 H |
        | 🚨 Anomaly Monitor | Isolation Forest | F-14 H, F-12 H, F-11 H |
        | ⚙️ Well Optimizer | GBR + Bayesian Opt. | F-14 H, F-12 H, F-11 H |
        """)
    with c2:
        st.markdown("""
        #### 🎯 Obiettivi
        - Previsione produzione con modelli di decline e ML
        - Rilevamento automatico anomalie operative
        - Ottimizzazione choke per massimizzare ricavo netto
        - Business impact con prezzi Brent storici reali (EIA)
        """)

    st.markdown("---")
    df_all      = df[df['WELL_BORE_CODE'].isin(POZZI)].copy()
    df_all_prod = df_all[df_all['BORE_OIL_VOL'] > 0]
    tot_olio    = df_all_prod['BORE_OIL_VOL'].sum() / 1e6
    media_g     = df_all_prod['BORE_OIL_VOL'].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Produzione cumulativa totale", f"{tot_olio:.2f} M Sm³")
    k2.metric("Giorni totali in produzione",  f"{len(df_all_prod):,}")
    k3.metric("Portata media giornaliera",    f"{media_g:.0f} Sm³/g")

    # Grafico Brent storico nel periodo Volve
    if df_brent is not None:
        st.markdown("---")
        st.markdown("#### 💹 Prezzo Brent nel periodo Volve (2007–2016)")
        df_bv = df_brent[(df_brent.index >= '2007-01-01') & (df_brent.index <= '2016-12-31')].copy()
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=df_bv.index, y=df_bv['BRENT'],
            mode='lines', line=dict(color='#e67e22', width=1.5), name='Brent (USD/bbl)'))
        fig_b.update_layout(title='Prezzo Brent Crude — Periodo operativo Volve',
                             xaxis_title='Data', yaxis_title='USD/bbl',
                             height=300, hovermode='x unified')
        st.plotly_chart(fig_b, use_container_width=True)

    # ── Confronto multi-pozzo ─────────────────────────────────────────────────
    @st.cache_data
    def calcola_stats_multipozzo(_df, _df_brent):
        """Calcola statistiche di produzione e ricavo per tutti i pozzi.

        Args:
            _df (pd.DataFrame): Dataset completo di produzione Volve.
            _df_brent (pd.DataFrame | None): Prezzi Brent storici EIA.

        Returns:
            pd.DataFrame: Tabella con una riga per pozzo e colonne:
                Pozzo, Giorni produzione, Prod. cumulativa (M Sm³),
                Portata media/massima/finale (Sm³/g), Ricavo stimato (M USD).
        """
        rows = []
        for pozzo in POZZI:
            dp = _df[_df['WELL_BORE_CODE'] == pozzo].copy()
            dp = dp[dp['BORE_OIL_VOL'] > 0].sort_values('DATEPRD').reset_index(drop=True)
            if len(dp) == 0:
                continue
            prezzo, _ = get_prezzo_medio_brent(dp, _df_brent)
            cum_sm3   = dp['BORE_OIL_VOL'].sum()
            ricavo_m  = cum_sm3 * BBL_PER_SM3 * prezzo / 1e6
            rows.append({
                'Pozzo':                      POZZI_LABEL[pozzo],
                'Giorni produzione':          len(dp),
                'Prod. cumulativa (M Sm³)':   round(cum_sm3 / 1e6, 3),
                'Portata media (Sm³/g)':      round(dp['BORE_OIL_VOL'].mean(), 1),
                'Portata massima (Sm³/g)':    round(dp['BORE_OIL_VOL'].max(), 1),
                'Portata finale (Sm³/g)':     round(dp['BORE_OIL_VOL'].iloc[-1], 1),
                'Ricavo stimato (M USD)':     round(ricavo_m, 2),
            })
        return pd.DataFrame(rows)

    st.markdown("---")
    st.markdown("#### 📊 Confronto multi-pozzo")

    df_stats = calcola_stats_multipozzo(df, df_brent)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # Grafico grouped bar: produzione cumulativa e ricavo stimato
    fig_mp = go.Figure()
    fig_mp.add_trace(go.Bar(
        name='Prod. cumulativa (M Sm³)',
        x=df_stats['Pozzo'],
        y=df_stats['Prod. cumulativa (M Sm³)'],
        marker_color='#3498db',
        text=df_stats['Prod. cumulativa (M Sm³)'].apply(lambda v: f"{v:.3f}"),
        textposition='outside',
        yaxis='y'
    ))
    fig_mp.add_trace(go.Bar(
        name='Ricavo stimato (M USD)',
        x=df_stats['Pozzo'],
        y=df_stats['Ricavo stimato (M USD)'],
        marker_color='#2ecc71',
        text=df_stats['Ricavo stimato (M USD)'].apply(lambda v: f"{v:.1f}"),
        textposition='outside',
        yaxis='y2'
    ))
    fig_mp.update_layout(
        title='Produzione cumulativa e ricavo stimato per pozzo',
        barmode='group',
        xaxis_title='Pozzo',
        yaxis=dict(title='M Sm³', side='left'),
        yaxis2=dict(title='M USD', side='right', overlaying='y'),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_mp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif sezione == "📈 Production Forecast":
    st.title("📈 Production Forecast")
    st.markdown("Previsione produzione con **Arps Decline Curves** e **XGBoost**")

    pozzo = st.selectbox("Seleziona pozzo", POZZI, format_func=lambda x: POZZI_LABEL[x])
    df_p  = df[df['WELL_BORE_CODE'] == pozzo].copy()
    df_p  = df_p[df_p['BORE_OIL_VOL'] > 0].sort_values('DATEPRD').reset_index(drop=True)
    df_p['DAYS'] = (df_p['DATEPRD'] - df_p['DATEPRD'].min()).dt.days
    t, q  = df_p['DAYS'].values, df_p['BORE_OIL_VOL'].values

    # Arps — estrai pcov per confidence interval
    pcov_esp = pcov_iper = None
    try:
        p_esp, pcov_esp = curve_fit(arps_esponenziale, t, q, p0=[q[0], 0.001], maxfev=5000)
        q_e = arps_esponenziale(t, *p_esp)
        mape_esp = np.mean(np.abs((q - q_e) / np.where(q>0,q,1))) * 100
        r2_esp   = 1 - np.sum((q-q_e)**2) / np.sum((q-q.mean())**2)
    except Exception: p_esp = None

    try:
        p_iper, pcov_iper = curve_fit(arps_iperbolica, t, q, p0=[q[0],0.001,0.5],
                               bounds=([0,0,0],[1e5,1,2]), maxfev=5000)
        q_i = arps_iperbolica(t, *p_iper)
        mape_iper = np.mean(np.abs((q-q_i)/np.where(q>0,q,1)))*100
        r2_iper   = 1 - np.sum((q-q_i)**2)/np.sum((q-q.mean())**2)
    except Exception: p_iper = None

    def ci_arps(params, pcov, t, model_fn, n_samples=300):
        """Calcola l'intervallo di confidenza al 95% per una curva Arps via Monte Carlo.

        Campiona n_samples vettori di parametri dalla distribuzione normale multivariata
        N(params, pcov) e calcola i percentili 2.5 e 97.5 della distribuzione delle curve.

        Args:
            params (array-like): Parametri ottimali restituiti da curve_fit.
            pcov (np.ndarray): Matrice di covarianza dei parametri restituita da curve_fit.
            t (np.ndarray): Array di tempi (giorni) su cui valutare la curva.
            model_fn (callable): Funzione Arps da valutare, es. arps_esponenziale.
            n_samples (int): Numero di campioni MC. Default: 300.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: Coppia (q_low, q_high)
                con i percentili 2.5 e 97.5 per ogni punto in t.
                Restituisce (None, None) se pcov non è finita o i campioni validi
                sono meno di 10.
        """
        try:
            if pcov is None or not np.all(np.isfinite(pcov)):
                return None, None
            rng = np.random.default_rng(42)
            samples = rng.multivariate_normal(params, pcov, size=n_samples)
            curves = []
            for s in samples:
                try:
                    y = model_fn(t, *s)
                    if np.all(np.isfinite(y)) and np.all(y >= 0):
                        curves.append(y)
                except Exception:
                    pass
            if len(curves) < 10:
                return None, None
            arr = np.array(curves)
            return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
        except Exception:
            return None, None

    # XGBoost per tutti i pozzi con modello disponibile
    xgb_model, xgb_scaler = xgb_modelli.get(pozzo, (None, None))
    xgb_ok, split = (xgb_model is not None), None
    if xgb_ok:
        df_ml = df_p[['DATEPRD','DAYS','BORE_OIL_VOL','BORE_GAS_VOL',
                       'BORE_WAT_VOL','AVG_DOWNHOLE_PRESSURE','AVG_CHOKE_SIZE_P','ON_STREAM_HRS']].copy()
        df_ml = df_ml.interpolate(method='linear').bfill()
        df_ml['GOR']        = df_ml['BORE_GAS_VOL']/df_ml['BORE_OIL_VOL'].replace(0,np.nan)
        df_ml['WATERCUT']   = df_ml['BORE_WAT_VOL']/(df_ml['BORE_OIL_VOL']+df_ml['BORE_WAT_VOL']).replace(0,np.nan)
        df_ml['OIL_ROLL7']  = df_ml['BORE_OIL_VOL'].rolling(7).mean()
        df_ml['OIL_ROLL30'] = df_ml['BORE_OIL_VOL'].rolling(30).mean()
        df_ml['OIL_CUMSUM'] = df_ml['BORE_OIL_VOL'].cumsum()
        df_ml['OIL_LAG1']   = df_ml['BORE_OIL_VOL'].shift(1)
        df_ml['OIL_LAG7']   = df_ml['BORE_OIL_VOL'].shift(7)
        df_ml = df_ml.dropna().reset_index(drop=True)
        feats_xgb = ['DAYS','BORE_GAS_VOL','BORE_WAT_VOL','AVG_DOWNHOLE_PRESSURE',
                     'AVG_CHOKE_SIZE_P','ON_STREAM_HRS','GOR','WATERCUT',
                     'OIL_ROLL7','OIL_ROLL30','OIL_CUMSUM','OIL_LAG1','OIL_LAG7']
        try:
            y_xgb = xgb_model.predict(df_ml[feats_xgb].values)
            split = int(len(df_ml)*0.8)
            y_te, y_pr = df_ml['BORE_OIL_VOL'].values[split:], y_xgb[split:]
            mape_xgb = np.mean(np.abs((y_te-y_pr)/np.where(y_te>0,y_te,1)))*100
            r2_xgb   = 1 - np.sum((y_te-y_pr)**2)/np.sum((y_te-y_te.mean())**2)
        except Exception: xgb_ok = False

    giorni_fc   = st.slider("Giorni di previsione futura", 90, 730, 365, 30)
    t_fut       = np.linspace(t.max(), t.max()+giorni_fc, giorni_fc)
    date_fut    = pd.date_range(df_p['DATEPRD'].max(), periods=giorni_fc, freq='D')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['DATEPRD'], y=df_p['BORE_OIL_VOL'],
        mode='markers', marker=dict(size=3, color='#95a5a6', opacity=0.5), name='Dati reali'))
    if p_esp is not None:
        lo_e, hi_e = ci_arps(p_esp, pcov_esp, t_fut, arps_esponenziale)
        if lo_e is not None:
            fig.add_trace(go.Scatter(
                x=list(date_fut) + list(date_fut[::-1]),
                y=list(hi_e) + list(lo_e[::-1]),
                fill='toself', fillcolor='rgba(52,152,219,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=True, name='CI 95% Exp', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=date_fut, y=arps_esponenziale(t_fut,*p_esp),
            mode='lines', line=dict(color='#3498db',width=2,dash='dash'),
            name=f'Arps Exp (MAPE {mape_esp:.1f}%)'))
    if p_iper is not None:
        lo_i, hi_i = ci_arps(p_iper, pcov_iper, t_fut, arps_iperbolica)
        if lo_i is not None:
            fig.add_trace(go.Scatter(
                x=list(date_fut) + list(date_fut[::-1]),
                y=list(hi_i) + list(lo_i[::-1]),
                fill='toself', fillcolor='rgba(230,126,34,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=True, name='CI 95% Iper', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=date_fut, y=arps_iperbolica(t_fut,*p_iper),
            mode='lines', line=dict(color='#e67e22',width=2,dash='dot'),
            name=f'Arps Iper (MAPE {mape_iper:.1f}%)'))
    if xgb_ok:
        fig.add_trace(go.Scatter(x=df_ml['DATEPRD'], y=y_xgb,
            mode='lines', line=dict(color='#2ecc71',width=2),
            name=f'XGBoost (MAPE {mape_xgb:.1f}%)'))
        if split:
            fig.add_vline(x=df_ml['DATEPRD'].iloc[split].timestamp()*1000,
                          line_dash='dash', line_color='gray', annotation_text='Train|Test')
    fig.update_layout(title=f'Production Forecast — {POZZI_LABEL[pozzo]}',
                      xaxis_title='Data', yaxis_title='Portata olio (Sm³/g)',
                      hovermode='x unified', height=460,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4 if xgb_ok else 3)
    cols[0].metric("Produzione iniziale", f"{q[0]:.0f} Sm³/g")
    cols[1].metric("Produzione finale",   f"{q[-1]:.0f} Sm³/g")
    cols[2].metric("Giorni produzione",   f"{len(df_p)}")
    if xgb_ok: cols[3].metric("XGBoost MAPE (test)", f"{mape_xgb:.1f}%")

    if p_esp is not None:
        rows = [{"Modello":"Arps Esponenziale","MAPE":f"{mape_esp:.1f}%","R²":f"{r2_esp:.3f}"}]
        if p_iper is not None: rows.append({"Modello":"Arps Iperbolico","MAPE":f"{mape_iper:.1f}%","R²":f"{r2_iper:.3f}"})
        if xgb_ok: rows.append({"Modello":"XGBoost (test)","MAPE":f"{mape_xgb:.1f}%","R²":f"{r2_xgb:.3f}"})
        st.markdown("#### 📋 Performance modelli")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if xgb_ok:
            with st.expander("📊 Feature Importance XGBoost"):
                feats_xgb_names = ['DAYS','BORE_GAS_VOL','BORE_WAT_VOL','AVG_DOWNHOLE_PRESSURE',
                                    'AVG_CHOKE_SIZE_P','ON_STREAM_HRS','GOR','WATERCUT',
                                    'OIL_ROLL7','OIL_ROLL30','OIL_CUMSUM','OIL_LAG1','OIL_LAG7']
                importances = xgb_model.feature_importances_
                fi_pairs = sorted(zip(feats_xgb_names, importances), key=lambda x: x[1])
                fi_labels = [p[0] for p in fi_pairs]
                fi_values = [p[1] for p in fi_pairs]
                n = len(fi_pairs)
                fi_colors = [
                    f"rgba({int(0 + (88-0)*(i/(n-1)))}, {int(100 + (150-100)*(i/(n-1)))}, {int(0 + (44-0)*(i/(n-1)))}, 1)"
                    for i in range(n)
                ]
                fig_fi = go.Figure(go.Bar(
                    x=fi_values, y=fi_labels,
                    orientation='h',
                    marker_color=fi_colors,
                ))
                fig_fi.update_layout(
                    title=f"Feature Importance — XGBoost ({POZZI_LABEL[pozzo]})",
                    xaxis_title="Importanza relativa",
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    # ── EUR — Estimated Ultimate Recovery ────────────────────────────────────
    def calcola_eur(modello, params, q_lim=50, dt=1):
        t, q, eur = 0, modello(0, *params), 0
        while q > q_lim and t < 10000:
            eur += q * dt
            t += dt
            q = modello(t, *params)
        return eur / 1e6  # milioni di Sm³

    prod_cum_reale = df_p['BORE_OIL_VOL'].sum() / 1e6

    eur_esp  = calcola_eur(arps_esponenziale, p_esp)  if p_esp  is not None else None
    eur_iper = calcola_eur(arps_iperbolica,   p_iper) if p_iper is not None else None

    if eur_esp is not None or eur_iper is not None:
        st.markdown("---")
        st.markdown("#### 📦 EUR — Estimated Ultimate Recovery")

        e1, e2, e3 = st.columns(3)
        e1.metric("EUR Esponenziale",          f"{eur_esp:.3f} M Sm³"  if eur_esp  is not None else "N/D")
        e2.metric("EUR Iperbolico",            f"{eur_iper:.3f} M Sm³" if eur_iper is not None else "N/D")
        e3.metric("Produzione cumulativa reale", f"{prod_cum_reale:.3f} M Sm³")

        bar_labels, bar_values, bar_colors = [], [], []
        if eur_esp  is not None:
            bar_labels.append("EUR Esponenziale"); bar_values.append(eur_esp);        bar_colors.append('#3498db')
        if eur_iper is not None:
            bar_labels.append("EUR Iperbolico");   bar_values.append(eur_iper);       bar_colors.append('#e67e22')
        bar_labels.append("Prod. cumulativa reale"); bar_values.append(prod_cum_reale); bar_colors.append('#95a5a6')

        fig_eur = go.Figure(go.Bar(
            x=bar_labels, y=bar_values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in bar_values],
            textposition='outside'
        ))
        fig_eur.update_layout(
            title=f'EUR vs Produzione reale — {POZZI_LABEL[pozzo]}',
            yaxis_title='Milioni di Sm³',
            height=350, showlegend=False,
            yaxis=dict(range=[0, max(bar_values) * 1.25])
        )
        st.plotly_chart(fig_eur, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif sezione == "🚨 Anomaly Monitor":
    st.title("🚨 Anomaly Monitor")
    st.markdown("Rilevamento anomalie operative con **Isolation Forest**")

    pozzo_ad = st.selectbox("Seleziona pozzo", POZZI, format_func=lambda x: POZZI_LABEL[x], key="pozzo_ad")
    c1, c2 = st.columns(2)
    with c1: contamination = st.slider("Contaminazione attesa (%)", 1, 15, 5, 1) / 100
    with c2: n_est_if = st.slider("Numero alberi IF", 50, 500, 200, 50)

    @st.cache_data
    def prepara_anomaly(pozzo, contam, n_est):
        """Addestra Isolation Forest e calcola anomaly score per un pozzo.

        Costruisce feature ingegneristiche (GOR, watercut, rolling mean, diff),
        normalizza con MinMaxScaler e applica IsolationForest.

        Args:
            pozzo (str): Codice WELL_BORE_CODE del pozzo da analizzare.
            contam (float): Frazione attesa di anomalie (parametro contamination
                di IsolationForest), nell'intervallo (0, 0.5].
            n_est (int): Numero di alberi dell'ensemble IsolationForest.

        Returns:
            pd.DataFrame: Dati giornalieri del pozzo arricchiti con colonne
                ANOMALY_IF (-1 anomalia, 1 normale) e ANOMALY_SCORE (score grezzo).
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import MinMaxScaler
        df_p = df[df['WELL_BORE_CODE']==pozzo].copy().sort_values('DATEPRD').reset_index(drop=True)
        df_p = df_p[df_p['BORE_OIL_VOL']>0].copy().reset_index(drop=True)
        df_p['GOR']       = df_p['BORE_GAS_VOL']/df_p['BORE_OIL_VOL'].replace(0,np.nan)
        df_p['WATERCUT']  = df_p['BORE_WAT_VOL']/(df_p['BORE_OIL_VOL']+df_p['BORE_WAT_VOL']).replace(0,np.nan)
        df_p['OIL_ROLL7'] = df_p['BORE_OIL_VOL'].rolling(7).mean()
        df_p['OIL_DIFF']  = df_p['BORE_OIL_VOL'].diff()
        feats = ['BORE_OIL_VOL','GOR','WATERCUT','AVG_DOWNHOLE_PRESSURE','ON_STREAM_HRS','OIL_ROLL7','OIL_DIFF']
        df_ad = df_p[['DATEPRD']+feats].dropna().reset_index(drop=True)
        X = MinMaxScaler().fit_transform(df_ad[feats].values)
        iso = IsolationForest(n_estimators=n_est, contamination=contam, random_state=42)
        df_ad['ANOMALY_IF']    = iso.fit_predict(X)
        df_ad['ANOMALY_SCORE'] = iso.score_samples(X)
        return df_ad

    with st.spinner("Esecuzione Isolation Forest..."):
        df_ad = prepara_anomaly(pozzo_ad, contamination, n_est_if)

    normali  = df_ad[df_ad['ANOMALY_IF']==1]
    anomalie = df_ad[df_ad['ANOMALY_IF']==-1]
    st.markdown("---")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Giorni analizzati", f"{len(df_ad)}")
    k2.metric("Anomalie rilevate", f"{len(anomalie)}",
              delta=f"{len(anomalie)/len(df_ad)*100:.1f}%", delta_color="inverse")
    k3.metric("Giorni normali", f"{len(normali)}")
    k4.metric("Score medio anomalie", f"{anomalie['ANOMALY_SCORE'].mean():.3f}")
    st.markdown("---")

    fig_oil = go.Figure()
    fig_oil.add_trace(go.Scatter(x=normali['DATEPRD'], y=normali['BORE_OIL_VOL'],
        mode='markers', marker=dict(size=4,color='#2ecc71',opacity=0.6), name='Normale'))
    fig_oil.add_trace(go.Scatter(x=anomalie['DATEPRD'], y=anomalie['BORE_OIL_VOL'],
        mode='markers', marker=dict(size=9,color='#e74c3c',symbol='x'), name='Anomalia',
        customdata=anomalie[['ANOMALY_SCORE','GOR','WATERCUT']].values,
        hovertemplate="<b>%{x}</b><br>Olio:%{y:.1f}<br>Score:%{customdata[0]:.3f}<br>GOR:%{customdata[1]:.1f}<extra></extra>"))
    fig_oil.update_layout(title=f'Produzione olio — {POZZI_LABEL[pozzo_ad]}',
                           xaxis_title='Data', yaxis_title='Portata olio (Sm³/g)',
                           height=380, legend=dict(orientation='h',yanchor='bottom',y=1.02))
    st.plotly_chart(fig_oil, use_container_width=True)

    cg1,cg2 = st.columns(2)
    with cg1:
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=normali['DATEPRD'],y=normali['GOR'],mode='markers',marker=dict(size=3,color='#f39c12',opacity=0.5),name='N'))
        fg.add_trace(go.Scatter(x=anomalie['DATEPRD'],y=anomalie['GOR'],mode='markers',marker=dict(size=8,color='#e74c3c',symbol='x'),name='A'))
        fg.update_layout(title='GOR',height=300,showlegend=False,xaxis_title='Data',yaxis_title='GOR (Sm³/Sm³)')
        st.plotly_chart(fg, use_container_width=True)
    with cg2:
        fw = go.Figure()
        fw.add_trace(go.Scatter(x=normali['DATEPRD'],y=normali['WATERCUT'],mode='markers',marker=dict(size=3,color='#3498db',opacity=0.5),name='N'))
        fw.add_trace(go.Scatter(x=anomalie['DATEPRD'],y=anomalie['WATERCUT'],mode='markers',marker=dict(size=8,color='#e74c3c',symbol='x'),name='A'))
        fw.update_layout(title='Watercut',height=300,showlegend=False,xaxis_title='Data',yaxis_title='Watercut')
        st.plotly_chart(fw, use_container_width=True)

    thr = df_ad[df_ad['ANOMALY_IF']==-1]['ANOMALY_SCORE'].max()
    fs  = go.Figure()
    fs.add_trace(go.Scatter(x=df_ad['DATEPRD'],y=df_ad['ANOMALY_SCORE'],
        mode='lines',line=dict(color='#9b59b6',width=1.5),name='Score'))
    fs.add_hline(y=thr,line_dash='dash',line_color='red',
                  annotation_text=f"Soglia:{thr:.3f}",annotation_position="bottom right")
    fs.update_layout(title='Anomaly Score',xaxis_title='Data',yaxis_title='Score',height=260)
    st.plotly_chart(fs, use_container_width=True)

    st.markdown("#### 📋 Dettaglio anomalie")
    df_disp = anomalie[['DATEPRD','BORE_OIL_VOL','GOR','WATERCUT','AVG_DOWNHOLE_PRESSURE','ANOMALY_SCORE']].copy()
    df_disp.columns = ['Data','Olio Sm³/g','GOR','Watercut','Pressione (bar)','Score']
    df_disp['Data'] = df_disp['Data'].dt.strftime('%Y-%m-%d')
    df_disp = df_disp.sort_values('Score').reset_index(drop=True).round(3)
    st.dataframe(df_disp, use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# WELL OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
elif sezione == "⚙️ Well Optimizer":
    st.title("⚙️ Well Optimizer")
    st.markdown("Ottimizzazione choke con **Bayesian Optimization** (Optuna) — prezzi Brent reali EIA")

    pozzo_opt = st.selectbox("Seleziona pozzo", POZZI, format_func=lambda x: POZZI_LABEL[x], key="pozzo_opt")

    # ── Prezzo Brent reale ────────────────────────────────────────────────────
    df_pozzo_tmp = df[df['WELL_BORE_CODE']==pozzo_opt].copy()
    df_pozzo_tmp = df_pozzo_tmp[df_pozzo_tmp['BORE_OIL_VOL']>0]
    prezzo_olio, brent_ok = get_prezzo_medio_brent(df_pozzo_tmp, df_brent)

    if brent_ok:
        st.success(f"💹 Prezzo Brent storico EIA — media periodo operativo **{POZZI_LABEL[pozzo_opt]}**: **${prezzo_olio:.2f}/bbl**")
    else:
        st.warning("⚠️ Dataset Brent non trovato — usando $80/bbl di default")

    st.markdown("---")
    st.markdown("#### Parametri ottimizzazione")
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        n_trials = st.slider("Trial Optuna", 50, 300, 150, 50)
        st.metric("Prezzo Brent medio periodo", f"${prezzo_olio:.2f}/bbl")
    with cp2:
        choke_min = st.slider("Choke minimo (%)", 10, 50, 20)
        choke_max = st.slider("Choke massimo (%)", 60, 100, 100)
    with cp3:
        pen_wc    = st.slider("Penalità watercut alto (%)", 0, 50, 30, help="Se watercut > 90%") / 100
        pen_choke = st.slider("Penalità choke > 85% (%)", 0, 30, 10, help="Rischio coning") / 100

    @st.cache_data(ttl=0)
    def prepara_simulatore(pozzo):
        """Addestra il simulatore GBR choke-produzione per il Well Optimizer.

        Filtra i giorni con choke disponibile, crea feature ingegneristiche,
        addestra un GradientBoostingRegressor e calcola i valori baseline
        dagli ultimi 30 giorni del dataset filtrato.

        Args:
            pozzo (str): Codice WELL_BORE_CODE del pozzo da simulare.

        Returns:
            tuple: (sim, baseline, mape, r2, feats) dove:
                sim (GradientBoostingRegressor): Modello addestrato.
                baseline (dict): Valori medi degli ultimi 30 gg con chiavi
                    'olio', 'watercut', 'gor', 'pressure', 'choke', 'choke_curve'.
                mape (float): MAPE % sul test set del simulatore.
                r2 (float): R² sul test set del simulatore.
                feats (list[str]): Lista delle feature usate dal modello.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_percentage_error
        df_p = df[df['WELL_BORE_CODE']==pozzo].copy().sort_values('DATEPRD').reset_index(drop=True)
        df_p = df_p[(df_p['BORE_OIL_VOL']>0)&df_p['AVG_CHOKE_SIZE_P'].notna()].copy().reset_index(drop=True)
        df_p['GOR']       = df_p['BORE_GAS_VOL']/df_p['BORE_OIL_VOL'].replace(0,np.nan)
        df_p['WATERCUT']  = df_p['BORE_WAT_VOL']/(df_p['BORE_OIL_VOL']+df_p['BORE_WAT_VOL']).replace(0,np.nan)
        df_p['OIL_ROLL30']= df_p['BORE_OIL_VOL'].rolling(30).mean()
        df_p['OIL_RATIO'] = df_p['BORE_OIL_VOL']/df_p['OIL_ROLL30']
        df_p['CHOKE_BIN'] = pd.cut(df_p['AVG_CHOKE_SIZE_P'], bins=10, labels=False)
        choke_curve = df_p.groupby('CHOKE_BIN').agg(
            choke_mid=('AVG_CHOKE_SIZE_P','mean'),
            oil_mean=('BORE_OIL_VOL','mean')
        ).dropna().reset_index()
        feats = ['AVG_CHOKE_SIZE_P','WATERCUT','GOR','AVG_DOWNHOLE_PRESSURE','ON_STREAM_HRS']
        df_sim = df_p[feats+['BORE_OIL_VOL']].dropna().copy()
        X,y = df_sim[feats].values, df_sim['BORE_OIL_VOL'].values
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,shuffle=False,random_state=42)
        sim = GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,random_state=42)
        sim.fit(X_tr,y_tr)
        y_pr = sim.predict(X_te)
        mape = mean_absolute_percentage_error(y_te,y_pr)*100
        r2   = 1-np.sum((y_te-y_pr)**2)/np.sum((y_te-y_te.mean())**2)
        ultimi = df_p.tail(30)
        baseline = {
            'olio':        ultimi['BORE_OIL_VOL'].mean(),
            'watercut':    ultimi['WATERCUT'].mean(),
            'gor':         ultimi['GOR'].mean(),
            'pressure':    ultimi['AVG_DOWNHOLE_PRESSURE'].mean(),
            'choke':       ultimi['AVG_CHOKE_SIZE_P'].mean(),
            'choke_curve': choke_curve,
        }
        return sim, baseline, mape, r2, feats

    with st.spinner("Calibrazione simulatore..."):
        simulatore, baseline, mape_sim, r2_sim, features_sim = prepara_simulatore(pozzo_opt)

    # 1.2 — Validazione automatica baseline
    baseline_reale = df_pozzo_tmp.sort_values('DATEPRD').tail(30)['BORE_OIL_VOL'].mean()
    baseline_sim   = baseline['olio']
    if abs(baseline_sim - baseline_reale) / max(baseline_reale, 1e-6) > 0.20:
        st.warning(
            f"⚠️ Baseline simulatore ({baseline_sim:.1f} Sm³/g) vs dati reali ({baseline_reale:.1f} Sm³/g): "
            f"differenza > 20%. Il simulatore è calibrato su un sottoinsieme dei dati "
            f"(giorni con choke disponibile) — i valori potrebbero essere sovrastimati."
        )

    def simula(choke_pct, bl):
        """Stima la produzione giornaliera per un dato valore di choke.

        Interpola la curva choke-produzione empirica e applica il rapporto
        rispetto alla condizione baseline, clippando tra 0.3x e 2.0x.

        Args:
            choke_pct (float): Apertura choke target in percentuale (0-100).
            bl (dict): Dizionario baseline con chiavi 'olio', 'choke' e
                'choke_curve' (DataFrame con colonne choke_mid e oil_mean).

        Returns:
            float: Produzione stimata in Sm³/giorno.
        """
        curve = bl['choke_curve']
        cv, ov = curve['choke_mid'].values, curve['oil_mean'].values
        q_c = float(np.interp(choke_pct, cv, ov))
        q_b = float(np.interp(bl['choke'], cv, ov))
        ratio = np.clip((q_c/q_b) if q_b>0 else 1.0, 0.3, 2.0)
        return bl['olio'] * ratio

    ricavo_base = baseline['olio'] * BBL_PER_SM3 * prezzo_olio

    st.markdown("---")
    st.markdown("#### 🔧 Simulatore di produzione")
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("MAPE simulatore GBR", f"{mape_sim:.1f}%")
    s2.metric("R² simulatore GBR",   f"{r2_sim:.3f}")
    s3.metric("Produzione baseline",  f"{baseline['olio']:.1f} Sm³/g")
    s4.metric("Choke baseline",       f"{baseline['choke']:.1f}%")

    st.markdown("---")
    st.markdown("#### 🎛️ Simulazione manuale choke")
    choke_man = st.slider("Apertura choke (%)",
                           min_value=float(choke_min), max_value=float(choke_max),
                           value=float(np.clip(baseline['choke'], choke_min, choke_max)), step=0.5)
    q_man = simula(choke_man, baseline)
    r_man = q_man * BBL_PER_SM3 * prezzo_olio

    m1,m2,m3 = st.columns(3)
    m1.metric("Produzione stimata",  f"{q_man:.1f} Sm³/g", delta=f"{q_man-baseline['olio']:+.1f} vs baseline")
    m2.metric("Ricavo giornaliero",  f"${r_man:,.0f}",     delta=f"${r_man-ricavo_base:+,.0f} vs baseline")
    m3.metric("Ricavo mensile stimato", f"${r_man*30:,.0f}")

    choke_v = np.linspace(choke_min, choke_max, 150)
    prod_v  = [simula(c, baseline) for c in choke_v]
    ric_v   = [q * BBL_PER_SM3 * prezzo_olio for q in prod_v]

    fc = go.Figure()
    fc.add_trace(go.Scatter(x=choke_v, y=prod_v, mode='lines',
        line=dict(color='#2ecc71',width=2.5), name='Produzione (Sm³/g)', yaxis='y1'))
    fc.add_trace(go.Scatter(x=choke_v, y=ric_v, mode='lines',
        line=dict(color='#3498db',width=2,dash='dot'), name='Ricavo (USD/g)', yaxis='y2'))
    fc.add_vline(x=baseline['choke'], line_dash='dash', line_color='gray',
                  annotation_text=f"Baseline {baseline['choke']:.1f}%")
    fc.add_vline(x=choke_man, line_dash='solid', line_color='orange',
                  annotation_text=f"Manuale {choke_man:.1f}%")
    fc.update_layout(
        title='Curva di risposta choke',
        xaxis_title='Apertura choke (%)',
        yaxis=dict(title='Portata olio (Sm³/g)', title_font=dict(color='#2ecc71')),
        yaxis2=dict(title='Ricavo (USD)', title_font=dict(color='#3498db'), overlaying='y', side='right'),
        hovermode='x unified', height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fc, use_container_width=True)

    # ── Sensitivity Analysis — Prezzo Brent ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Sensitivity Analysis — Prezzo Brent")

    prezzi_range = np.arange(40, 125, 5)
    q_baseline   = simula(baseline['choke'], baseline)
    ricavi_range = [q_baseline * BBL_PER_SM3 * p for p in prezzi_range]

    fig_sa = go.Figure()

    # banda range operativo tipico $60–$100
    fig_sa.add_vrect(
        x0=60, x1=100,
        fillcolor="rgba(52, 152, 219, 0.12)",
        layer="below", line_width=0,
        annotation_text="Range operativo tipico",
        annotation_position="top left",
        annotation_font_size=11,
    )

    # linea ricavo
    fig_sa.add_trace(go.Scatter(
        x=prezzi_range, y=ricavi_range,
        mode='lines+markers',
        line=dict(color='#2ecc71', width=2.5),
        marker=dict(size=5),
        name='Ricavo giornaliero',
    ))

    # linea verticale prezzo Brent attuale
    fig_sa.add_vline(
        x=prezzo_olio,
        line_dash='dash', line_color='#e74c3c', line_width=1.8,
        annotation_text=f"Brent attuale ${prezzo_olio:.0f}",
        annotation_position="top right",
        annotation_font_color='#e74c3c',
    )

    fig_sa.update_layout(
        title="Sensitivity Analysis — Ricavo vs Prezzo Brent",
        xaxis_title="Prezzo Brent (USD/bbl)",
        yaxis_title="Ricavo giornaliero (USD)",
        height=400,
        hovermode='x unified',
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig_sa, use_container_width=True)

    # tabella 5 scenari
    scenari_prezzi = [40, 60, 80, 100, 120]
    scenari_rows = []
    for sp in scenari_prezzi:
        r_day   = q_baseline * BBL_PER_SM3 * sp
        r_month = r_day * 30
        scenari_rows.append({
            "Prezzo Brent ($/bbl)": f"${sp}",
            "Ricavo giornaliero (USD)": f"${r_day:,.0f}",
            "Ricavo mensile (USD)":     f"${r_month:,.0f}",
        })
    st.dataframe(pd.DataFrame(scenari_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🤖 Ottimizzazione bayesiana (Optuna)")

    if st.button("▶️ Esegui ottimizzazione", type="primary"):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            ch = trial.suggest_float('choke_pct', float(choke_min), float(choke_max))
            q  = simula(ch, baseline)
            r  = q * BBL_PER_SM3 * prezzo_olio
            p1 = r * pen_wc    if baseline['watercut'] > 0.90 else 0
            p2 = r * pen_choke if ch > 85 else 0
            return r - p1 - p2

        pb = st.progress(0, text="Ottimizzazione in corso...")
        study = optuna.create_study(direction='maximize')
        batch, done = max(1, n_trials//20), 0
        while done < n_trials:
            step = min(batch, n_trials-done)
            study.optimize(objective, n_trials=step, show_progress_bar=False)
            done += step
            pb.progress(done/n_trials, text=f"Trial {done}/{n_trials}...")
        pb.empty()

        choke_ott = study.best_params['choke_pct']
        q_ott     = simula(choke_ott, baseline)
        r_ott     = q_ott * BBL_PER_SM3 * prezzo_olio

        st.success(f"✅ Ottimizzazione completata — {n_trials} trial")

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Choke ottimale",        f"{choke_ott:.1f}%",
                  delta=f"{choke_ott-baseline['choke']:+.1f}% vs baseline")
        r2.metric("Produzione ottimizzata", f"{q_ott:.1f} Sm³/g",
                  delta=f"{q_ott-baseline['olio']:+.1f} vs baseline")
        r3.metric("Ricavo ottimizzato",     f"${r_ott:,.0f}/g",
                  delta=f"${r_ott-ricavo_base:+,.0f} vs baseline")
        r4.metric("Ricavo mensile aggiuntivo", f"${(r_ott-ricavo_base)*30:,.0f}")

        st.markdown("#### 📊 Confronto scenari")
        rows = []
        for nome, ch in [('Conservativo',20.0),('Baseline',baseline['choke']),
                          ('Ottimale AI',choke_ott),('Aggressivo',95.0)]:
            q_s = simula(ch, baseline)
            r_s = q_s * BBL_PER_SM3 * prezzo_olio
            rows.append({'Scenario':nome,'Choke (%)':round(ch,1),
                         'Olio (Sm³/g)':round(q_s,1),'Ricavo (USD/g)':round(r_s,0),
                         'Δ vs Baseline':f"${r_s-ricavo_base:+,.0f}"})
        df_sc = pd.DataFrame(rows)
        st.dataframe(df_sc, use_container_width=True, hide_index=True)

        fb = go.Figure(go.Bar(
            x=df_sc['Scenario'], y=df_sc['Olio (Sm³/g)'],
            marker_color=['#95a5a6','#3498db','#2ecc71','#e74c3c'],
            text=df_sc['Olio (Sm³/g)'].astype(str)+' Sm³/g', textposition='outside'))
        fb.add_hline(y=baseline['olio'], line_dash='dash', line_color='gray',
                      annotation_text='Baseline')
        fb.update_layout(title='Produzione per scenario',
                          yaxis_title='Portata olio (Sm³/g)', height=350, showlegend=False)
        st.plotly_chart(fb, use_container_width=True)

        st.markdown("#### 📉 Convergenza ottimizzazione")
        tv = [t.value for t in study.trials]
        bv = np.maximum.accumulate(tv)
        fconv = go.Figure()
        fconv.add_trace(go.Scatter(y=tv, mode='markers',
            marker=dict(size=3,color='#95a5a6',opacity=0.5), name='Trial'))
        fconv.add_trace(go.Scatter(y=bv, mode='lines',
            line=dict(color='#e74c3c',width=2), name='Best so far'))
        fconv.update_layout(title='Convergenza Optuna', xaxis_title='Trial',
                             yaxis_title='Ricavo netto (USD)', height=300,
                             legend=dict(orientation='h',yanchor='bottom',y=1.02))
        st.plotly_chart(fconv, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💰 Business Impact")
        delta_g = r_ott - ricavo_base
        b1,b2,b3 = st.columns(3)
        b1.metric("Incremento giornaliero",    f"${delta_g:,.0f}")
        b2.metric("Incremento annuale stimato", f"${delta_g*365:,.0f}")
        b3.metric("Incremento % ricavo",
                  f"{(r_ott/ricavo_base-1)*100:.1f}%" if ricavo_base>0 else "N/A")
    else:
        st.info("Configura i parametri e premi **▶️ Esegui ottimizzazione** per avviare.")

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif sezione == "🔮 What-if Analysis":
    st.title("🔮 What-if Analysis")
    st.markdown("Simula l'impatto di un cambio di apertura choke sulla produzione futura")

    pozzo_wi = st.selectbox("Seleziona pozzo", POZZI, format_func=lambda x: POZZI_LABEL[x], key="pozzo_wi")

    # ── Fit Arps esponenziale sui dati reali del pozzo ───────────────────────
    df_wi = df[df['WELL_BORE_CODE'] == pozzo_wi].copy()
    df_wi = df_wi[df_wi['BORE_OIL_VOL'] > 0].sort_values('DATEPRD').reset_index(drop=True)
    df_wi['DAYS'] = (df_wi['DATEPRD'] - df_wi['DATEPRD'].min()).dt.days
    t_wi, q_wi = df_wi['DAYS'].values, df_wi['BORE_OIL_VOL'].values

    p_wi = None
    try:
        p_wi, _ = curve_fit(arps_esponenziale, t_wi, q_wi, p0=[q_wi[0], 0.001], maxfev=5000)
    except Exception:
        st.error("Impossibile fittare la curva Arps esponenziale per questo pozzo.")

    # ── Choke baseline dai dati reali (media ultimi 30 gg con choke notna) ───
    choke_base_series = df_wi['AVG_CHOKE_SIZE_P'].dropna()
    choke_baseline = float(choke_base_series.tail(30).mean()) if len(choke_base_series) >= 1 else 50.0

    # ── Prezzo Brent: media ultimi 365 giorni disponibili nel dataset ────────
    if df_brent is not None:
        prezzo_wi = float(df_brent['BRENT'].dropna().tail(365).mean())
        brent_label = f"${prezzo_wi:.2f}/bbl (media ultimi 365 gg dataset EIA)"
    else:
        prezzo_wi = 80.0
        brent_label = "$80.00/bbl (fallback)"

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        choke_target = st.slider("Apertura choke target (%)", 10, 100,
                                  int(np.clip(choke_baseline, 10, 100)), 1)
    with c2:
        orizzonte = st.select_slider("Orizzonte temporale (giorni)",
                                      options=[30, 60, 90, 180], value=90)
    with c3:
        st.metric("Choke baseline", f"{choke_baseline:.1f}%")
        st.caption(f"Brent proxy: {brent_label}")

    if p_wi is not None:
        qi_fit, Di_fit = p_wi

        # Produzione baseline all'ultimo giorno storico
        t_last   = int(t_wi[-1])
        q_last   = float(arps_esponenziale(t_last, qi_fit, Di_fit))

        # Il rapporto choke modula il tasso di decline:
        # choke più aperto → decline più lento (moltiplicatore su Di)
        # choke più chiuso → decline più veloce
        ratio = choke_target / choke_baseline if choke_baseline > 0 else 1.0
        Di_target = Di_fit / np.clip(ratio, 0.1, 5.0)

        giorni = np.arange(1, orizzonte + 1)
        q_base_fc   = np.array([arps_esponenziale(t_last + g, qi_fit, Di_fit)    for g in giorni])
        q_target_fc = np.array([arps_esponenziale(t_last + g, q_last, Di_target) for g in giorni])
        q_base_fc   = np.maximum(q_base_fc,   0)
        q_target_fc = np.maximum(q_target_fc, 0)

        cum_base   = np.cumsum(q_base_fc)
        cum_target = np.cumsum(q_target_fc)

        date_fc = pd.date_range(df_wi['DATEPRD'].max() + pd.Timedelta(days=1), periods=orizzonte, freq='D')

        # ── Grafico doppio: portata + cumulativa ─────────────────────────────
        fig_wi = go.Figure()

        fig_wi.add_trace(go.Scatter(
            x=date_fc, y=q_base_fc,
            mode='lines', line=dict(color='#95a5a6', width=2, dash='dash'),
            name=f'Baseline ({choke_baseline:.1f}%)'))
        fig_wi.add_trace(go.Scatter(
            x=date_fc, y=q_target_fc,
            mode='lines', line=dict(color='#2ecc71', width=2),
            name=f'Choke target ({choke_target}%)'))
        fig_wi.add_trace(go.Scatter(
            x=date_fc, y=cum_base,
            mode='lines', line=dict(color='#bdc3c7', width=1.5, dash='dot'),
            name='Cum. baseline', yaxis='y2'))
        fig_wi.add_trace(go.Scatter(
            x=date_fc, y=cum_target,
            mode='lines', line=dict(color='#27ae60', width=1.5, dash='dot'),
            name='Cum. target', yaxis='y2'))

        fig_wi.update_layout(
            title=f'What-if Analysis — {POZZI_LABEL[pozzo_wi]}',
            xaxis_title='Data',
            yaxis=dict(title='Portata giornaliera (Sm³/g)'),
            yaxis2=dict(title='Produzione cumulativa (Sm³)', overlaying='y', side='right'),
            hovermode='x unified', height=460,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_wi, use_container_width=True)

        # ── KPI ──────────────────────────────────────────────────────────────
        tot_base   = float(cum_base[-1])
        tot_target = float(cum_target[-1])
        rev_base   = tot_base   * BBL_PER_SM3 * prezzo_wi
        rev_target = tot_target * BBL_PER_SM3 * prezzo_wi
        delta_rev  = rev_target - rev_base

        st.markdown("---")
        st.markdown(f"#### 📊 KPI — orizzonte {orizzonte} giorni")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Prod. cum. baseline",    f"{tot_base:,.0f} Sm³")
        k2.metric("Prod. cum. target",      f"{tot_target:,.0f} Sm³",
                  delta=f"{tot_target - tot_base:+,.0f} Sm³")
        k3.metric("Ricavo cum. baseline",   f"${rev_base:,.0f}")
        k4.metric("Ricavo cum. target",     f"${rev_target:,.0f}")
        k5.metric("Delta ricavo",           f"${delta_rev:+,.0f}",
                  delta=f"{(delta_rev/rev_base*100):+.1f}%" if rev_base > 0 else None)
    else:
        st.info("Fit Arps non disponibile — seleziona un altro pozzo.")
