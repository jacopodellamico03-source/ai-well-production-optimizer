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
    return pd.read_excel(os.path.join(BASE_DIR, 'data', 'Volve production data.xlsx'))

@st.cache_data
def carica_brent():
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
    """Prezzo Brent medio nel periodo operativo del pozzo. Ritorna (prezzo, bool)."""
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
    ["🏠 Home", "📈 Production Forecast", "🚨 Anomaly Monitor", "⚙️ Well Optimizer"])
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

    # Arps
    try:
        p_esp, _ = curve_fit(arps_esponenziale, t, q, p0=[q[0], 0.001], maxfev=5000)
        q_e = arps_esponenziale(t, *p_esp)
        mape_esp = np.mean(np.abs((q - q_e) / np.where(q>0,q,1))) * 100
        r2_esp   = 1 - np.sum((q-q_e)**2) / np.sum((q-q.mean())**2)
    except Exception: p_esp = None

    try:
        p_iper, _ = curve_fit(arps_iperbolica, t, q, p0=[q[0],0.001,0.5],
                               bounds=([0,0,0],[1e5,1,2]), maxfev=5000)
        q_i = arps_iperbolica(t, *p_iper)
        mape_iper = np.mean(np.abs((q-q_i)/np.where(q>0,q,1)))*100
        r2_iper   = 1 - np.sum((q-q_i)**2)/np.sum((q-q.mean())**2)
    except Exception: p_iper = None

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
        fig.add_trace(go.Scatter(x=date_fut, y=arps_esponenziale(t_fut,*p_esp),
            mode='lines', line=dict(color='#3498db',width=2,dash='dash'),
            name=f'Arps Exp (MAPE {mape_esp:.1f}%)'))
    if p_iper is not None:
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

    def simula(choke_pct, bl):
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
