from typing import Callable, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.optimize import curve_fit

from utils.data import (
    carica_dati, carica_brent, carica_modelli_xgb,
    get_prezzo_medio_brent, calcola_stats_multipozzo,
    POZZI, POZZI_LABEL, BBL_PER_SM3,
)
from utils.models import (
    arps_esponenziale, arps_iperbolica, ci_arps,
    prepara_anomaly, prepara_simulatore, simula,
)

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

df          = carica_dati()
df_brent    = carica_brent()
xgb_modelli = carica_modelli_xgb()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛢️ Well Production Optimizer")
st.sidebar.markdown("**Campo Volve — Mare del Nord**")
st.sidebar.markdown("---")
sezione = st.sidebar.radio("Navigazione",
    ["🏠 Home", "📈 Production Forecast", "🚨 Anomaly Monitor", "⚙️ Well Optimizer", "🔮 What-if Analysis", "🔧 Predictive Maintenance"])
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

        if xgb_ok:
            with st.expander("📉 Residual Analysis XGBoost"):
                y_te_res = df_ml['BORE_OIL_VOL'].values[split:]
                y_pr_res = y_xgb[split:]
                residui  = y_te_res - y_pr_res
                abs_res  = np.abs(residui)
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=y_pr_res, y=residui,
                    mode='markers',
                    marker=dict(
                        color=abs_res,
                        colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
                        showscale=True,
                        colorbar=dict(title='|Residuo|'),
                        size=6, opacity=0.7,
                    ),
                    name='Residui test set',
                ))
                fig_res.add_hline(y=0, line_dash='dash', line_color='gray')
                fig_res.update_layout(
                    title=f'Residual Plot — XGBoost ({POZZI_LABEL[pozzo]})',
                    xaxis_title='Valori predetti (Sm³/g)',
                    yaxis_title='Residui (reale - predetto)',
                    height=400,
                )
                st.plotly_chart(fig_res, use_container_width=True)

    # ── EUR — Estimated Ultimate Recovery ────────────────────────────────────
    def calcola_eur(modello: Callable, params: np.ndarray, q_lim: float = 50, dt: int = 1) -> float:
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

    with st.spinner("Esecuzione Isolation Forest..."):
        df_ad = prepara_anomaly(df, pozzo_ad, contamination, n_est_if)

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
        opex_giornaliero = st.number_input(
            "Costo operativo giornaliero (OPEX, USD/g)",
            min_value=0, max_value=500000, value=20000, step=1000,
            help="Stima basata su dati NPD per campi offshore North Sea di dimensioni simili a Volve (~$15-35/boe). Aggiornare quando disponibili dati reali.",
        )
        st.metric("Prezzo Brent medio periodo", f"${prezzo_olio:.2f}/bbl")
    with cp2:
        choke_min = st.slider("Choke minimo (%)", 10, 50, 20)
        choke_max = st.slider("Choke massimo (%)", 60, 100, 100)
    with cp3:
        pen_wc    = st.slider("Penalità watercut alto (%)", 0, 50, 30, help="Se watercut > 90%") / 100
        pen_choke = st.slider("Penalità choke > 85% (%)", 0, 30, 10, help="Rischio coning") / 100

    with st.spinner("Calibrazione simulatore..."):
        simulatore, baseline, mape_sim, r2_sim, features_sim = prepara_simulatore(df, pozzo_opt)

    # 1.2 — Validazione automatica baseline
    baseline_reale = df_pozzo_tmp.sort_values('DATEPRD').tail(30)['BORE_OIL_VOL'].mean()
    baseline_sim   = baseline['olio']
    if abs(baseline_sim - baseline_reale) / max(baseline_reale, 1e-6) > 0.20:
        st.warning(
            f"⚠️ Baseline simulatore ({baseline_sim:.1f} Sm³/g) vs dati reali ({baseline_reale:.1f} Sm³/g): "
            f"differenza > 20%. Il simulatore è calibrato su un sottoinsieme dei dati "
            f"(giorni con choke disponibile) — i valori potrebbero essere sovrastimati."
        )

    ricavo_base   = baseline['olio'] * BBL_PER_SM3 * prezzo_olio
    profitto_base = ricavo_base - opex_giornaliero

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
    p_man = r_man - opex_giornaliero

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Produzione stimata",      f"{q_man:.1f} Sm³/g", delta=f"{q_man-baseline['olio']:+.1f} vs baseline")
    m2.metric("Ricavo lordo",            f"${r_man:,.0f}",     delta=f"${r_man-ricavo_base:+,.0f} vs baseline")
    m3.metric("Profitto netto",          f"${p_man:,.0f}",     delta=f"${p_man-profitto_base:+,.0f} vs baseline")
    m4.metric("Profitto mensile stimato", f"${p_man*30:,.0f}")

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

    prezzi_range   = np.arange(40, 125, 5)
    q_baseline     = simula(baseline['choke'], baseline)
    ricavi_range   = [q_baseline * BBL_PER_SM3 * p for p in prezzi_range]
    profitti_range = [r - opex_giornaliero for r in ricavi_range]

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

    # linea ricavo lordo
    fig_sa.add_trace(go.Scatter(
        x=prezzi_range, y=ricavi_range,
        mode='lines+markers',
        line=dict(color='#2ecc71', width=2.5),
        marker=dict(size=5),
        name='Ricavo lordo',
    ))

    # linea profitto netto
    fig_sa.add_trace(go.Scatter(
        x=prezzi_range, y=profitti_range,
        mode='lines+markers',
        line=dict(color='#3498db', width=2, dash='dash'),
        marker=dict(size=5),
        name='Profitto netto',
    ))

    # linea break-even (y=0)
    fig_sa.add_hline(
        y=0,
        line_dash='dot', line_color='#e74c3c', line_width=1.5,
        annotation_text="Break-even", annotation_position="bottom right",
        annotation_font_color='#e74c3c',
    )

    # linea verticale prezzo Brent attuale
    fig_sa.add_vline(
        x=prezzo_olio,
        line_dash='dash', line_color='#e74c3c', line_width=1.8,
        annotation_text=f"Brent attuale ${prezzo_olio:.0f}",
        annotation_position="top right",
        annotation_font_color='#e74c3c',
    )

    fig_sa.update_layout(
        title="Sensitivity Analysis — Ricavo lordo e Profitto netto vs Prezzo Brent",
        xaxis_title="Prezzo Brent (USD/bbl)",
        yaxis_title="USD/giorno",
        height=400,
        hovermode='x unified',
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig_sa, use_container_width=True)

    # tabella 5 scenari
    scenari_prezzi = [40, 60, 80, 100, 120]
    scenari_rows = []
    for sp in scenari_prezzi:
        r_day = q_baseline * BBL_PER_SM3 * sp
        p_day = r_day - opex_giornaliero
        scenari_rows.append({
            "Prezzo Brent ($/bbl)":       f"${sp}",
            "Ricavo lordo/g (USD)":       f"${r_day:,.0f}",
            "Profitto netto/g (USD)":     f"${p_day:,.0f}",
            "Profitto mensile (USD)":     f"${p_day*30:,.0f}",
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
            r  = q * BBL_PER_SM3 * prezzo_olio - opex_giornaliero
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
        p_ott     = r_ott - opex_giornaliero

        st.success(f"✅ Ottimizzazione completata — {n_trials} trial")

        r1,r2,r3,r4,r5 = st.columns(5)
        r1.metric("Choke ottimale",           f"{choke_ott:.1f}%",
                  delta=f"{choke_ott-baseline['choke']:+.1f}% vs baseline")
        r2.metric("Produzione ottimizzata",   f"{q_ott:.1f} Sm³/g",
                  delta=f"{q_ott-baseline['olio']:+.1f} vs baseline")
        r3.metric("Ricavo lordo",             f"${r_ott:,.0f}/g",
                  delta=f"${r_ott-ricavo_base:+,.0f} vs baseline")
        r4.metric("Profitto netto",           f"${p_ott:,.0f}/g",
                  delta=f"${p_ott-profitto_base:+,.0f} vs baseline")
        r5.metric("Profitto mensile aggiuntivo", f"${(p_ott-profitto_base)*30:,.0f}")

        st.markdown("#### 📊 Confronto scenari")
        rows = []
        for nome, ch in [('Conservativo',20.0),('Baseline',baseline['choke']),
                          ('Ottimale AI',choke_ott),('Aggressivo',95.0)]:
            q_s = simula(ch, baseline)
            r_s = q_s * BBL_PER_SM3 * prezzo_olio
            p_s = r_s - opex_giornaliero
            rows.append({'Scenario':nome,'Choke (%)':round(ch,1),
                         'Olio (Sm³/g)':round(q_s,1),
                         'Ricavo lordo (USD/g)':round(r_s,0),
                         'Profitto netto (USD/g)':round(p_s,0),
                         'Δ profitto vs Baseline':f"${p_s-profitto_base:+,.0f}"})
        df_sc = pd.DataFrame(rows)
        st.dataframe(df_sc, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Scarica risultati ottimizzazione (CSV)",
            data=df_sc.to_csv(index=False),
            file_name=f"ottimizzazione_{POZZI_LABEL[pozzo_opt]}_{choke_ott:.0f}pct.csv",
            mime="text/csv",
        )

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
        delta_g = p_ott - profitto_base
        b1,b2,b3 = st.columns(3)
        b1.metric("Incremento profitto giornaliero",     f"${delta_g:,.0f}")
        b2.metric("Incremento profitto annuale stimato", f"${delta_g*365:,.0f}")
        b3.metric("Incremento % profitto netto",
                  f"{(p_ott/profitto_base-1)*100:.1f}%" if profitto_base > 0 else "N/A")
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

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif sezione == "🔧 Predictive Maintenance":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score

    st.title("🔧 Predictive Maintenance")
    st.markdown("Previsione anomalie future con **Random Forest** addestrato su segnali rolling")

    st.warning(
        "⚠️ Modello addestrato su dati storici limitati (3 pozzi). "
        "Le previsioni hanno valore indicativo e non operativo."
    )

    pozzo_pm = st.selectbox("Seleziona pozzo", POZZI, format_func=lambda x: POZZI_LABEL[x], key="pozzo_pm")

    @st.cache_data
    def train_maintenance_model(pozzo_key: str, contamination: float = 0.05, n_est_if: int = 200) -> Tuple:
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score

        df_pm = df[df['WELL_BORE_CODE'] == pozzo_key].copy()
        df_pm = df_pm[df_pm['BORE_OIL_VOL'] > 0].sort_values('DATEPRD').reset_index(drop=True)

        # --- Isolation Forest per label anomalie ---
        feats_if = ['BORE_OIL_VOL', 'GOR', 'WATERCUT', 'AVG_DOWNHOLE_PRESSURE']
        df_pm['GOR'] = (df_pm['BORE_GAS_VOL'] / df_pm['BORE_OIL_VOL'].replace(0, np.nan)).fillna(0)
        df_pm['WATERCUT'] = (
            df_pm['BORE_WAT_VOL'] /
            (df_pm['BORE_OIL_VOL'] + df_pm['BORE_WAT_VOL']).replace(0, np.nan)
        ).fillna(0)

        avail = [c for c in feats_if if c in df_pm.columns]
        X_if = df_pm[avail].fillna(0).values
        clf_if = IsolationForest(n_estimators=n_est_if, contamination=contamination, random_state=42)
        df_pm['ANOMALY_IF'] = clf_if.fit_predict(X_if)

        # --- Feature engineering: rolling 7 giorni ---
        roll_cols = ['BORE_OIL_VOL', 'GOR', 'WATERCUT', 'AVG_DOWNHOLE_PRESSURE']
        roll_feats = []
        for col in roll_cols:
            if col in df_pm.columns:
                df_pm[f'{col}_roll7_mean'] = df_pm[col].rolling(7, min_periods=1).mean()
                df_pm[f'{col}_roll7_std']  = df_pm[col].rolling(7, min_periods=1).std().fillna(0)
                roll_feats += [f'{col}_roll7_mean', f'{col}_roll7_std']

        # --- Target: 1 se nei prossimi 7 giorni c'è almeno una anomalia ---
        df_pm['IS_ANOMALY'] = (df_pm['ANOMALY_IF'] == -1).astype(int)
        df_pm['TARGET'] = (
            df_pm['IS_ANOMALY']
            .rolling(7, min_periods=1)
            .max()
            .shift(-7)
            .fillna(0)
            .astype(int)
        )

        df_model = df_pm[roll_feats + ['TARGET', 'DATEPRD', 'IS_ANOMALY']].dropna().reset_index(drop=True)

        X = df_model[roll_feats].values
        y = df_model['TARGET'].values

        split = int(len(df_model) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)

        THRESHOLD = 0.3

        proba_test  = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= THRESHOLD).astype(int)
        prec  = precision_score(y_test, y_pred_test, zero_division=0)
        rec   = recall_score(y_test, y_pred_test, zero_division=0)
        f1    = f1_score(y_test, y_pred_test, zero_division=0)

        proba_train  = clf.predict_proba(X_train)[:, 1]
        y_pred_train = (proba_train >= THRESHOLD).astype(int)
        prec_tr = precision_score(y_train, y_pred_train, zero_division=0)
        rec_tr  = recall_score(y_train, y_pred_train, zero_division=0)
        f1_tr   = f1_score(y_train, y_pred_train, zero_division=0)

        proba_all = clf.predict_proba(X)[:, 1]

        return df_model, proba_all, prec, rec, f1, prec_tr, rec_tr, f1_tr, split, THRESHOLD

    with st.spinner("Addestramento Random Forest..."):
        df_model, proba_all, prec, rec, f1, prec_tr, rec_tr, f1_tr, split, THRESHOLD = train_maintenance_model(pozzo_pm)

    # ── KPI ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    n_pred_correct = int(((proba_all[split:] >= THRESHOLD).astype(int) == df_model['TARGET'].values[split:]).sum())

    st.markdown(f"##### Test set (soglia {THRESHOLD})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision (test)",  f"{prec:.2f}")
    k2.metric("Recall (test)",     f"{rec:.2f}")
    k3.metric("F1-score (test)",   f"{f1:.2f}")
    k4.metric("Giorni predetti correttamente (test)", f"{n_pred_correct}")

    st.markdown(f"##### Training set (soglia {THRESHOLD})")
    t1, t2, t3 = st.columns(3)
    t1.metric("Precision (train)", f"{prec_tr:.2f}")
    t2.metric("Recall (train)",    f"{rec_tr:.2f}")
    t3.metric("F1-score (train)",  f"{f1_tr:.2f}")

    st.info(
        "Il modello identifica pattern precursori nelle ultime 7 giorni prima di un'anomalia. "
        "Con dati limitati (3 pozzi) tende a overfittare sul training set. "
        "Le probabilità mostrate nel grafico hanno valore esplorativo."
    )

    # ── Timeline probabilità anomalia ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Probabilità anomalia giorno per giorno")

    dates    = df_model['DATEPRD']
    is_anom  = df_model['IS_ANOMALY'].values
    high_risk = proba_all >= THRESHOLD

    fig_pm = go.Figure()

    # Area probabilità
    fig_pm.add_trace(go.Scatter(
        x=dates, y=proba_all,
        mode='lines', line=dict(color='#3498db', width=1.5),
        name='P(anomalia prossimi 7gg)', fill='tozeroy',
        fillcolor='rgba(52,152,219,0.15)'
    ))

    # Giorni ad alto rischio in rosso
    fig_pm.add_trace(go.Scatter(
        x=dates[high_risk], y=proba_all[high_risk],
        mode='markers', marker=dict(color='#e74c3c', size=5, opacity=0.7),
        name=f'P ≥ {THRESHOLD} (alto rischio)'
    ))

    # Soglia
    fig_pm.add_hline(y=THRESHOLD, line_dash='dash', line_color='red',
                     annotation_text=f'Soglia {THRESHOLD}', annotation_position='bottom right')

    # Anomalie reali IF come markers rossi
    anom_mask = is_anom == 1
    if anom_mask.any():
        fig_pm.add_trace(go.Scatter(
            x=dates[anom_mask], y=np.ones(anom_mask.sum()) * 1.02,
            mode='markers', marker=dict(color='#c0392b', size=8, symbol='x'),
            name='Anomalia reale (IF)', yaxis='y'
        ))

    # Linea train/test split
    if split < len(dates):
        fig_pm.add_vline(
            x=dates.iloc[split].timestamp() * 1000,
            line_dash='dash', line_color='gray',
            annotation_text='Train | Test'
        )

    fig_pm.update_layout(
        title=f'Probabilità anomalia futura — {POZZI_LABEL[pozzo_pm]}',
        xaxis_title='Data',
        yaxis=dict(title='Probabilità', range=[0, 1.1]),
        height=460, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_pm, use_container_width=True)
