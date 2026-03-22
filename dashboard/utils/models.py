import numpy as np
import pandas as pd
import streamlit as st


def arps_esponenziale(t, qi, Di):
    return qi * np.exp(-Di * t)


def arps_iperbolica(t, qi, Di, b):
    return qi / (1 + b * Di * t) ** (1 / b)


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


@st.cache_data
def prepara_anomaly(_df, pozzo, contam, n_est):
    """Addestra Isolation Forest e calcola anomaly score per un pozzo.

    Costruisce feature ingegneristiche (GOR, watercut, rolling mean, diff),
    normalizza con MinMaxScaler e applica IsolationForest.

    Args:
        _df (pd.DataFrame): Dataset completo di produzione Volve.
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
    df_p = _df[_df['WELL_BORE_CODE'] == pozzo].copy().sort_values('DATEPRD').reset_index(drop=True)
    df_p = df_p[df_p['BORE_OIL_VOL'] > 0].copy().reset_index(drop=True)
    df_p['GOR']       = df_p['BORE_GAS_VOL'] / df_p['BORE_OIL_VOL'].replace(0, np.nan)
    df_p['WATERCUT']  = df_p['BORE_WAT_VOL'] / (df_p['BORE_OIL_VOL'] + df_p['BORE_WAT_VOL']).replace(0, np.nan)
    df_p['OIL_ROLL7'] = df_p['BORE_OIL_VOL'].rolling(7).mean()
    df_p['OIL_DIFF']  = df_p['BORE_OIL_VOL'].diff()
    feats = ['BORE_OIL_VOL', 'GOR', 'WATERCUT', 'AVG_DOWNHOLE_PRESSURE', 'ON_STREAM_HRS', 'OIL_ROLL7', 'OIL_DIFF']
    df_ad = df_p[['DATEPRD'] + feats].dropna().reset_index(drop=True)
    X = MinMaxScaler().fit_transform(df_ad[feats].values)
    iso = IsolationForest(n_estimators=n_est, contamination=contam, random_state=42)
    df_ad['ANOMALY_IF']    = iso.fit_predict(X)
    df_ad['ANOMALY_SCORE'] = iso.score_samples(X)
    return df_ad


@st.cache_data(ttl=0)
def prepara_simulatore(_df, pozzo):
    """Addestra il simulatore GBR choke-produzione per il Well Optimizer.

    Filtra i giorni con choke disponibile, crea feature ingegneristiche,
    addestra un GradientBoostingRegressor e calcola i valori baseline
    dagli ultimi 30 giorni del dataset filtrato.

    Args:
        _df (pd.DataFrame): Dataset completo di produzione Volve.
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
    df_p = _df[_df['WELL_BORE_CODE'] == pozzo].copy().sort_values('DATEPRD').reset_index(drop=True)
    df_p = df_p[(df_p['BORE_OIL_VOL'] > 0) & df_p['AVG_CHOKE_SIZE_P'].notna()].copy().reset_index(drop=True)
    df_p['GOR']        = df_p['BORE_GAS_VOL'] / df_p['BORE_OIL_VOL'].replace(0, np.nan)
    df_p['WATERCUT']   = df_p['BORE_WAT_VOL'] / (df_p['BORE_OIL_VOL'] + df_p['BORE_WAT_VOL']).replace(0, np.nan)
    df_p['OIL_ROLL30'] = df_p['BORE_OIL_VOL'].rolling(30).mean()
    df_p['OIL_RATIO']  = df_p['BORE_OIL_VOL'] / df_p['OIL_ROLL30']
    df_p['CHOKE_BIN']  = pd.cut(df_p['AVG_CHOKE_SIZE_P'], bins=10, labels=False)
    choke_curve = df_p.groupby('CHOKE_BIN').agg(
        choke_mid=('AVG_CHOKE_SIZE_P', 'mean'),
        oil_mean=('BORE_OIL_VOL', 'mean')
    ).dropna().reset_index()
    feats = ['AVG_CHOKE_SIZE_P', 'WATERCUT', 'GOR', 'AVG_DOWNHOLE_PRESSURE', 'ON_STREAM_HRS']
    df_sim = df_p[feats + ['BORE_OIL_VOL']].dropna().copy()
    X, y = df_sim[feats].values, df_sim['BORE_OIL_VOL'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    sim = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    sim.fit(X_tr, y_tr)
    y_pr = sim.predict(X_te)
    mape = mean_absolute_percentage_error(y_te, y_pr) * 100
    r2   = 1 - np.sum((y_te - y_pr) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
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
    ratio = np.clip((q_c / q_b) if q_b > 0 else 1.0, 0.3, 2.0)
    return bl['olio'] * ratio
