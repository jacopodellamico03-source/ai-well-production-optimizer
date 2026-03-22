import os
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POZZI = ['NO 15/9-F-14 H', 'NO 15/9-F-12 H', 'NO 15/9-F-11 H']
POZZI_LABEL = {
    'NO 15/9-F-14 H': 'F-14 H',
    'NO 15/9-F-12 H': 'F-12 H',
    'NO 15/9-F-11 H': 'F-11 H',
}
BBL_PER_SM3 = 6.29


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
