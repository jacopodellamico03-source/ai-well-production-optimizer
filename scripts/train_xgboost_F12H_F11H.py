"""
Train XGBoost per NO 15/9-F-12 H e NO 15/9-F-11 H
Stesso approccio del notebook 02_Decline_Curve_Analysis.ipynb (F-14 H)
"""

import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── CARICAMENTO DATI ──────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'Volve production data.xlsx')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("Caricamento dataset...")
df = pd.read_excel(DATA_PATH)
print(f"Righe totali: {len(df)}")
print(f"Pozzi disponibili: {df['WELL_BORE_CODE'].unique()}\n")

# ── FEATURE E TARGET ──────────────────────────────────────────────────────────
FEATURES = ['DAYS', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'AVG_DOWNHOLE_PRESSURE',
            'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS', 'GOR', 'WATERCUT',
            'OIL_ROLL7', 'OIL_ROLL30', 'OIL_CUMSUM', 'OIL_LAG1', 'OIL_LAG7']
TARGET = 'BORE_OIL_VOL'

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# ── FUNZIONI ──────────────────────────────────────────────────────────────────

def prepara_dati(df, well_name):
    """Filtra, ordina e crea feature ingegneristiche per un pozzo."""
    df_pozzo = df[df['WELL_BORE_CODE'] == well_name].copy()
    df_pozzo = df_pozzo.sort_values('DATEPRD').reset_index(drop=True)

    # Solo giorni in produzione
    df_prod = df_pozzo[df_pozzo['BORE_OIL_VOL'] > 0].copy().reset_index(drop=True)
    df_prod['DAYS'] = (df_prod['DATEPRD'] - df_prod['DATEPRD'].min()).dt.days

    # Selezione colonne base
    df_ml = df_prod[['DATEPRD', 'DAYS', 'BORE_OIL_VOL',
                     'BORE_GAS_VOL', 'BORE_WAT_VOL',
                     'AVG_DOWNHOLE_PRESSURE', 'AVG_CHOKE_SIZE_P',
                     'ON_STREAM_HRS']].copy()

    # Interpolazione valori mancanti
    df_ml = df_ml.interpolate(method='linear').bfill()

    # Feature ingegneristiche
    df_ml['GOR']        = df_ml['BORE_GAS_VOL'] / df_ml['BORE_OIL_VOL'].replace(0, np.nan)
    df_ml['WATERCUT']   = df_ml['BORE_WAT_VOL'] / (df_ml['BORE_OIL_VOL'] + df_ml['BORE_WAT_VOL']).replace(0, np.nan)
    df_ml['OIL_ROLL7']  = df_ml['BORE_OIL_VOL'].rolling(7).mean()
    df_ml['OIL_ROLL30'] = df_ml['BORE_OIL_VOL'].rolling(30).mean()
    df_ml['OIL_CUMSUM'] = df_ml['BORE_OIL_VOL'].cumsum()
    df_ml['OIL_LAG1']   = df_ml['BORE_OIL_VOL'].shift(1)
    df_ml['OIL_LAG7']   = df_ml['BORE_OIL_VOL'].shift(7)

    df_ml = df_ml.dropna().reset_index(drop=True)

    print(f"  Giorni in produzione (dopo preprocessing): {len(df_ml)}")
    print(f"  Dal: {df_ml['DATEPRD'].min().date()} - Al: {df_ml['DATEPRD'].max().date()}")
    return df_ml


def metriche(nome, y_reale, y_pred):
    rmse = np.sqrt(mean_squared_error(y_reale, y_pred))
    mae  = mean_absolute_error(y_reale, y_pred)
    mape = np.mean(np.abs((np.array(y_reale) - np.array(y_pred)) / np.array(y_reale))) * 100
    r2   = 1 - np.sum((np.array(y_reale) - np.array(y_pred))**2) / \
               np.sum((np.array(y_reale) - np.array(y_reale).mean())**2)
    print(f"  {nome:6} | RMSE: {rmse:7.1f} | MAE: {mae:7.1f} | MAPE: {mape:5.1f}% | R²: {r2:.3f}")
    return mape, r2


def train_well(df, well_name, model_suffix):
    """Pipeline completa: preprocessing → training → salvataggio → metriche."""
    print(f"\n{'='*60}")
    print(f"POZZO: {well_name}")
    print('='*60)

    df_ml = prepara_dati(df, well_name)

    # Split temporale 80/20
    split = int(len(df_ml) * 0.8)
    train = df_ml.iloc[:split]
    test  = df_ml.iloc[split:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    print(f"\n  Train: {len(train)} giorni "
          f"({train['DATEPRD'].min().date()} - {train['DATEPRD'].max().date()})")
    print(f"  Test:  {len(test)} giorni "
          f"({test['DATEPRD'].min().date()} - {test['DATEPRD'].max().date()})")

    # Scaler (fit su train)
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Training XGBoost (non richiede scaling, ma salviamo lo scaler per uso futuro)
    print("\n  Training XGBoost...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print("  Training completato!")

    # Previsioni
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Metriche
    print(f"\n  {'Set':6} | {'RMSE':>7} | {'MAE':>7} | {'MAPE':>7} | R²")
    print(f"  {'-'*50}")
    metriche("Train", y_train, y_pred_train)
    mape_test, r2_test = metriche("Test",  y_test,  y_pred_test)

    # Salvataggio
    model_path  = os.path.join(MODELS_DIR, f'xgboost_{model_suffix}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{model_suffix}.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n  Salvato: models/xgboost_{model_suffix}.pkl")
    print(f"  Salvato: models/scaler_{model_suffix}.pkl")

    return mape_test, r2_test


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = {}

    mape, r2 = train_well(df, 'NO 15/9-F-12 H', 'F12H')
    results['F-12 H'] = (mape, r2)

    mape, r2 = train_well(df, 'NO 15/9-F-11 H', 'F11H')
    results['F-11 H'] = (mape, r2)

    # Riepilogo finale
    print(f"\n{'='*60}")
    print("RIEPILOGO METRICHE TEST SET")
    print('='*60)
    print(f"  {'Pozzo':12} | {'MAPE':>8} | {'R²':>6}")
    print(f"  {'-'*35}")
    for well, (mape, r2) in results.items():
        print(f"  {well:12} | {mape:7.1f}% | {r2:.3f}")
    print()
