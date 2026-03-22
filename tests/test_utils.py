"""
Test unitari per le funzioni core in utils/data.py e utils/models.py.

Strategia di import:
- streamlit viene mockato prima dell'import per evitare che i decoratori
  @st.cache_data / @st.cache_resource richiedano un contesto Streamlit attivo.
- plotly viene mockato perché importato da dashboard/app.py (non usato qui).
- simula() è importata direttamente da utils.models.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ── 1. Mock streamlit (deve avvenire prima dell'import di utils) ──────────────
def _cache_passthrough(func=None, **kw):
    """Rimpiazza st.cache_data / st.cache_resource con un no-op."""
    if callable(func):
        return func          # @st.cache_data  senza parentesi
    return lambda f: f       # @st.cache_data(...) con args → restituisce decorator

_st = MagicMock()
_st.cache_data     = _cache_passthrough
_st.cache_resource = _cache_passthrough
sys.modules['streamlit'] = _st

# ── 2. Mock plotly ────────────────────────────────────────────────────────────
sys.modules['plotly']               = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

# ── 3. Aggiungi dashboard/ al path (utils/ è dentro dashboard/) ──────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(ROOT, 'dashboard')
if DASHBOARD not in sys.path:
    sys.path.insert(0, DASHBOARD)

# ── 4. Import da utils ────────────────────────────────────────────────────────
from utils.data import get_prezzo_medio_brent, BBL_PER_SM3
from utils.models import arps_esponenziale, simula


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — arps_esponenziale: monotonia decrescente
# ═══════════════════════════════════════════════════════════════════════════════
def test_arps_esponenziale_monotona_decrescente():
    """arps_esponenziale deve produrre valori strettamente decrescenti."""
    t = np.linspace(0, 1000, 100)
    q = arps_esponenziale(t, qi=1000, Di=0.001)
    diffs = np.diff(q)
    assert np.all(diffs < 0), (
        f"La curva non è monotona: {np.sum(diffs >= 0)} punti non decrescenti trovati"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — arps_esponenziale: valore al tempo zero == qi
# ═══════════════════════════════════════════════════════════════════════════════
def test_arps_esponenziale_valore_iniziale():
    """arps_esponenziale(0, qi, Di) deve restituire esattamente qi."""
    qi = 1500.0
    q0 = arps_esponenziale(0, qi=qi, Di=0.002)
    assert q0 == pytest.approx(qi), f"Atteso {qi}, ottenuto {q0}"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — get_prezzo_medio_brent: fallback quando df_brent è None
# ═══════════════════════════════════════════════════════════════════════════════
def test_get_prezzo_medio_brent_fallback():
    """Deve restituire (80.0, False) quando df_brent è None."""
    df_prod = pd.DataFrame({
        'DATEPRD': pd.date_range('2010-01-01', periods=10, freq='D')
    })
    prezzo, trovato = get_prezzo_medio_brent(df_prod, df_brent=None)
    assert prezzo == pytest.approx(80.0)
    assert trovato is False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — get_prezzo_medio_brent: calcolo con dati reali
# ═══════════════════════════════════════════════════════════════════════════════
def test_get_prezzo_medio_brent_calcolo():
    """Con dati Brent validi deve restituire prezzo > 0 e trovato == True."""
    df_prod = pd.DataFrame({
        'DATEPRD': pd.date_range('2010-01-01', periods=365, freq='D')
    })
    date_brent = pd.date_range('2009-01-01', periods=1000, freq='D')
    df_brent   = pd.DataFrame(
        {'BRENT': np.linspace(60.0, 100.0, 1000)},
        index=date_brent
    )
    prezzo, trovato = get_prezzo_medio_brent(df_prod, df_brent)
    assert trovato is True
    assert prezzo > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — simula: deve restituire un float positivo
# ═══════════════════════════════════════════════════════════════════════════════
def test_simula_ritorna_float_positivo():
    """simula() con baseline e choke_curve validi deve restituire float > 0."""
    choke_curve = pd.DataFrame({
        'choke_mid': [10.0, 30.0, 50.0, 70.0, 100.0],
        'oil_mean':  [50.0, 100.0, 150.0, 180.0, 200.0],
    })
    bl = {'olio': 150.0, 'choke': 50.0, 'choke_curve': choke_curve}
    result = simula(60.0, bl)
    assert isinstance(result, float)
    assert result > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6 — BBL_PER_SM3: costante di conversione
# ═══════════════════════════════════════════════════════════════════════════════
def test_bbl_per_sm3_costante():
    """BBL_PER_SM3 deve valere esattamente 6.29."""
    assert BBL_PER_SM3 == pytest.approx(6.29)
