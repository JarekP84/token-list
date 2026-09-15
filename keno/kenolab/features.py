"""Inzynieria cech dla modeli nadzorowanych.

Dla kazdej pary (losowanie t, liczba j) budujemy wektor cech wyliczony
WYLACZNIE z losowan < t (brak wycieku informacji z przyszlosci - kluczowe dla
poprawnosci walk-forward backtestu). Etykieta y = 1, jesli liczba j wypadla w t.
"""
from __future__ import annotations

import numpy as np

from .config import POOL, DRAW_SIZE, P_SINGLE

# Okna do sredniej kroczacej czestotliwosci
ROLL_WINDOWS = (10, 25, 50, 100, 250, 500, 1000)
FEATURE_NAMES = (
    [f"freq_{w}" for w in ROLL_WINDOWS]
    + [f"z_{w}" for w in ROLL_WINDOWS]
    + ["gap", "gap_norm", "mean_gap", "gap_ratio",
       "hit_t1", "hit_t2", "hit_t3", "hits_last5",
       "cooc_prev", "cooc_prev2", "number_norm", "parity", "decade"]
)


def running_state(M: np.ndarray) -> dict:
    """Prekomputacja skumulowanych sum - pozwala liczyc cechy w O(1) na okno."""
    cs = np.vstack([np.zeros((1, POOL), dtype=np.int64), np.cumsum(M, axis=0)])
    return {"cumsum": cs}


def _window_count(cs: np.ndarray, t: int, w: int) -> np.ndarray:
    """Liczba trafien kazdej liczby w oknie [t-w, t)."""
    lo = max(0, t - w)
    return (cs[t] - cs[lo]).astype(float), t - lo


def features_at(M: np.ndarray, t: int, state: dict, C: np.ndarray | None = None) -> np.ndarray:
    """Macierz cech (70 x n_features) dla losowania o indeksie t.

    Uzywa tylko wierszy M[:t]. C - macierz wspolwystepowania policzona na M[:t].
    """
    cs = state["cumsum"]
    cols = []
    for w in ROLL_WINDOWS:
        cnt, eff = _window_count(cs, t, w)
        freq = cnt / eff if eff else np.full(POOL, P_SINGLE)
        cols.append(freq)
    for w in ROLL_WINDOWS:
        cnt, eff = _window_count(cs, t, w)
        sd = np.sqrt(eff * P_SINGLE * (1 - P_SINGLE)) if eff else 1.0
        cols.append((cnt - eff * P_SINGLE) / sd if sd else np.zeros(POOL))

    # --- przerwy ---
    gap = np.empty(POOL)
    hist = M[:t]
    if t == 0:
        gap[:] = 0
    else:
        last_hit = np.where(hist.any(axis=0),
                            t - 1 - np.argmax(hist[::-1], axis=0), t)
        gap = (t - 1 - last_hit).astype(float)
        gap[~hist.any(axis=0)] = t
    total = cs[t].astype(float)
    mean_gap = np.divide(t, np.maximum(total, 1), out=np.full(POOL, float(t)), where=total > 0)
    cols += [gap, gap / max(t, 1), mean_gap,
             np.divide(gap, np.maximum(mean_gap, 1e-9))]

    # --- ostatnie losowania ---
    for lag in (1, 2, 3):
        cols.append(M[t - lag].astype(float) if t >= lag else np.zeros(POOL))
    cols.append(M[max(0, t - 5):t].sum(axis=0).astype(float))

    # --- wspolwystepowanie z poprzednimi losowaniami ---
    for lag in (1, 2):
        if C is not None and t >= lag:
            prev = np.flatnonzero(M[t - lag])
            sc = C[:, prev].sum(axis=1).astype(float)
            denom = np.maximum(total, 1)
            cols.append(sc / denom)
        else:
            cols.append(np.zeros(POOL))

    # --- cechy statyczne liczby ---
    nums = np.arange(1, POOL + 1)
    cols += [nums / POOL, (nums % 2).astype(float), ((nums - 1) // 10).astype(float)]
    return np.column_stack(cols)


def build_dataset(M: np.ndarray, start: int, end: int, stride: int = 1,
                  use_cooc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Buduje (X, y) dla losowan z przedzialu [start, end).

    Wazne: cechy dla losowania t korzystaja wylacznie z M[:t].
    """
    state = running_state(M)
    Xs, ys = [], []
    # macierz wspolwystepowania aktualizowana inkrementalnie (O(400) na krok)
    C = (M[:start].T.astype(np.int64) @ M[:start].astype(np.int64)) if use_cooc else None
    for t in range(start, end):
        if t >= start + 1 and use_cooc:
            v = M[t - 1].astype(np.int64)
            C += np.outer(v, v)
        if (t - start) % stride == 0:
            Xs.append(features_at(M, t, state, C))
            ys.append(M[t].astype(np.int8))
    if not Xs:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int8)
    return np.vstack(Xs), np.concatenate(ys)
