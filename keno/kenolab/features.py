"""Inzynieria cech dla modeli nadzorowanych + inkrementalny stan historii.

Dla kazdej pary (losowanie t, liczba j) budujemy wektor cech wyliczony
WYLACZNIE z losowan < t (brak wycieku informacji z przyszlosci - kluczowe dla
poprawnosci walk-forward backtestu). Etykieta y = 1, jesli liczba j wypadla w t.

HistoryState trzyma sumy kumulacyjne, indeksy ostatnich trafien i (opcjonalnie)
macierz wspolwystepowania, dzieki czemu kolejny krok walk-forward kosztuje
O(70), a nie O(t*70).
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


class HistoryState:
    """Inkrementalny stan historii losowan.

    Niezmiennik: po sync(M[:t]) stan opisuje WYLACZNIE losowania o indeksach < t.
    Skrocenie historii (inny przebieg) powoduje pelny reset - nigdy nie mieszamy
    stanow z roznych serii danych.
    """

    def __init__(self, with_cooc: bool = False):
        self.with_cooc = with_cooc
        self.cs = np.zeros((1, POOL), dtype=np.int64)    # cs[i] = suma wierszy < i
        self.last_hit = np.full(POOL, -1, dtype=np.int64)
        self.n = 0
        self.C = np.zeros((POOL, POOL), dtype=np.int64) if with_cooc else None

    def sync(self, M: np.ndarray) -> "HistoryState":
        """Dociaga stan do dlugosci len(M)."""
        m = len(M)
        if m < self.n:
            self.__init__(with_cooc=self.with_cooc)
        if m == self.n:
            return self
        new = M[self.n:m].astype(np.int64)
        self.cs = np.vstack([self.cs, self.cs[-1] + np.cumsum(new, axis=0)])
        for off in range(new.shape[0]):
            idx = np.flatnonzero(new[off])
            self.last_hit[idx] = self.n + off
            if self.C is not None:
                self.C[np.ix_(idx, idx)] += 1
        self.n = m
        return self

    # --- odczyty ---------------------------------------------------------- #
    def window_count(self, t: int, w: int) -> tuple[np.ndarray, int]:
        """Liczba trafien kazdej liczby w oknie [t-w, t) oraz efektywna dlugosc okna."""
        lo = max(0, t - w)
        return (self.cs[t] - self.cs[lo]).astype(float), t - lo

    def total(self, t: int) -> np.ndarray:
        """Liczba trafien kazdej liczby w losowaniach < t."""
        return self.cs[t].astype(float)

    def gap(self, t: int) -> np.ndarray:
        """Aktualna przerwa: ile losowan minelo od ostatniego trafienia przed t."""
        g = np.where(self.last_hit >= 0, t - 1 - self.last_hit, t).astype(float)
        return np.maximum(g, 0.0)


def running_state(M: np.ndarray, with_cooc: bool = False) -> HistoryState:
    """Buduje stan od zera dla calej historii M."""
    return HistoryState(with_cooc=with_cooc).sync(M)


def features_at(M: np.ndarray, t: int, state: HistoryState,
                C: np.ndarray | None = None) -> np.ndarray:
    """Macierz cech (70 x n_features) dla losowania o indeksie t.

    Wymaga stanu zsynchronizowanego do dlugosci t; korzysta tylko z M[:t].
    """
    cols = []
    for w in ROLL_WINDOWS:
        cnt, eff = state.window_count(t, w)
        cols.append(cnt / eff if eff else np.full(POOL, P_SINGLE))
    for w in ROLL_WINDOWS:
        cnt, eff = state.window_count(t, w)
        sd = np.sqrt(eff * P_SINGLE * (1 - P_SINGLE)) if eff else 0.0
        cols.append((cnt - eff * P_SINGLE) / sd if sd else np.zeros(POOL))

    # --- przerwy ---
    gap = state.gap(t)
    total = state.total(t)
    mean_gap = np.divide(t, np.maximum(total, 1), out=np.full(POOL, float(t)),
                         where=total > 0)
    cols += [gap, gap / max(t, 1), mean_gap, gap / np.maximum(mean_gap, 1e-9)]

    # --- ostatnie losowania ---
    for lag in (1, 2, 3):
        cols.append(M[t - lag].astype(float) if t >= lag else np.zeros(POOL))
    cols.append(M[max(0, t - 5):t].sum(axis=0).astype(float))

    # --- wspolwystepowanie z poprzednimi losowaniami ---
    for lag in (1, 2):
        if C is not None and t >= lag:
            prev = np.flatnonzero(M[t - lag])
            cols.append(C[:, prev].sum(axis=1).astype(float) / np.maximum(total, 1))
        else:
            cols.append(np.zeros(POOL))

    # --- cechy statyczne liczby ---
    nums = np.arange(1, POOL + 1)
    cols += [nums / POOL, (nums % 2).astype(float), ((nums - 1) // 10).astype(float)]
    return np.column_stack(cols)


def build_dataset(M: np.ndarray, start: int, end: int, stride: int = 1,
                  use_cooc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Buduje (X, y) dla losowan z przedzialu [start, end).

    Stan jest synchronizowany krok po kroku, wiec cechy dla losowania t
    korzystaja wylacznie z M[:t] - nie da sie podejrzec przyszlosci.
    """
    st = HistoryState(with_cooc=use_cooc)
    Xs, ys = [], []
    for t in range(start, end):
        st.sync(M[:t])
        if (t - start) % stride == 0:
            Xs.append(features_at(M, t, st, st.C if use_cooc else None))
            ys.append(M[t].astype(np.int8))
    if not Xs:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int8)
    return np.vstack(Xs), np.concatenate(ys)
