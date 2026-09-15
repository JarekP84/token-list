"""Predykcja na NASTEPNE losowanie: ranking 70 liczb + przykladowe zestawy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POOL, DRAW_SIZE, P_SINGLE, NUM_MIN, RANDOM_SEED
from .models import default_models


def rank_numbers(M: np.ndarray, models: dict | None = None,
                 weights: dict | None = None) -> pd.DataFrame:
    """Ranking liczb na nastepne losowanie.

    Zwraca DataFrame: number, score kazdego modelu, ensemble_score,
    probability (score przeskalowany tak, by sumowal sie do 20), rank.
    Ensemble = srednia (wazona) z_score'ow modeli - odporna na rozne skale.
    """
    models = models or default_models(fast=True)
    scores = {}
    for name, m in models.items():
        m.fit(M)
        p = np.asarray(m.predict_proba(M), dtype=float)
        scores[name] = p
    df = pd.DataFrame(scores)
    df.insert(0, "number", np.arange(NUM_MIN, NUM_MIN + POOL))
    # standaryzacja per model, zeby zaden nie zdominowal ensemble skala
    zs = {}
    for name in scores:
        v = df[name].values
        sd = v.std(ddof=0)
        zs[name] = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
    Z = np.column_stack([zs[n] for n in scores])
    w = np.array([(weights or {}).get(n, 1.0) for n in scores], dtype=float)
    df["ensemble_z"] = Z @ w / w.sum()
    # przeskalowanie na "prawdopodobienstwo": baza 20/70 + niewielka korekta
    raw = np.clip(P_SINGLE * (1 + 0.02 * df["ensemble_z"].values), 1e-6, 1)
    df["probability"] = raw / raw.sum() * DRAW_SIZE
    df = df.sort_values("ensemble_z", ascending=False).reset_index(drop=True)
    df.insert(1, "rank", np.arange(1, len(df) + 1))
    return df


def top_lists(ranking: pd.DataFrame, sizes=(10, 15, 20)) -> dict[int, pd.DataFrame]:
    """TOP 10 / 15 / 20 wraz ze score i prawdopodobienstwem."""
    return {k: ranking.head(k)[["rank", "number", "ensemble_z", "probability"]]
            for k in sizes}


def sample_sets(ranking: pd.DataFrame, n_sets: int = 10, bet_size: int = 10,
                top_pool: int = 25, seed: int = RANDOM_SEED) -> list[list[int]]:
    """Generuje n_sets przykladowych zestawow testowych z puli TOP 'top_pool'.

    UWAGA: to zestawy DO TESTOW pipeline'u, nie rekomendacje ani gwarancja
    wygranej. Przy uczciwym losowaniu kazdy zestaw ma identyczna szanse.
    """
    rng = np.random.default_rng(seed)
    pool = ranking.head(top_pool)["number"].values
    w = ranking.head(top_pool)["probability"].values
    w = w / w.sum()
    sets, seen = [], set()
    guard = 0
    while len(sets) < n_sets and guard < 10000:
        guard += 1
        pick = tuple(sorted(rng.choice(pool, size=bet_size, replace=False, p=w).tolist()))
        if pick in seen:
            continue
        seen.add(pick)
        sets.append(list(pick))
    return sets


def score_prediction(picked: list[int], actual: list[int]) -> dict:
    """Ocena pojedynczej predykcji po ogloszeniu prawdziwego wyniku."""
    hits = sorted(set(picked) & set(actual))
    k = len(picked)
    return {"bet_size": k, "hits": len(hits), "hit_numbers": hits,
            "expected_random": k * P_SINGLE,
            "diff_vs_expected": len(hits) - k * P_SINGLE}
