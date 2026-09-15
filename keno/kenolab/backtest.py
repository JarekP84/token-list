"""Walk-forward backtest, benchmarki i statystyka istotnosci.

Zasada zelazna: predykcja dla losowania t powstaje tylko z danych < t.
Model nigdy nie widzi wyniku, ktory ocenia.

Metryki na zaklad o rozmiarze k (KENO: k = 1..10):
    hits            - ile z k wybranych liczb wypadlo (0..k)
    rozklad trafien - udzial 0,1,2,...,k trafien
    CI              - 95% przedzial ufnosci sredniej (normalny, na duzej probie)
    p-value         - dwustronny test wzgledem strategii losowej (Welch t-test
                      + test permutacyjny na sparowanych roznicach)
Dodatkowo log-loss / Brier / AUC dla pelnego wektora 70 prawdopodobienstw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import (POOL, DRAW_SIZE, P_SINGLE, BET_SIZES, PAYOUTS,
                     SPLIT_TRAIN, SPLIT_VALID, RANDOM_SEED)


# --------------------------------------------------------------------------- #
# Podzial chronologiczny
# --------------------------------------------------------------------------- #
def chronological_split(n: int, train=SPLIT_TRAIN, valid=SPLIT_VALID,
                        final_holdout: int = 0) -> dict:
    """Zwraca indeksy granic TRAIN / VALIDATION / TEST (+ opcjonalny HOLDOUT).

    HOLDOUT to ostatnie 'final_holdout' losowan wyciete z konca - uzywane tylko
    do finalnej weryfikacji metody, ktora wykazala przewage na TEST.
    """
    n_eff = n - final_holdout
    i_train = int(n_eff * train)
    i_valid = int(n_eff * (train + valid))
    return {"train": (0, i_train), "valid": (i_train, i_valid),
            "test": (i_valid, n_eff), "holdout": (n_eff, n)}


# --------------------------------------------------------------------------- #
# Rdzen walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(M: np.ndarray, model, start: int, end: int,
                 min_history: int = 500, refit_stride: int | None = None,
                 progress: bool = False) -> np.ndarray:
    """Zwraca macierz predykcji (end-start) x 70.

    Dla kazdego t z [start, end) model jest (co refit_stride krokow) trenowany
    na M[:t] i przewiduje rozklad dla losowania t.
    """
    start = max(start, min_history)
    preds = np.empty((max(end - start, 0), POOL))
    stride = refit_stride or getattr(model, "refit_every", 1)
    for i, t in enumerate(range(start, end)):
        if i % stride == 0:
            model.fit(M[:t])          # WYLACZNIE dane przeszle
        preds[i] = model.predict_proba(M[:t])
        if progress and i % 500 == 0:
            print(f"    {model.__class__.__name__}: {i}/{end-start}", flush=True)
    return preds


def random_predictions(n_rows: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Benchmark losowy: dla kazdego losowania losowa permutacja preferencji."""
    rng = np.random.default_rng(seed)
    return rng.random((n_rows, POOL))


# --------------------------------------------------------------------------- #
# Ocena zakladow
# --------------------------------------------------------------------------- #
def hits_for_bets(preds: np.ndarray, actual: np.ndarray,
                  bet_sizes=BET_SIZES) -> dict[int, np.ndarray]:
    """Dla kazdego rozmiaru zakladu k: wektor liczby trafien w kolejnych losowaniach.

    Wybor = k liczb o najwyzszym score w danym losowaniu.
    """
    out = {}
    order = np.argsort(-preds, axis=1, kind="stable")
    for k in bet_sizes:
        picks = order[:, :k]
        out[k] = np.take_along_axis(actual, picks, axis=1).sum(axis=1)
    return out


def expected_hits_random(k: int) -> float:
    """Oczekiwana liczba trafien dla losowego zakladu k liczb = k*20/70."""
    return k * P_SINGLE


def hypergeom_distribution(k: int) -> np.ndarray:
    """Teoretyczny rozklad trafien dla losowego zakladu k liczb (hipergeometryczny)."""
    return stats.hypergeom.pmf(np.arange(k + 1), POOL, DRAW_SIZE, k)


def payout_ev(k: int, hits: np.ndarray) -> float:
    """Sredni zwrot na 1 zl stawki wg tabeli wyplat z config (orientacyjnie)."""
    table = PAYOUTS.get(k, {})
    mult = np.array([table.get(h, 0) for h in range(k + 1)], dtype=float)
    return float(mult[hits].mean())


def summarize(hits: np.ndarray, k: int, baseline: np.ndarray | None = None,
              n_perm: int = 20000, seed: int = RANDOM_SEED) -> dict:
    """Pelna statystyka jednej strategii dla zakladu k liczb."""
    n = len(hits)
    mean, sd = float(hits.mean()), float(hits.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else np.nan
    dist = np.bincount(hits, minlength=k + 1)[:k + 1] / max(n, 1)
    res = {
        "bet_size": k, "n_trials": n,
        "mean_hits": mean, "median_hits": float(np.median(hits)),
        "std_hits": sd, "max_hits": int(hits.max()) if n else 0,
        "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
        "expected_random": expected_hits_random(k),
        "edge_vs_theory": mean - expected_hits_random(k),
        "ev_payout_per_1zl": payout_ev(k, hits),
    }
    for h in range(k + 1):
        res[f"p_hits_{h}"] = float(dist[h])
        res[f"theory_hits_{h}"] = float(hypergeom_distribution(k)[h])
    # test wzgledem teoretycznej sredniej losowej (jednoprobkowy t-test)
    if n > 1:
        res["p_value_vs_theory"] = float(stats.ttest_1samp(
            hits, expected_hits_random(k)).pvalue)
    if baseline is not None and n > 1:
        d = hits.astype(float) - baseline.astype(float)
        res["mean_diff_vs_random"] = float(d.mean())
        res["p_value_vs_random_ttest"] = float(stats.ttest_rel(hits, baseline).pvalue)
        # test permutacyjny znakow (odporny na nienormalnosc)
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        null = (signs * d).mean(axis=1)
        res["p_value_vs_random_perm"] = float(
            (np.abs(null) >= abs(d.mean())).mean())
    return res


def evaluate_probabilistic(preds: np.ndarray, actual: np.ndarray) -> dict:
    """Jakosc kalibracji pelnego wektora 70 prawdopodobienstw."""
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    p = np.clip(preds.ravel(), 1e-6, 1 - 1e-6)
    y = actual.ravel()
    base = np.full_like(p, P_SINGLE)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "log_loss": float(log_loss(y, p)),
        "log_loss_baseline": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, base)),
    }


# --------------------------------------------------------------------------- #
# Monte Carlo na poziomie zakladow (>= 1 000 000 symulacji)
# --------------------------------------------------------------------------- #
def monte_carlo_bets(n_sim: int = 1_000_000, bet_sizes=BET_SIZES,
                     seed: int = RANDOM_SEED, chunk: int = 50_000) -> pd.DataFrame:
    """Symuluje n_sim uczciwych losowan i ocenia staly zaklad k liczb.

    Sluzy jako rozklad odniesienia (null distribution) dla wynikow backtestu:
    kazda strategia MUSI byc porownana z tym rozkladem.
    """
    rng = np.random.default_rng(seed)
    acc = {k: np.zeros(k + 1, dtype=np.int64) for k in bet_sizes}
    sums = {k: 0.0 for k in bet_sizes}
    sq = {k: 0.0 for k in bet_sizes}
    maxes = {k: 0 for k in bet_sizes}
    done = 0
    kmax = max(bet_sizes)
    while done < n_sim:
        b = min(chunk, n_sim - done)
        # kazdy wiersz: 20 wylosowanych liczb (indeksy 0..69) bez zwracania
        draws = np.argpartition(rng.random((b, POOL)), DRAW_SIZE, axis=1)[:, :DRAW_SIZE]
        mask = np.zeros((b, POOL), dtype=np.int8)
        np.put_along_axis(mask, draws, 1, axis=1)
        # zaklad = liczby 0..k-1 (bez utraty ogolnosci - losowanie jest symetryczne)
        cum = np.cumsum(mask[:, :kmax], axis=1)
        for k in bet_sizes:
            h = cum[:, k - 1]
            acc[k] += np.bincount(h, minlength=k + 1)[:k + 1]
            sums[k] += float(h.sum())
            sq[k] += float((h.astype(np.int64) ** 2).sum())
            maxes[k] = max(maxes[k], int(h.max()))
        done += b
    rows = []
    for k in bet_sizes:
        n = n_sim
        mean = sums[k] / n
        var = max(sq[k] / n - mean ** 2, 0.0)
        row = {"bet_size": k, "n_sim": n, "mean_hits": mean,
               "std_hits": np.sqrt(var), "max_hits": maxes[k],
               "expected_theory": expected_hits_random(k)}
        dist = acc[k] / n
        for h in range(k + 1):
            row[f"p_hits_{h}"] = float(dist[h])
        row["ev_payout_per_1zl"] = float(
            sum(PAYOUTS.get(k, {}).get(h, 0) * dist[h] for h in range(k + 1)))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pelny backtest wielu modeli
# --------------------------------------------------------------------------- #
def run_backtest(M: np.ndarray, models: dict, split: dict, segment: str = "test",
                 bet_sizes=BET_SIZES, min_history: int = 500,
                 progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Backtest wszystkich modeli na wskazanym segmencie (train/valid/test/holdout).

    Zwraca (tabela_wynikow, slownik_predykcji).
    """
    lo, hi = split[segment]
    lo = max(lo, min_history)
    actual = M[lo:hi]
    rand_preds = random_predictions(hi - lo)
    rand_hits = hits_for_bets(rand_preds, actual, bet_sizes)

    rows, preds_store = [], {}
    for name, model in models.items():
        if progress:
            print(f"  [{segment}] {name} ...", flush=True)
        preds = walk_forward(M, model, lo, hi, min_history=min_history)
        preds_store[name] = preds
        hits = hits_for_bets(preds, actual, bet_sizes)
        prob = evaluate_probabilistic(preds, actual)
        for k in bet_sizes:
            r = summarize(hits[k], k, baseline=rand_hits[k])
            r.update({"model": name, "segment": segment})
            r.update({f"prob_{key}": v for key, v in prob.items()})
            rows.append(r)
    # benchmark losowy tez raportujemy jako "model"
    for k in bet_sizes:
        r = summarize(rand_hits[k], k, baseline=rand_hits[k])
        r.update({"model": "bench_random", "segment": segment})
        rows.append(r)
    df = pd.DataFrame(rows)
    front = ["model", "segment", "bet_size", "n_trials", "mean_hits", "median_hits",
             "max_hits", "ci95_low", "ci95_high", "expected_random",
             "edge_vs_theory", "p_value_vs_theory", "p_value_vs_random_ttest",
             "p_value_vs_random_perm", "ev_payout_per_1zl"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols], preds_store


def significance_verdict(df: pd.DataFrame, alpha: float = 0.05,
                         segment: str = "test") -> dict:
    """Czy ktorykolwiek model ma statystycznie istotna przewage na segmencie?

    Wymagamy: dodatniej przewagi ORAZ p < alpha po korekcie Bonferroniego na
    liczbe porownan (modele x rozmiary zakladow).
    """
    sub = df[(df["segment"] == segment) & (df["model"] != "bench_random")].copy()
    if sub.empty:
        return {"has_edge": False, "winners": [], "n_comparisons": 0}
    n_cmp = len(sub)
    pcol = "p_value_vs_random_perm" if "p_value_vs_random_perm" in sub else "p_value_vs_theory"
    sub["p_adj"] = (sub[pcol] * n_cmp).clip(upper=1.0)
    win = sub[(sub["mean_diff_vs_random"] > 0) & (sub["p_adj"] < alpha)] \
        if "mean_diff_vs_random" in sub else sub[(sub["edge_vs_theory"] > 0) & (sub["p_adj"] < alpha)]
    return {
        "has_edge": bool(len(win)),
        "n_comparisons": n_cmp,
        "alpha": alpha,
        "winners": win.sort_values("p_adj")[
            ["model", "bet_size", "mean_hits", "expected_random", "p_adj"]
        ].to_dict("records"),
        "best_p_adj": float(sub["p_adj"].min()),
    }
