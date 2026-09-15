"""Statystyka opisowa i testy losowosci dla historii KENO.

Wszystkie funkcje przyjmuja macierz binarna M (n_losowan x 70), zwracana przez
db.to_matrix, uporzadkowana chronologicznie (wiersz 0 = najstarsze losowanie).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import POOL, DRAW_SIZE, P_SINGLE, NUM_MIN, WINDOWS


# --------------------------------------------------------------------------- #
# 4/5. Czestotliwosci, przerwy, z-score
# --------------------------------------------------------------------------- #
def frequency_table(M: np.ndarray) -> pd.DataFrame:
    """Dla kazdej liczby: czestotliwosc, przerwy (srednia/max/aktualna), z-score.

    z-score liczony wzgledem rozkladu dwumianowego Bin(n, 20/70) - przy uczciwym
    losowaniu |z| > ~3 dla pojedynczej liczby jest rzadkie (ale przy 70 testach
    trzeba korekty na wielokrotne testowanie - patrz chi_square_numbers).
    """
    n = M.shape[0]
    counts = M.sum(axis=0)
    exp = n * P_SINGLE
    sd = np.sqrt(n * P_SINGLE * (1 - P_SINGLE))
    rows = []
    for j in range(POOL):
        idx = np.flatnonzero(M[:, j])
        gaps = np.diff(idx) if len(idx) > 1 else np.array([])
        rows.append({
            "number": j + NUM_MIN,
            "count": int(counts[j]),
            "freq": counts[j] / n if n else np.nan,
            "expected": exp,
            "deviation": float(counts[j] - exp),
            "z_score": float((counts[j] - exp) / sd) if sd else np.nan,
            "mean_gap": float(gaps.mean()) if gaps.size else np.nan,
            "median_gap": float(np.median(gaps)) if gaps.size else np.nan,
            "max_gap": int(gaps.max()) if gaps.size else np.nan,
            "current_gap": int(n - 1 - idx[-1]) if idx.size else n,
        })
    df = pd.DataFrame(rows)
    # dwustronne p-value dla kazdej liczby + korekta Bonferroniego (70 testow)
    df["p_value"] = 2 * stats.norm.sf(df["z_score"].abs())
    df["p_bonferroni"] = (df["p_value"] * POOL).clip(upper=1.0)
    return df


def frequency_windows(M: np.ndarray, windows=WINDOWS) -> pd.DataFrame:
    """Czestotliwosci w oknach: cala historia + ostatnie N losowan."""
    out = {"number": np.arange(NUM_MIN, NUM_MIN + POOL)}
    out["count_all"] = M.sum(axis=0)
    out["z_all"] = frequency_table(M)["z_score"].values
    for w in windows:
        if M.shape[0] < w:
            continue
        sub = M[-w:]
        out[f"count_{w}"] = sub.sum(axis=0)
        out[f"z_{w}"] = frequency_table(sub)["z_score"].values
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 6. Pary, trojki, czworki
# --------------------------------------------------------------------------- #
def cooccurrence(M: np.ndarray) -> np.ndarray:
    """Macierz 70x70 licznikow wspolwystepowania (diagonala = czestosc liczby)."""
    return M.T.astype(np.int64) @ M.astype(np.int64)


def top_pairs(M: np.ndarray, top: int = 30) -> pd.DataFrame:
    """Najczestsze pary z z-score wzgledem oczekiwania dla losowania bez zwracania.

    P(oba) = (20/70)*(19/69) dla uczciwego losowania.
    """
    n = M.shape[0]
    C = cooccurrence(M)
    p = (DRAW_SIZE / POOL) * ((DRAW_SIZE - 1) / (POOL - 1))
    exp, sd = n * p, np.sqrt(n * p * (1 - p))
    rows = []
    for a, b in combinations(range(POOL), 2):
        c = int(C[a, b])
        rows.append((a + NUM_MIN, b + NUM_MIN, c, (c - exp) / sd))
    df = pd.DataFrame(rows, columns=["a", "b", "count", "z_score"])
    df["expected"] = exp
    df["p_value"] = 2 * stats.norm.sf(df["z_score"].abs())
    # liczba testow = C(70,2) = 2415
    df["p_bonferroni"] = (df["p_value"] * len(df)).clip(upper=1.0)
    return df.sort_values("count", ascending=False).head(top).reset_index(drop=True)


def top_ktuples(df_numbers: pd.Series, k: int = 3, top: int = 20,
                n_draws: int | None = None) -> pd.DataFrame:
    """Najczestsze k-elementowe kombinacje (k=3,4) + z-score.

    Uwaga wydajnosciowa: dla k=3 to C(20,3)=1140 kombinacji na losowanie,
    dla k=4 C(20,4)=4845 - liczymy zlicznikiem, bez macierzy.
    """
    cnt: Counter = Counter()
    n = 0
    for nums in df_numbers:
        n += 1
        for combo in combinations(sorted(nums), k):
            cnt[combo] += 1
    n_draws = n_draws or n
    # P(k konkretnych liczb razem) = iloczyn (20-i)/(70-i)
    p = 1.0
    for i in range(k):
        p *= (DRAW_SIZE - i) / (POOL - i)
    exp, sd = n_draws * p, np.sqrt(n_draws * p * (1 - p))
    rows = [{"combo": "-".join(map(str, c)), "count": v, "expected": exp,
             "z_score": (v - exp) / sd} for c, v in cnt.most_common(top)]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value"] = 2 * stats.norm.sf(out["z_score"].abs())
        # przyblizona liczba testow = liczba obserwowanych kombinacji
        out["p_bonferroni"] = (out["p_value"] * max(len(cnt), 1)).clip(upper=1.0)
    return out


# --------------------------------------------------------------------------- #
# 7. Zaleznosci czasowe
# --------------------------------------------------------------------------- #
def conditional_repeat(M: np.ndarray) -> pd.DataFrame:
    """Czy liczba czesciej wypada PO tym, jak wypadla w poprzednim losowaniu?

    Dla kazdej liczby: P(wypadnie | wypadla poprzednio) vs P(wypadnie | nie wypadla).
    Przy niezaleznosci obie wartosci ~ 20/70. Test: dwumianowy z-test roznicy proporcji.
    """
    rows = []
    for j in range(POOL):
        prev, cur = M[:-1, j], M[1:, j]
        n1, n0 = prev.sum(), (1 - prev).sum()
        if n1 == 0 or n0 == 0:
            continue
        p1 = cur[prev == 1].mean()
        p0 = cur[prev == 0].mean()
        pooled = cur.mean()
        se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
        z = (p1 - p0) / se if se else np.nan
        rows.append({"number": j + NUM_MIN, "p_after_hit": p1, "p_after_miss": p0,
                     "diff": p1 - p0, "z_score": z,
                     "p_value": 2 * stats.norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df["p_value"] * len(df)).clip(upper=1.0)
    return df


def transition_matrix(M: np.ndarray, top: int = 25) -> pd.DataFrame:
    """Czy po liczbie A czesciej pojawia sie liczba B w NASTEPNYM losowaniu?

    Zwraca najbardziej odstajace pary (A -> B) wg z-score.
    """
    prev, nxt = M[:-1].astype(np.int64), M[1:].astype(np.int64)
    joint = prev.T @ nxt                     # ile razy A(t) i B(t+1)
    nA = prev.sum(axis=0)                    # ile razy A wystapilo
    pB = nxt.mean(axis=0)                    # baza dla B
    exp = np.outer(nA, pB)
    sd = np.sqrt(np.outer(nA, pB * (1 - pB)))
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (joint - exp) / sd
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(Z), axis=None), Z.shape))[0][:top]
    rows = [{"from": int(a) + NUM_MIN, "to": int(b) + NUM_MIN,
             "count": int(joint[a, b]), "expected": float(exp[a, b]),
             "z_score": float(Z[a, b]),
             "p_value": float(2 * stats.norm.sf(abs(Z[a, b])))} for a, b in idx]
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df["p_value"] * POOL * POOL).clip(upper=1.0)
    return df


def autocorrelation(M: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    """Autokorelacja szeregu wskaznikow kazdej liczby, agregowana po liczbach.

    Zwraca dla kazdego lagu: srednia autokorelacje, max |ac| oraz p-value testu
    Ljung-Box-owego przyblizenia (statystyka z = ac*sqrt(n)).
    """
    n = M.shape[0]
    rows = []
    X = M.astype(float) - M.mean(axis=0, keepdims=True)
    denom = (X * X).sum(axis=0)
    for lag in range(1, max_lag + 1):
        num = (X[:-lag] * X[lag:]).sum(axis=0)
        ac = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        z = ac * np.sqrt(n - lag)
        rows.append({"lag": lag, "mean_ac": float(ac.mean()),
                     "max_abs_ac": float(np.abs(ac).max()),
                     "max_abs_z": float(np.abs(z).max()),
                     "p_value_min": float(2 * stats.norm.sf(np.abs(z).max())),
                     "p_bonferroni": float(min(1.0, 2 * stats.norm.sf(np.abs(z).max()) * POOL))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 8. Rozklady strukturalne
# --------------------------------------------------------------------------- #
def structure_stats(df_numbers: pd.Series) -> pd.DataFrame:
    """Per-losowanie: parzyste, wysokie, suma, liczby kolejne, dekady."""
    rows = []
    for nums in df_numbers:
        a = np.asarray(nums)
        dec = np.bincount((a - 1) // 10, minlength=7)
        row = {"even": int((a % 2 == 0).sum()), "odd": int((a % 2 == 1).sum()),
               "low": int((a <= 35).sum()), "high": int((a > 35).sum()),
               "sum": int(a.sum()), "consecutive": int((np.diff(np.sort(a)) == 1).sum())}
        row.update({f"dec_{i*10+1}_{i*10+10}": int(dec[i]) for i in range(7)})
        rows.append(row)
    return pd.DataFrame(rows)


def structure_summary(S: pd.DataFrame) -> pd.DataFrame:
    """Statystyki opisowe rozkladow + oczekiwania teoretyczne."""
    # E[even] = 20 * 35/70 = 10; E[low] = 10; E[sum] = 20*35.5 = 710
    exp = {"even": 10.0, "odd": 10.0, "low": 10.0, "high": 10.0, "sum": 710.0}
    for i in range(7):
        exp[f"dec_{i*10+1}_{i*10+10}"] = 20 * 10 / 70
    rows = []
    for col in S.columns:
        v = S[col].values
        e = exp.get(col, np.nan)
        se = v.std(ddof=1) / np.sqrt(len(v))
        rows.append({"metric": col, "mean": v.mean(), "std": v.std(ddof=1),
                     "min": v.min(), "max": v.max(), "expected": e,
                     "t_stat": (v.mean() - e) / se if e == e and se else np.nan,
                     "p_value": float(stats.ttest_1samp(v, e).pvalue) if e == e else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 9. Testy statystyczne
# --------------------------------------------------------------------------- #
def chi_square_numbers(M: np.ndarray) -> dict:
    """Chi-kwadrat jednorodnosci czestotliwosci 70 liczb (df = 69)."""
    counts = M.sum(axis=0)
    exp = np.full(POOL, counts.sum() / POOL)
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    df = POOL - 1
    return {"test": "chi2_uniformity_numbers", "chi2": chi2, "df": df,
            "p_value": float(stats.chi2.sf(chi2, df))}


def chi_square_independence_consecutive(M: np.ndarray) -> dict:
    """Test niezaleznosci kolejnych losowan: tabela 2x2 (liczba w t-1) x (w t),
    zagregowana po wszystkich liczbach (chi2 z poprawka Yatesa)."""
    prev, cur = M[:-1].ravel(), M[1:].ravel()
    tab = np.array([[int(((prev == 0) & (cur == 0)).sum()), int(((prev == 0) & (cur == 1)).sum())],
                    [int(((prev == 1) & (cur == 0)).sum()), int(((prev == 1) & (cur == 1)).sum())]])
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=True)
    return {"test": "chi2_independence_t_vs_t-1", "table": tab.tolist(),
            "chi2": float(chi2), "df": int(dof), "p_value": float(p)}


def runs_test(M: np.ndarray) -> pd.DataFrame:
    """Wald-Wolfowitz runs test dla szeregu 0/1 kazdej liczby."""
    rows = []
    for j in range(POOL):
        x = M[:, j]
        n1, n0 = int(x.sum()), int(len(x) - x.sum())
        if n1 < 2 or n0 < 2:
            continue
        runs = 1 + int((np.diff(x) != 0).sum())
        mu = 2 * n1 * n0 / (n1 + n0) + 1
        var = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        z = (runs - mu) / np.sqrt(var)
        rows.append({"number": j + NUM_MIN, "runs": runs, "expected_runs": mu,
                     "z_score": z, "p_value": float(2 * stats.norm.sf(abs(z)))})
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df["p_value"] * len(df)).clip(upper=1.0)
    return df


def entropy_analysis(M: np.ndarray) -> dict:
    """Entropia rozkladu liczb (Shannon) vs maksimum log2(70) oraz entropia
    warunkowa nastepnego losowania wzgledem poprzedniego (przyblizenie)."""
    p = M.sum(axis=0) / M.sum()
    H = float(-np.sum(p * np.log2(p, out=np.zeros_like(p), where=p > 0)))
    Hmax = float(np.log2(POOL))
    # entropia per-liczba (Bernoulli) - laczna dla niezaleznych wskaznikow
    q = M.mean(axis=0)
    lg = lambda v: np.log2(v, out=np.zeros_like(v), where=v > 0)
    Hb = float(-np.sum(q * lg(q) + (1 - q) * lg(1 - q)))
    return {"test": "entropy", "shannon_bits": H, "max_bits": Hmax,
            "efficiency": H / Hmax, "bernoulli_joint_bits": Hb,
            "bernoulli_max_bits": float(POOL * stats.entropy([P_SINGLE, 1 - P_SINGLE], base=2))}


def monte_carlo_uniformity(M: np.ndarray, n_sim: int = 2000, seed: int = 0,
                           n_draws_sim: int | None = None) -> dict:
    """Monte Carlo dla statystyki chi2 jednorodnosci.

    Kazda symulacja to CALA sztuczna historia n losowan po 20 z 70 (bez
    zwracania w obrebie losowania). p-value empiryczne = udzial symulacji z
    chi2 >= chi2 obserwowanego. Koszt jednej symulacji to O(n*70), dlatego
    domyslnie 2000 symulacji - milion symulacji Monte Carlo wykonywany jest na
    poziomie zakladow w module backtest (monte_carlo_bets).
    """
    rng = np.random.default_rng(seed)
    n = n_draws_sim or M.shape[0]
    obs = chi_square_numbers(M)["chi2"]
    exp = n * DRAW_SIZE / POOL
    ge = 0
    for _ in range(n_sim):
        picks = np.argpartition(rng.random((n, POOL)), DRAW_SIZE, axis=1)[:, :DRAW_SIZE]
        c = np.bincount(picks.ravel(), minlength=POOL)
        ge += int(((c - exp) ** 2 / exp).sum() >= obs)
    return {"test": "monte_carlo_chi2", "observed_chi2": obs, "n_sim": n_sim,
            "n_draws_per_sim": n, "p_value_empirical": ge / n_sim}


def stability_check(M: np.ndarray, n_blocks: int = 5) -> pd.DataFrame:
    """Czy anomalia jest stabilna w czasie? Dzielimy historie na bloki i
    porownujemy z-score kazdej liczby blok po bloku + korelacje miedzy blokami."""
    blocks = np.array_split(np.arange(M.shape[0]), n_blocks)
    zs = []
    for b in blocks:
        zs.append(frequency_table(M[b])["z_score"].values)
    Z = np.vstack(zs)
    out = pd.DataFrame(Z.T, columns=[f"block_{i+1}_z" for i in range(n_blocks)])
    out.insert(0, "number", np.arange(NUM_MIN, NUM_MIN + POOL))
    out["sign_consistency"] = (np.sign(Z).sum(axis=0) / n_blocks).__abs__()
    # korelacja z-score miedzy kolejnymi blokami: jesli anomalia realna, > 0
    corrs = [float(np.corrcoef(Z[i], Z[i + 1])[0, 1]) for i in range(n_blocks - 1)]
    out.attrs["block_to_block_corr"] = corrs
    out.attrs["mean_corr"] = float(np.mean(corrs)) if corrs else np.nan
    return out
