"""Orkiestracja: pobranie danych -> analiza -> backtest -> predykcja -> raport."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as an
from . import backtest as bt
from . import db as store
from . import fetch as fx
from . import predict as pd_mod
from . import tracking as tr
from .config import (REPORT_DIR, WINDOWS, BET_SIZES, FINAL_HOLDOUT,
                     MIN_MONTE_CARLO, PAYOUTS_VERIFIED, DB_PATH, CSV_PATH)
from .models import default_models, benchmarks


# --------------------------------------------------------------------------- #
# 1-3. Pobranie, zapis, walidacja
# --------------------------------------------------------------------------- #
def update_data(prefer: str = "auto", allow_synthetic: bool = False,
                n_synthetic: int = 30000, path: Path = DB_PATH) -> dict:
    """Pobiera wyniki, zapisuje do SQLite + CSV, deduplikuje i waliduje."""
    draws, source = fx.fetch_all(prefer=prefer, allow_synthetic=allow_synthetic,
                                 n_synthetic=n_synthetic)
    added = store.upsert(draws, path)
    df = store.load_frame(path)
    store.export_csv(df, CSV_PATH)
    report = store.validate(df)
    store.set_meta("last_update", datetime.now(timezone.utc).isoformat(timespec="seconds"), path)
    store.set_meta("source", source, path)
    n_settled = tr.settle(path)                      # rozlicz stare predykcje
    return {"source": source, "fetched": len(draws), "added": added,
            "total": len(df), "validation": report, "settled_predictions": n_settled,
            "csv": str(CSV_PATH), "db": str(path)}


# --------------------------------------------------------------------------- #
# 4-10. Analiza pelna
# --------------------------------------------------------------------------- #
def run_analysis(df: pd.DataFrame, mc_sims: int = 2000) -> dict:
    """Wykonuje caly blok analityczny (punkty 4-10 specyfikacji)."""
    M = store.to_matrix(df)
    out = {
        "n_draws": int(len(df)),
        "date_range": [df["date"].min(), df["date"].max()],
        "frequency": an.frequency_table(M),
        "frequency_windows": an.frequency_windows(M, WINDOWS),
        "top_pairs": an.top_pairs(M, top=30),
        "top_triples": an.top_ktuples(df["numbers"], 3, 20, len(df)),
        "top_quads": an.top_ktuples(df["numbers"], 4, 20, len(df)),
        "conditional_repeat": an.conditional_repeat(M),
        "transitions": an.transition_matrix(M, 25),
        "autocorrelation": an.autocorrelation(M, 20),
        "structure": an.structure_summary(an.structure_stats(df["numbers"])),
        "tests": {
            "chi2_numbers": an.chi_square_numbers(M),
            "chi2_independence": an.chi_square_independence_consecutive(M),
            "entropy": an.entropy_analysis(M),
            "monte_carlo_chi2": an.monte_carlo_uniformity(M, n_sim=mc_sims),
        },
        "runs_test": an.runs_test(M),
        "stability": an.stability_check(M, 5),
    }
    out["stability_corr"] = out["stability"].attrs.get("block_to_block_corr")
    return out


# --------------------------------------------------------------------------- #
# Backtest walk-forward + Monte Carlo
# --------------------------------------------------------------------------- #
def run_full_backtest(df: pd.DataFrame, fast: bool = True,
                      mc_sims: int = MIN_MONTE_CARLO,
                      final_holdout: int = FINAL_HOLDOUT,
                      min_history: int = 1000,
                      segments=("valid", "test"),
                      max_eval: int | None = None) -> dict:
    """Walk-forward backtest modeli i benchmarkow na VALIDATION i TEST.

    max_eval - opcjonalne ograniczenie dlugosci segmentu (przyspiesza run;
    liczba prob = dlugosc segmentu x 10 rozmiarow zakladu).
    """
    M = store.to_matrix(df)
    n = len(M)
    holdout = final_holdout if n > final_holdout * 2 else 0
    split = bt.chronological_split(n, final_holdout=holdout)
    if max_eval:
        for seg in ("valid", "test"):
            lo, hi = split[seg]
            split[seg] = (max(lo, hi - max_eval), hi)

    models = {**default_models(fast=fast), **{k: v for k, v in benchmarks().items()
                                              if k != "bench_random"}}
    results, preds = {}, {}
    for seg in segments:
        r, p = bt.run_backtest(M, models, split, segment=seg,
                               bet_sizes=BET_SIZES, min_history=min_history)
        results[seg] = r
        preds[seg] = p

    verdict = bt.significance_verdict(results["test"], segment="test") \
        if "test" in results else {"has_edge": False, "winners": []}

    # Weryfikacja na swiezym hold-oucie tylko jesli byla przewaga na TEST
    holdout_res = None
    if verdict.get("has_edge") and holdout:
        winners = sorted({w["model"] for w in verdict["winners"]})
        sel = {k: v for k, v in models.items() if k in winners}
        holdout_res, _ = bt.run_backtest(M, sel, split, segment="holdout",
                                         bet_sizes=BET_SIZES, min_history=min_history)

    mc = bt.monte_carlo_bets(n_sim=mc_sims)
    n_trials = int(sum(len(r) and (r["n_trials"].iloc[0] * len(BET_SIZES)) for r in results.values()))
    return {"split": split, "results": results, "verdict": verdict,
            "holdout": holdout_res, "monte_carlo": mc,
            "n_backtest_trials": n_trials, "payouts_verified": PAYOUTS_VERIFIED}


# --------------------------------------------------------------------------- #
# Predykcja nastepnego losowania
# --------------------------------------------------------------------------- #
def run_prediction(df: pd.DataFrame, n_sets: int = 10, bet_size: int = 10,
                   save: bool = True, path: Path = DB_PATH) -> dict:
    """Ranking 70 liczb na nastepne losowanie + TOP10/15/20 + przykladowe zestawy."""
    M = store.to_matrix(df)
    ranking = pd_mod.rank_numbers(M)
    tops = pd_mod.top_lists(ranking)
    sets = pd_mod.sample_sets(ranking, n_sets=n_sets, bet_size=bet_size)
    pid = tr.save_prediction(ranking, int(df["draw_id"].iloc[-1]), path=path) if save else None
    return {"ranking": ranking, "top": tops, "sample_sets": sets,
            "prediction_id": pid, "after_draw_id": int(df["draw_id"].iloc[-1])}


# --------------------------------------------------------------------------- #
# Raport
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", max_rows: int = 30) -> str:
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(analysis: dict, backtest: dict | None, prediction: dict | None,
                 meta: dict, out_dir: Path = REPORT_DIR) -> Path:
    """Generuje raport Markdown + zapisuje tabele CSV do reports/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # zrzuty CSV
    for name, obj in analysis.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out_dir / f"analysis_{name}.csv", index=False)
    if backtest:
        for seg, r in backtest["results"].items():
            r.to_csv(out_dir / f"backtest_{seg}.csv", index=False)
        backtest["monte_carlo"].to_csv(out_dir / "monte_carlo_bets.csv", index=False)
        if backtest.get("holdout") is not None:
            backtest["holdout"].to_csv(out_dir / "backtest_holdout.csv", index=False)
    if prediction:
        prediction["ranking"].to_csv(out_dir / "prediction_ranking.csv", index=False)

    L = ["# Raport KenoLab", "",
         f"- Wygenerowano: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         f"- Zrodlo danych: **{meta.get('source')}**",
         f"- Liczba losowan: **{analysis['n_draws']}**, zakres dat: {analysis['date_range'][0]} .. {analysis['date_range'][1]}",
         ""]
    if meta.get("source") == "synthetic":
        L += ["> **OSTRZEZENIE:** raport policzony na danych SYNTETYCZNYCH "
              "(uczciwie losowych), nie na prawdziwych wynikach KENO. "
              "Sluzy wylacznie do walidacji pipeline'u.", ""]

    L += ["## 1. Walidacja danych", "",
          f"```json\n{json.dumps({k: v for k, v in meta.get('validation', {}).items() if k in ('ok','n_draws','date_range','duplicate_draw_ids','gaps_in_draw_id')}, default=str, indent=2)}\n```", ""]

    L += ["## 2. Czestotliwosci i przerwy (TOP 15 wg |z-score|)", "",
          _md_table(analysis["frequency"].reindex(
              analysis["frequency"]["z_score"].abs().sort_values(ascending=False).index)
              [["number", "count", "freq", "expected", "deviation", "z_score",
                "mean_gap", "max_gap", "current_gap", "p_value", "p_bonferroni"]], max_rows=15), "",
          "### Czestotliwosci w oknach (50/100/250/500/1000/5000/10000)", "",
          _md_table(analysis["frequency_windows"], max_rows=70), ""]

    L += ["## 3. Pary / trojki / czworki", "",
          "### Najczestsze pary", "", _md_table(analysis["top_pairs"], max_rows=15), "",
          "### Najczestsze trojki", "", _md_table(analysis["top_triples"], max_rows=10), "",
          "### Najczestsze czworki", "", _md_table(analysis["top_quads"], max_rows=10), ""]

    L += ["## 4. Zaleznosci czasowe", "",
          "### Powtorzenia (liczba po wlasnym wystapieniu) - TOP 10 wg |z|", "",
          _md_table(analysis["conditional_repeat"].reindex(
              analysis["conditional_repeat"]["z_score"].abs().sort_values(ascending=False).index),
              max_rows=10), "",
          "### Przejscia A(t) -> B(t+1) - TOP 10", "",
          _md_table(analysis["transitions"], max_rows=10), "",
          "### Autokorelacja (lag 1-20)", "",
          _md_table(analysis["autocorrelation"], max_rows=20), ""]

    L += ["## 5. Rozklady strukturalne", "",
          _md_table(analysis["structure"], max_rows=20), ""]

    t = analysis["tests"]
    L += ["## 6. Testy statystyczne", "",
          f"```json\n{json.dumps(t, indent=2, default=str)}\n```", "",
          "### Runs test - TOP 10 wg |z|", "",
          _md_table(analysis["runs_test"].reindex(
              analysis["runs_test"]["z_score"].abs().sort_values(ascending=False).index),
              max_rows=10), "",
          "## 7. Stabilnosc anomalii w czasie", "",
          f"Korelacja z-score miedzy kolejnymi blokami historii: "
          f"{analysis.get('stability_corr')}", "",
          "(Realna, trwala anomalia dawalaby wysoka dodatnia korelacje miedzy blokami.)", ""]

    if backtest:
        L += ["## 8. Walk-forward backtest", "",
              f"Podzial chronologiczny: {backtest['split']}", "",
              f"Liczba prob historycznych (losowania x rozmiary zakladow): "
              f"**{backtest['n_backtest_trials']}**", "",
              f"Symulacje Monte Carlo: **{int(backtest['monte_carlo']['n_sim'].iloc[0]):,}**", ""]
        if not backtest["payouts_verified"]:
            L += ["> Kolumny `ev_payout_per_1zl` sa ORIENTACYJNE - tabela wyplat w "
                  "`config.PAYOUTS` wymaga weryfikacji na lotto.pl "
                  "(`PAYOUTS_VERIFIED = False`). Nie wyciagaj z nich wnioskow.", ""]
        for seg, r in backtest["results"].items():
            cols = ["model", "bet_size", "n_trials", "mean_hits", "median_hits",
                    "max_hits", "ci95_low", "ci95_high", "expected_random",
                    "edge_vs_theory", "p_value_vs_random_perm"]
            L += [f"### Segment {seg.upper()} (wszystkie rozmiary zakladu 1-10)", "",
                  _md_table(r[[c for c in cols if c in r.columns]], max_rows=200), ""]
        L += ["### Monte Carlo - rozklad odniesienia (uczciwe losowanie)", "",
              _md_table(backtest["monte_carlo"], max_rows=10), "",
              "### Werdykt istotnosci (segment TEST, korekta Bonferroniego)", "",
              f"```json\n{json.dumps(backtest['verdict'], indent=2, default=str)}\n```", ""]
        if backtest["verdict"].get("has_edge"):
            L += ["**Znaleziono kandydata na przewage - weryfikacja na swiezym hold-oucie:**", ""]
            if backtest.get("holdout") is not None:
                L += [_md_table(backtest["holdout"][["model", "bet_size", "mean_hits",
                                                     "expected_random", "p_value_vs_random_perm"]],
                                max_rows=100), ""]
        else:
            L += ["### WNIOSEK", "", "**Nie znaleziono przewagi predykcyjnej**", "",
                  "Zaden model nie pobil losowego wyboru w sposob statystycznie "
                  "istotny na zbiorze TEST (po korekcie na wielokrotne testowanie).", ""]

    if prediction:
        L += ["## 9. Predykcja na nastepne losowanie", "",
              f"Po losowaniu nr **{prediction['after_draw_id']}**", ""]
        for k, tdf in prediction["top"].items():
            L += [f"### TOP {k}", "", _md_table(tdf, max_rows=k), ""]
        L += ["### 10 przykladowych zestawow (tylko do testow - NIE sa gwarancja wygranej)", ""]
        L += [f"{i+1}. {s}" for i, s in enumerate(prediction["sample_sets"])] + [""]

    L += ["---", "", "## Zastrzezenie", "",
          "KENO to losowanie bez pamieci. Kazda liczba ma prawdopodobienstwo "
          "20/70 = 28,57% w kazdym losowaniu, niezaleznie od historii. Ten system "
          "sluzy do TESTOWANIA hipotez o anomaliach, a nie do obiecywania wygranych. "
          "Wartosc oczekiwana zakladow KENO jest ujemna.", ""]

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
