"""Interfejs linii komend KenoLab.

Przyklady:
    python -m kenolab.cli update                 # pobierz wyniki z lotto.pl
    python -m kenolab.cli update --synthetic     # tryb offline (dane losowe)
    python -m kenolab.cli analyze
    python -m kenolab.cli backtest --fast --max-eval 3000 --mc-sims 1000000
    python -m kenolab.cli predict
    python -m kenolab.cli accuracy               # skutecznosc dotychczasowych predykcji
    python -m kenolab.cli all --fast
    python -m kenolab.cli dashboard              # http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import db as store
from . import pipeline as pl
from . import tracking as tr
from .config import MIN_MONTE_CARLO, FINAL_HOLDOUT


def _frame():
    df = store.load_frame()
    if df.empty:
        raise SystemExit("Baza pusta - uruchom najpierw: python -m kenolab.cli update")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kenolab", description="System analityczny KENO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("update", help="pobierz i zapisz wyniki")
    p.add_argument("--source", choices=["auto", "api", "mirror", "synthetic"], default="auto")
    p.add_argument("--synthetic", action="store_true",
                   help="dopusc fallback do danych syntetycznych (offline)")
    p.add_argument("--n-synthetic", type=int, default=30000)

    sub.add_parser("analyze", help="analiza statystyczna + raport")

    p = sub.add_parser("backtest", help="walk-forward backtest + Monte Carlo")
    p.add_argument("--fast", action="store_true", help="rzadsze przetrenowania modeli ML")
    p.add_argument("--max-eval", type=int, default=None,
                   help="ogranicz dlugosc segmentow VALID/TEST")
    p.add_argument("--mc-sims", type=int, default=MIN_MONTE_CARLO)
    p.add_argument("--holdout", type=int, default=FINAL_HOLDOUT)
    p.add_argument("--min-history", type=int, default=1000)

    p = sub.add_parser("predict", help="ranking liczb na nastepne losowanie")
    p.add_argument("--sets", type=int, default=10)
    p.add_argument("--bet-size", type=int, default=10)

    sub.add_parser("accuracy", help="skutecznosc dotychczasowych predykcji (zaklady 1-10)")

    p = sub.add_parser("all", help="update (jesli trzeba) + analyze + backtest + predict")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--max-eval", type=int, default=3000)
    p.add_argument("--mc-sims", type=int, default=MIN_MONTE_CARLO)
    p.add_argument("--synthetic", action="store_true")

    p = sub.add_parser("dashboard", help="uruchom lokalny dashboard Flask")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)

    a = ap.parse_args(argv)

    if a.cmd == "update":
        info = pl.update_data(prefer=a.source, allow_synthetic=a.synthetic or a.source == "synthetic",
                              n_synthetic=a.n_synthetic)
        print(json.dumps(info, indent=2, default=str))

    elif a.cmd == "analyze":
        df = _frame()
        res = pl.run_analysis(df)
        meta = {"source": store.get_meta("source"), "validation": store.validate(df)}
        path = pl.write_report(res, None, None, meta)
        print(json.dumps(res["tests"], indent=2, default=str))
        print(f"Raport: {path}")

    elif a.cmd == "backtest":
        df = _frame()
        res = pl.run_full_backtest(df, fast=a.fast, mc_sims=a.mc_sims,
                                   final_holdout=a.holdout, min_history=a.min_history,
                                   max_eval=a.max_eval)
        print(res["results"]["test"].to_string(index=False, max_colwidth=18))
        print(json.dumps(res["verdict"], indent=2, default=str))
        if not res["verdict"]["has_edge"]:
            print("\n>>> Nie znaleziono przewagi predykcyjnej")

    elif a.cmd == "predict":
        df = _frame()
        res = pl.run_prediction(df, n_sets=a.sets, bet_size=a.bet_size)
        for k, t in res["top"].items():
            print(f"\n--- TOP {k} ---")
            print(t.to_string(index=False))
        print("\nPrzykladowe zestawy (NIE sa gwarancja wygranej):")
        for i, s in enumerate(res["sample_sets"], 1):
            print(f"  {i:2d}. {s}")

    elif a.cmd == "accuracy":
        n = tr.settle()
        t = tr.accuracy_table()
        print(f"Rozliczono nowych predykcji: {n}")
        print(t.to_string(index=False) if len(t) else "Brak rozliczonych predykcji.")

    elif a.cmd == "all":
        info = pl.update_data(allow_synthetic=a.synthetic)
        print(json.dumps({k: v for k, v in info.items() if k != "validation"}, indent=2))
        df = _frame()
        analysis = pl.run_analysis(df)
        back = pl.run_full_backtest(df, fast=a.fast, max_eval=a.max_eval, mc_sims=a.mc_sims)
        pred = pl.run_prediction(df)
        meta = {"source": info["source"], "validation": info["validation"]}
        print(f"Raport: {pl.write_report(analysis, back, pred, meta)}")
        if not back["verdict"]["has_edge"]:
            print(">>> Nie znaleziono przewagi predykcyjnej")

    elif a.cmd == "dashboard":
        from .dashboard import create_app
        create_app().run(host=a.host, port=a.port, debug=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
