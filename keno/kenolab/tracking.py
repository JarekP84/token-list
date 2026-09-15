"""Sledzenie skutecznosci predykcji w czasie (zaklady KENO 1-10 liczb).

Po kazdej predykcji zapisujemy ranking do bazy. Gdy pojawi sie prawdziwy wynik
kolejnego losowania, automatycznie liczymy trafienia dla kazdego rozmiaru
zakladu (TOP1..TOP10) i porownujemy z oczekiwaniem losowym (k*20/70).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH, BET_SIZES, P_SINGLE
from .db import connect, load_frame

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    after_draw_id INTEGER NOT NULL,   -- ostatnie znane losowanie w chwili predykcji
    model TEXT NOT NULL,              -- 'ensemble' lub nazwa modelu
    ranking TEXT NOT NULL,            -- JSON: lista 70 liczb w kolejnosci rankingu
    scores TEXT NOT NULL              -- JSON: lista score'ow rownolegla do ranking
);
CREATE TABLE IF NOT EXISTS prediction_results (
    prediction_id INTEGER NOT NULL,
    target_draw_id INTEGER NOT NULL,
    bet_size INTEGER NOT NULL,
    hits INTEGER NOT NULL,
    expected REAL NOT NULL,
    PRIMARY KEY (prediction_id, bet_size)
);
"""


def _con(path=DB_PATH):
    con = connect(path)
    con.executescript(SCHEMA)
    return con


def save_prediction(ranking: pd.DataFrame, after_draw_id: int,
                    model: str = "ensemble", path=DB_PATH) -> int:
    """Zapisuje ranking predykcji; zwraca id rekordu."""
    con = _con(path)
    cur = con.execute(
        "INSERT INTO predictions(created_at,after_draw_id,model,ranking,scores)"
        " VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), int(after_draw_id), model,
         json.dumps([int(x) for x in ranking["number"]]),
         json.dumps([float(x) for x in ranking.get("ensemble_z", ranking.iloc[:, -1])])),
    )
    con.commit()
    pid = cur.lastrowid
    con.close()
    return pid


def settle(path=DB_PATH) -> int:
    """Rozlicza wszystkie nierozliczone predykcje, dla ktorych znany jest juz
    wynik nastepnego losowania. Zwraca liczbe nowo rozliczonych predykcji."""
    df = load_frame(path)
    if df.empty:
        return 0
    order = df["draw_id"].tolist()
    pos = {d: i for i, d in enumerate(order)}
    con = _con(path)
    rows = con.execute(
        "SELECT p.id, p.after_draw_id, p.ranking FROM predictions p "
        "WHERE NOT EXISTS (SELECT 1 FROM prediction_results r WHERE r.prediction_id=p.id)"
    ).fetchall()
    settled = 0
    for pid, after_id, ranking_json in rows:
        i = pos.get(after_id)
        if i is None or i + 1 >= len(order):
            continue                                 # wynik jeszcze nieznany
        target_id = order[i + 1]
        actual = set(df["numbers"].iloc[i + 1])
        rank = json.loads(ranking_json)
        payload = []
        for k in BET_SIZES:
            hits = len(set(rank[:k]) & actual)
            payload.append((pid, int(target_id), k, hits, k * P_SINGLE))
        con.executemany(
            "INSERT OR REPLACE INTO prediction_results"
            "(prediction_id,target_draw_id,bet_size,hits,expected) VALUES (?,?,?,?,?)",
            payload)
        settled += 1
    con.commit()
    con.close()
    return settled


def accuracy_table(path=DB_PATH, model: str | None = None) -> pd.DataFrame:
    """Skutecznosc historyczna per rozmiar zakladu: srednia trafien vs oczekiwanie."""
    con = _con(path)
    q = ("SELECT r.bet_size, r.hits, r.expected, p.model FROM prediction_results r "
         "JOIN predictions p ON p.id=r.prediction_id")
    df = pd.read_sql_query(q, con)
    con.close()
    if df.empty:
        return df
    if model:
        df = df[df["model"] == model]
    g = df.groupby("bet_size")["hits"]
    out = pd.DataFrame({
        "bet_size": g.mean().index,
        "n_predictions": g.count().values,
        "mean_hits": g.mean().values,
        "median_hits": g.median().values,
        "max_hits": g.max().values,
        "expected_random": [k * P_SINGLE for k in g.mean().index],
    })
    out["edge"] = out["mean_hits"] - out["expected_random"]
    # 95% CI sredniej + p-value jednoprobkowe
    from scipy import stats
    pv, lo, hi = [], [], []
    for k in out["bet_size"]:
        v = df.loc[df["bet_size"] == k, "hits"].values.astype(float)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        lo.append(v.mean() - 1.96 * se)
        hi.append(v.mean() + 1.96 * se)
        pv.append(float(stats.ttest_1samp(v, k * P_SINGLE).pvalue) if len(v) > 1 else np.nan)
    out["ci95_low"], out["ci95_high"], out["p_value"] = lo, hi, pv
    return out


def history(path=DB_PATH, limit: int = 50) -> pd.DataFrame:
    """Ostatnie rozliczone predykcje (dla dashboardu)."""
    con = _con(path)
    df = pd.read_sql_query(
        "SELECT p.id, p.created_at, p.after_draw_id, r.target_draw_id, p.model,"
        " r.bet_size, r.hits, r.expected FROM predictions p"
        " JOIN prediction_results r ON r.prediction_id=p.id"
        " ORDER BY r.target_draw_id DESC, r.bet_size LIMIT ?", con, params=(limit * 10,))
    con.close()
    return df
