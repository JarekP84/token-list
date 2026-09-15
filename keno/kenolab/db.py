"""Warstwa danych: SQLite + CSV, deduplikacja i walidacja."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH, CSV_PATH, DRAW_SIZE, NUM_MIN, NUM_MAX
from .fetch import Draw

SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    draw_id INTEGER PRIMARY KEY,   -- numer losowania (klucz => dedup automatyczny)
    date    TEXT NOT NULL,         -- YYYY-MM-DD
    time    TEXT,                  -- HH:MM
    numbers TEXT NOT NULL          -- 20 liczb rozdzielonych przecinkiem
);
CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(date, time);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert(draws: list[Draw], path: Path = DB_PATH) -> int:
    """Wstawia losowania, ignorujac duplikaty po draw_id. Zwraca liczbe nowych."""
    con = connect(path)
    before = con.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    con.executemany(
        "INSERT OR IGNORE INTO draws(draw_id,date,time,numbers) VALUES (?,?,?,?)",
        [(d.draw_id, d.date, d.time, ",".join(map(str, d.numbers))) for d in draws],
    )
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
    con.close()
    return after - before


def set_meta(key: str, value: str, path: Path = DB_PATH) -> None:
    con = connect(path)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, value))
    con.commit()
    con.close()


def get_meta(key: str, default: str = "", path: Path = DB_PATH) -> str:
    con = connect(path)
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    con.close()
    return row[0] if row else default


def load_frame(path: Path = DB_PATH) -> pd.DataFrame:
    """Zwraca DataFrame losowan uporzadkowany chronologicznie."""
    con = connect(path)
    df = pd.read_sql_query(
        "SELECT draw_id,date,time,numbers FROM draws ORDER BY date, time, draw_id", con)
    con.close()
    if df.empty:
        return df
    df["numbers"] = df["numbers"].map(lambda s: [int(x) for x in s.split(",")])
    return df.reset_index(drop=True)


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Macierz binarna (n_losowan x 70): 1 = liczba wypadla."""
    m = np.zeros((len(df), NUM_MAX - NUM_MIN + 1), dtype=np.int8)
    for i, nums in enumerate(df["numbers"].values):
        m[i, [n - NUM_MIN for n in nums]] = 1
    return m


def export_csv(df: pd.DataFrame, path: Path = CSV_PATH) -> Path:
    """Zapisuje CSV: draw_id,date,time,n1..n20."""
    rows = []
    for r in df.itertuples(index=False):
        row = {"draw_id": r.draw_id, "date": r.date, "time": r.time}
        row.update({f"n{i+1}": n for i, n in enumerate(r.numbers)})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return path


def validate(df: pd.DataFrame) -> dict:
    """Kontrola poprawnosci danych. Zwraca raport z listami problemow."""
    problems = {"bad_count": [], "out_of_range": [], "duplicated_numbers": [],
                "duplicate_draw_ids": [], "gaps_in_draw_id": [], "bad_dates": []}
    for r in df.itertuples(index=False):
        nums = r.numbers
        if len(nums) != DRAW_SIZE:
            problems["bad_count"].append(r.draw_id)
        if any(n < NUM_MIN or n > NUM_MAX for n in nums):
            problems["out_of_range"].append(r.draw_id)
        if len(set(nums)) != len(nums):
            problems["duplicated_numbers"].append(r.draw_id)
    ids = df["draw_id"].values
    dup = df["draw_id"][df["draw_id"].duplicated()].tolist()
    problems["duplicate_draw_ids"] = dup
    if len(ids) > 1:
        diffs = np.diff(np.sort(ids))
        problems["gaps_in_draw_id"] = int((diffs > 1).sum())
    problems["bad_dates"] = df.loc[
        pd.to_datetime(df["date"], errors="coerce").isna(), "draw_id"].tolist()
    problems["n_draws"] = len(df)
    problems["date_range"] = (df["date"].min(), df["date"].max()) if len(df) else None
    problems["ok"] = not (problems["bad_count"] or problems["out_of_range"]
                          or problems["duplicated_numbers"] or dup
                          or problems["bad_dates"])
    return problems
