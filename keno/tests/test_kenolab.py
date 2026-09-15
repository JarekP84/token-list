"""Testy jednostkowe KenoLab (pytest).

Najwazniejszy test: test_no_lookahead - sprawdza, ze predykcja dla losowania t
nie zmienia sie, gdy podmienimy przyszle losowania (brak wycieku z przyszlosci).
"""
import numpy as np
import pandas as pd
import pytest

from kenolab import analysis as an
from kenolab import backtest as bt
from kenolab import db as store
from kenolab import fetch as fx
from kenolab import features as ft
from kenolab import models as mo
from kenolab import predict as pr
from kenolab.config import POOL, DRAW_SIZE


@pytest.fixture(scope="module")
def frame():
    draws = fx.generate_synthetic(2000, seed=7)
    return pd.DataFrame([{"draw_id": d.draw_id, "date": d.date, "time": d.time,
                          "numbers": d.numbers} for d in draws])


@pytest.fixture(scope="module")
def M(frame):
    return store.to_matrix(frame)


def test_matrix_shape_and_rowsums(M):
    assert M.shape == (2000, POOL)
    assert (M.sum(axis=1) == DRAW_SIZE).all()


def test_validation_clean(frame):
    rep = store.validate(frame)
    assert rep["ok"] and rep["n_draws"] == 2000


def test_dedup(tmp_path):
    draws = fx.generate_synthetic(50, seed=1)
    p = tmp_path / "k.sqlite"
    assert store.upsert(draws, p) == 50
    assert store.upsert(draws, p) == 0          # duplikaty ignorowane


def test_frequency_table_consistency(M):
    f = an.frequency_table(M)
    assert len(f) == POOL
    assert f["count"].sum() == M.sum() == 2000 * DRAW_SIZE
    assert f["z_score"].abs().max() < 6         # uczciwe dane: brak ekstremow


def test_chi2_uniform_on_fair_data(M):
    assert an.chi_square_numbers(M)["p_value"] > 0.001


def test_independence_on_fair_data(M):
    assert an.chi_square_independence_consecutive(M)["p_value"] > 0.001


def test_entropy_near_max(M):
    e = an.entropy_analysis(M)
    assert e["efficiency"] > 0.99


def test_models_return_70_scores(M):
    for name, m in mo.default_models(fast=True).items():
        m.fit(M[:1500])
        p = m.predict_proba(M[:1500])
        assert p.shape == (POOL,), name
        assert np.isfinite(p).all(), name


def test_no_lookahead():
    """Predykcja dla losowania t musi zaleZec tylko od M[:t]."""
    M1 = store.to_matrix(pd.DataFrame(
        [{"draw_id": d.draw_id, "date": d.date, "time": d.time, "numbers": d.numbers}
         for d in fx.generate_synthetic(1500, seed=11)]))
    M2 = M1.copy()
    rng = np.random.default_rng(0)
    # podmieniamy PRZYSZLOSC (wiersze >= 1200) na inne losowania
    for i in range(1200, 1500):
        M2[i] = 0
        M2[i, rng.choice(POOL, DRAW_SIZE, replace=False)] = 1
    for name, m in mo.default_models(fast=True).items():
        m.fit(M1[:1200])
        a = m.predict_proba(M1[:1200])
        m.fit(M2[:1200])
        b = m.predict_proba(M2[:1200])
        assert np.allclose(a, b), f"{name} widzi przyszlosc"


def test_features_no_future_leak():
    M = store.to_matrix(pd.DataFrame(
        [{"draw_id": d.draw_id, "date": d.date, "time": d.time, "numbers": d.numbers}
         for d in fx.generate_synthetic(600, seed=3)]))
    X1, _ = ft.build_dataset(M, 400, 500)
    M2 = M.copy()
    M2[500:] = 0
    X2, _ = ft.build_dataset(M2, 400, 500)
    assert np.allclose(X1, X2)


def test_hits_for_bets_bounds(M):
    preds = bt.random_predictions(100, seed=5)
    hits = bt.hits_for_bets(preds, M[:100], [1, 5, 10])
    for k, h in hits.items():
        assert h.min() >= 0 and h.max() <= k


def test_monte_carlo_matches_theory():
    mc = bt.monte_carlo_bets(n_sim=50_000, bet_sizes=[1, 5, 10], seed=2)
    for _, r in mc.iterrows():
        assert abs(r["mean_hits"] - r["expected_theory"]) < 0.02


def test_hypergeom_distribution_sums_to_one():
    for k in range(1, 11):
        assert abs(bt.hypergeom_distribution(k).sum() - 1.0) < 1e-9


def test_chronological_split_no_overlap():
    sp = bt.chronological_split(20000, final_holdout=5000)
    assert sp["train"][1] == sp["valid"][0]
    assert sp["valid"][1] == sp["test"][0]
    assert sp["test"][1] == sp["holdout"][0] == 15000
    assert sp["holdout"][1] == 20000


def test_ranking_and_sets(M):
    r = pr.rank_numbers(M[:1500], models={"frequency": mo.FrequencyModel(None),
                                          "gap": mo.GapModel()})
    assert len(r) == POOL and r["rank"].iloc[0] == 1
    assert abs(r["probability"].sum() - DRAW_SIZE) < 1e-6
    sets = pr.sample_sets(r, n_sets=10, bet_size=10)
    assert len(sets) == 10 and all(len(set(s)) == 10 for s in sets)


def test_score_prediction():
    s = pr.score_prediction([1, 2, 3], [1, 2, 50])
    assert s["hits"] == 2 and s["hit_numbers"] == [1, 2]


def test_tracking_roundtrip(tmp_path):
    from kenolab import tracking as tk
    p = tmp_path / "t.sqlite"
    draws = fx.generate_synthetic(300, seed=4)
    store.upsert(draws, p)
    df = store.load_frame(p)
    M = store.to_matrix(df)
    r = pr.rank_numbers(M[:-1], models={"frequency": mo.FrequencyModel(None)})
    tk.save_prediction(r, int(df["draw_id"].iloc[-2]), path=p)
    assert tk.settle(p) == 1
    acc = tk.accuracy_table(p)
    assert len(acc) == 10 and acc["n_predictions"].max() == 1


def test_mirror_parser():
    txt = "1,01.01.2016," + ",".join(str(i) for i in range(1, 21)) + "\n"
    d = fx.parse_mirror_text(txt)
    assert len(d) == 1 and d[0].date == "2016-01-01" and len(d[0].numbers) == 20
