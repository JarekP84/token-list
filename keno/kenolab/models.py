"""Modele predykcyjne: kazdy zwraca 70 prawdopodobienstw dla NASTEPNEGO losowania.

Kontrakt (wspolny interfejs):
    m.fit(M)             - trenuje na historii M (n x 70), wylacznie dane przeszle
    m.predict_proba(M)   - zwraca wektor 70 wartosci (score/prawdopodobienstwo)
                           dla losowania nastepujacego PO ostatnim wierszu M
    m.needs_refit(step)  - czy przetrenowac przy danym kroku walk-forward

Uwaga metodologiczna: przy uczciwym losowaniu KENO kazde prawdopodobienstwo to
20/70 = 0.2857. Modele moga jedynie probowac wykryc odstepstwa; brak odstepstw
oznacza, ze scores beda szumem wokol tej wartosci.
"""
from __future__ import annotations

import numpy as np

from .config import POOL, DRAW_SIZE, P_SINGLE, RANDOM_SEED
from .features import HistoryState, features_at, build_dataset


class BaseModel:
    """Wspolna baza modeli.

    _hist to inkrementalny stan historii (sumy kumulacyjne, ostatnie trafienia),
    dzieki ktoremu kolejny krok walk-forward kosztuje O(70), nie O(t*70).
    """
    name = "base"
    refit_every = 1          # co ile krokow walk-forward trenowac od nowa
    needs_cooc = False

    def __init__(self):
        self._hist = None

    def _sync(self, M: np.ndarray) -> HistoryState:
        if self._hist is None:
            self._hist = HistoryState(with_cooc=self.needs_cooc)
        return self._hist.sync(M)

    def fit(self, M: np.ndarray) -> "BaseModel":
        return self

    def predict_proba(self, M: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def needs_refit(self, step: int) -> bool:
        return step % self.refit_every == 0

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Skaluje dowolne score do sumy = 20 (oczekiwana liczba trafien),
        zachowujac kolejnosc - czyli do 'prawdopodobienstw' porownywalnych z 20/70."""
        s = np.asarray(scores, dtype=float)
        s = np.clip(s, 1e-9, None)
        return s / s.sum() * DRAW_SIZE


# --------------------------------------------------------------------------- #
# Modele proste (heurystyczne)
# --------------------------------------------------------------------------- #
class UniformModel(BaseModel):
    """Benchmark: wszystkie liczby rownie prawdopodobne (20/70)."""
    name = "uniform"

    def predict_proba(self, M):
        return np.full(POOL, P_SINGLE)


class FrequencyModel(BaseModel):
    """Model czestotliwosci: im czesciej liczba wypadala, tym wyzszy score.

    window=None -> cala historia. Wygladzanie Laplace'a stabilizuje krotkie okna.
    """
    name = "frequency"

    def __init__(self, window: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.window, self.alpha = window, alpha

    def _counts(self, M):
        """Zwraca (licznik trafien w oknie, dlugosc okna) na bazie sum kumulacyjnych."""
        st = self._sync(M)
        t = len(M)
        w = self.window or t
        cnt, eff = st.window_count(t, w)
        return cnt, max(eff, 1)

    def predict_proba(self, M):
        cnt, n = self._counts(M)
        return self._normalize((cnt + self.alpha) / (n + 2 * self.alpha))


class ColdModel(FrequencyModel):
    """Benchmark 'najrzadsze liczby': odwrotnosc czestotliwosci."""
    name = "cold"

    def predict_proba(self, M):
        cnt, n = self._counts(M)
        return self._normalize(1.0 - (cnt + self.alpha) / (n + 2 * self.alpha))


class RecencyModel(BaseModel):
    """Model recency: wykladniczo wazona czestotliwosc (ostatnie losowania wazniejsze).

    half_life - po ilu losowaniach waga spada o polowe.
    mode='hot'  -> preferuje liczby ostatnio czeste
    mode='cold' -> preferuje liczby z dluga aktualna przerwa
    """
    name = "recency"

    def __init__(self, half_life: float = 100.0, mode: str = "hot"):
        super().__init__()
        self.half_life, self.mode = half_life, mode
        self._r = np.exp(-np.log(2) / half_life)   # wspolczynnik zapominania
        self._S = np.zeros(POOL)                   # nieznormalizowana suma wazona
        self._W = 0.0                              # suma wag
        self._n = 0

    def _sync_ewma(self, M):
        """Aktualizuje EWMA tylko o nowe wiersze: S <- r*S + M_t."""
        if len(M) < self._n:                       # inna historia -> policz od nowa
            self._S, self._W, self._n = np.zeros(POOL), 0.0, 0
        for t in range(self._n, len(M)):
            self._S = self._r * self._S + M[t]
            self._W = self._r * self._W + 1.0
        self._n = len(M)

    def predict_proba(self, M):
        if len(M) == 0:
            return np.full(POOL, P_SINGLE)
        self._sync_ewma(M)
        ewma = self._S / max(self._W, 1e-12)
        if self.mode == "cold":
            return self._normalize(1.0 - ewma)
        return self._normalize(ewma)


class GapModel(BaseModel):
    """Model przerw: score = aktualna przerwa / srednia przerwa liczby.

    Testuje popularny (i teoretycznie bledny) 'blad gracza' - jesli jest
    nieistotny, potwierdza brak pamieci generatora.
    """
    name = "gap"

    def predict_proba(self, M):
        n = len(M)
        if n == 0:
            return np.full(POOL, P_SINGLE)
        st = self._sync(M)
        gap = st.gap(n)
        counts = st.total(n)
        mean_gap = np.divide(n, np.maximum(counts, 1), out=np.full(POOL, float(n)),
                             where=counts > 0)
        return self._normalize(gap / np.maximum(mean_gap, 1e-9) + 1e-6)


class CooccurrenceModel(BaseModel):
    """Model par/wspolwystepowania: score liczby = sila jej zwiazku z liczbami,
    ktore wypadly w poprzednim losowaniu (lift wzgledem oczekiwania)."""
    name = "cooccurrence"

    needs_cooc = True

    def __init__(self, lags: tuple[int, ...] = (1, 2)):
        super().__init__()
        self.lags = lags
        self.C = None

    def fit(self, M):
        st = self._sync(M)                 # aktualizuje C tylko o nowe wiersze
        self.C, self.n = st.C, len(M)
        return self

    def predict_proba(self, M):
        self.fit(M)
        score = np.zeros(POOL)
        p_pair = (DRAW_SIZE / POOL) * ((DRAW_SIZE - 1) / (POOL - 1))
        for lag in self.lags:
            if len(M) < lag:
                continue
            prev = np.flatnonzero(M[-lag])
            if prev.size == 0:
                continue
            obs = self.C[:, prev].sum(axis=1).astype(float)
            exp = self.n * p_pair * prev.size
            score += obs / max(exp, 1e-9)
        return self._normalize(score + 1e-6)


# --------------------------------------------------------------------------- #
# Modele uczenia maszynowego
# --------------------------------------------------------------------------- #
class SupervisedModel(BaseModel):
    """Baza dla modeli ML: uczy klasyfikator na (X, y) z modulu features.

    train_span  - ile ostatnich losowan uzyc do treningu
    stride      - co ktore losowanie brac do treningu (oszczednosc czasu)
    refit_every - co ile krokow walk-forward przetrenowac model
    """
    name = "supervised"

    needs_cooc = True

    def __init__(self, train_span: int = 5000, stride: int = 1,
                 refit_every: int = 250, seed: int = RANDOM_SEED):
        super().__init__()
        self.train_span, self.stride = train_span, stride
        self.refit_every, self.seed = refit_every, seed
        self.clf = None

    def _make_clf(self):
        raise NotImplementedError

    def fit(self, M):
        end = len(M)
        start = max(1, end - self.train_span)
        # osobny stan treningowy: build_dataset synchronizuje go krok po kroku,
        # wiec zadna cecha nie widzi losowania, ktore przewiduje
        X, y = build_dataset(M, start, end, stride=self.stride)
        if len(np.unique(y)) < 2 or len(X) < 100:
            self.clf = None
            return self
        self.clf = self._make_clf()
        self.clf.fit(X, y)
        return self

    def predict_proba(self, M):
        if self.clf is None:
            return np.full(POOL, P_SINGLE)
        st = self._sync(M)          # stan opisuje wylacznie losowania < len(M)
        X = features_at(M, len(M), st, st.C)
        return self.clf.predict_proba(X)[:, 1]


class LogisticModel(SupervisedModel):
    """Regresja logistyczna ze standaryzacja cech."""
    name = "logistic"

    def _make_clf(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, C=0.1))


class RandomForestModel(SupervisedModel):
    """Random Forest (ograniczona glebokosc - dane sa prawie czystym szumem)."""
    name = "random_forest"

    def _make_clf(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=6,
                                      min_samples_leaf=200, n_jobs=-1,
                                      random_state=self.seed)


class LightGBMModel(SupervisedModel):
    """Gradient boosting (LightGBM). Fallback: HistGradientBoosting z sklearn."""
    name = "lightgbm"

    def _make_clf(self):
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                  num_leaves=15, min_child_samples=200,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=self.seed, verbose=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                                  random_state=self.seed)


class NeuralNetModel(SupervisedModel):
    """Prosta siec neuronowa (MLP 32-16) na tych samych cechach."""
    name = "neural_net"

    def _make_clf(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32, 16),
                                           max_iter=60, early_stopping=True,
                                           random_state=self.seed))


# --------------------------------------------------------------------------- #
# Rejestr modeli
# --------------------------------------------------------------------------- #
def default_models(fast: bool = False) -> dict[str, BaseModel]:
    """Zwraca slownik modeli do backtestu. fast=True -> rzadsze przetrenowania."""
    refit = 1000 if fast else 250
    span = 3000 if fast else 6000
    stride = 3 if fast else 1
    return {
        "frequency_all": FrequencyModel(None),
        "frequency_1000": FrequencyModel(1000),
        "recency_hot": RecencyModel(100, "hot"),
        "recency_cold": RecencyModel(100, "cold"),
        "gap": GapModel(),
        "cooccurrence": CooccurrenceModel(),
        "logistic": LogisticModel(span, stride, refit),
        "random_forest": RandomForestModel(span, max(stride, 2), refit),
        "lightgbm": LightGBMModel(span, stride, refit),
        "neural_net": NeuralNetModel(span, max(stride, 2), refit),
    }


def benchmarks() -> dict[str, BaseModel]:
    """Trzy benchmarki wymagane w zadaniu."""
    return {
        "bench_random": UniformModel(),      # losowy wybor liczb
        "bench_hottest": FrequencyModel(None),   # najczestsze liczby
        "bench_coldest": ColdModel(None),        # najrzadsze liczby
    }
