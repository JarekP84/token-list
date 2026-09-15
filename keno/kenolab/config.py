"""Konfiguracja globalna KenoLab."""
from pathlib import Path

# --- Zasady gry -------------------------------------------------------------
NUM_MIN, NUM_MAX = 1, 70          # zakres liczb w KENO
POOL = NUM_MAX - NUM_MIN + 1      # 70 liczb w puli
DRAW_SIZE = 20                    # 20 liczb wypada w kazdym losowaniu
P_SINGLE = DRAW_SIZE / POOL       # 20/70 ~ 0.2857 - szansa pojedynczej liczby

# --- Sciezki ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "keno.sqlite"
CSV_PATH = DATA_DIR / "keno.csv"
for _d in (DATA_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Okna analityczne -------------------------------------------------------
WINDOWS = [50, 100, 250, 500, 1000, 5000, 10000]   # None = cala historia

# --- Podzial chronologiczny -------------------------------------------------
# Udzialy TRAIN / VALIDATION / TEST liczone po osi czasu (bez tasowania!).
SPLIT_TRAIN = 0.60
SPLIT_VALID = 0.20
SPLIT_TEST = 0.20
# Dodatkowy, "swiezy" hold-out: ostatnie N losowan, uzywane tylko do finalnej
# weryfikacji metody, ktora wykazala przewage na TEST.
FINAL_HOLDOUT = 10000

# --- Backtest / Monte Carlo -------------------------------------------------
MIN_BACKTEST_TRIALS = 100_000      # min. liczba prob historycznych (losowania x zaklady)
MIN_MONTE_CARLO = 1_000_000        # min. liczba symulacji Monte Carlo
BET_SIZES = list(range(1, 11))     # zaklady KENO od 1 do 10 liczb

# --- Tabela wyplat KENO (mnoznik stawki) ------------------------------------
# UWAGA: wartosci przyblizone, wg publikowanej tabeli wygranych KENO (stawka 1 zl,
# mnoznik x1). Sluza wylacznie do orientacyjnego liczenia EV - PRZED uzyciem
# zweryfikuj aktualna tabele na lotto.pl i popraw ponizsze liczby.
PAYOUTS = {
    1:  {1: 2},
    2:  {2: 9},
    3:  {3: 40, 2: 2},
    4:  {4: 130, 3: 6, 2: 1},
    5:  {5: 900, 4: 20, 3: 3},
    6:  {6: 2000, 5: 60, 4: 6, 3: 1},
    7:  {7: 5000, 6: 150, 5: 20, 4: 3},
    8:  {8: 20000, 7: 500, 6: 50, 5: 8, 4: 1},
    9:  {9: 50000, 8: 2500, 7: 150, 6: 20, 5: 4},
    10: {10: 100000, 9: 5000, 8: 400, 7: 40, 6: 8, 5: 2},
}

# Ustaw na True po samodzielnym zweryfikowaniu tabeli powyzej - dopoki jest
# False, wszystkie kolumny EV (ev_payout_per_1zl) sa oznaczane jako orientacyjne
# i NIE naleza do wnioskow analizy.
PAYOUTS_VERIFIED = False

RANDOM_SEED = 20240915
