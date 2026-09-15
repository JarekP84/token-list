# KenoLab — system analityczny dla polskiego KENO (20 z 70)

Kompletny, modularny pipeline w Pythonie: pobieranie oficjalnych wyników → baza SQLite/CSV →
analiza statystyczna → testy losowości → modele predykcyjne → **walk-forward backtest** →
ranking liczb na następne losowanie → dashboard z przyciskiem „Aktualizuj dane”
i śledzeniem skuteczności dla zakładów **1–10 liczb**.

Założenie metodologiczne: **nie zakładamy, że KENO da się przewidzieć.** System jest
narzędziem do falsyfikacji — sprawdza, czy w danych są powtarzalne anomalie, i jeśli ich nie
ma, mówi to wprost („Nie znaleziono przewagi predykcyjnej”).

## Instalacja na macOS (bez wpisywania komend)

1. W Finderze wejdz do katalogu `keno`.
2. Kliknij dwukrotnie **`install-mac.command`** - sprawdzi Pythona, doinstaluje
   `libomp` (wymagany przez LightGBM na macOS), utworzy `.venv`, zainstaluje
   biblioteki i pobierze historie losowan.
3. Kliknij dwukrotnie **`start-mac.command`** - uruchomi dashboard i otworzy
   http://127.0.0.1:5000 w przegladarce.

Gdy macOS zablokuje plik ("nie mozna otworzyc, bo pochodzi od
niezidentyfikowanego dewelopera"): kliknij plik prawym przyciskiem -> *Otworz*
-> *Otworz*, albo raz wykonaj w Terminalu `xattr -d com.apple.quarantine
install-mac.command start-mac.command`.

## Instalacja recznie (Linux / macOS / Windows)

```bash
cd keno
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Szybki start

```bash
python -m kenolab.cli update                  # pobierz historię z lotto.pl
python -m kenolab.cli analyze                 # analiza + raport reports/REPORT.md
python -m kenolab.cli backtest --fast --max-eval 6000 --mc-sims 1000000
python -m kenolab.cli predict                 # TOP10/15/20 + 10 zestawów testowych
python -m kenolab.cli accuracy                # skuteczność dotychczasowych predykcji
python -m kenolab.cli all --fast              # wszystko w jednym przebiegu
python -m kenolab.cli dashboard               # http://127.0.0.1:5000
```

Tryb offline (bez sieci) — dane syntetyczne, uczciwie losowe, do walidacji pipeline'u:

```bash
python -m kenolab.cli update --source synthetic --n-synthetic 30000
```

### Klucz do oficjalnego API

`api.lotto.pl` wymaga darmowego klucza (`https://developers.lotto.pl`):

```bash
export LOTTO_API_SECRET=twoj_klucz
python -m kenolab.cli update --source api
```

Bez klucza użyj `--source mirror` (publiczne archiwum tekstowe) albo wskaż własny CSV
(`kenolab.fetch.load_csv` + `kenolab.db.upsert`).

## Moduły

| Plik | Rola |
|---|---|
| `kenolab/config.py` | zasady gry, okna analityczne, podział TRAIN/VALID/TEST, tabela wypłat |
| `kenolab/fetch.py` | API lotto.pl, mirror tekstowy, generator syntetyczny, wczytywanie CSV |
| `kenolab/db.py` | SQLite + eksport CSV, deduplikacja po `draw_id`, walidacja danych |
| `kenolab/analysis.py` | częstotliwości/przerwy/z-score, pary–trójki–czwórki, zależności czasowe, chi², runs test, entropia, autokorelacja, Monte Carlo, test stabilności |
| `kenolab/features.py` | cechy dla modeli ML — liczone **wyłącznie z przeszłości** |
| `kenolab/models.py` | frequency, recency, gap, co-occurrence, regresja logistyczna, Random Forest, LightGBM, MLP + 3 benchmarki |
| `kenolab/backtest.py` | walk-forward, rozkłady trafień, CI, p-value (t-test + test permutacyjny), Monte Carlo ≥1 mln, werdykt istotności z korektą Bonferroniego |
| `kenolab/predict.py` | ranking 70 liczb (ensemble z-score), TOP 10/15/20, przykładowe zestawy |
| `kenolab/tracking.py` | zapis predykcji i rozliczanie ich po prawdziwym losowaniu (zakłady 1–10) |
| `kenolab/pipeline.py` | orkiestracja + generator raportu `reports/REPORT.md` |
| `kenolab/dashboard.py` | Flask: „Aktualizuj dane” = pobierz → rozlicz → przelicz → pokaż ranking |
| `tests/test_kenolab.py` | 18 testów, m.in. `test_no_lookahead` (brak wycieku z przyszłości) |

## Metodologia backtestu

1. Podział **chronologiczny** (bez tasowania): TRAIN 60% / VALIDATION 20% / TEST 20%,
   plus osobny hold-out = ostatnie 10 000 losowań.
2. Dla każdego losowania `t` model jest trenowany na `M[:t]` i przewiduje rozkład dla `t`.
   Test `test_no_lookahead` weryfikuje, że podmiana przyszłych losowań nie zmienia predykcji.
3. Zakład o rozmiarze `k` = `k` liczb o najwyższym score. Liczymy trafienia dla `k = 1..10`.
4. Porównanie z trzema benchmarkami: losowy wybór, najczęstsze liczby, najrzadsze liczby;
   dodatkowo z rozkładem hipergeometrycznym i z ≥1 000 000 symulacji Monte Carlo.
5. Model wybieramy na VALIDATION, **oceniamy na TEST**; przewaga potwierdzona na TEST jest
   dodatkowo weryfikowana na świeżym hold-oucie 10 000 losowań.
6. Istotność: dodatnia przewaga **i** p < 0,05 po korekcie Bonferroniego na liczbę
   porównań (modele × rozmiary zakładu).

## Metryki raportowane dla każdej strategii

średnia trafień, mediana, maksimum, rozkład 0/1/2/…/k trafień (obok teorii
hipergeometrycznej), 95% CI średniej, p-value vs strategia losowa (t-test parowany
i test permutacyjny), log-loss/Brier/AUC pełnego wektora 70 prawdopodobieństw.

## Tabela wypłat

`config.PAYOUTS` zawiera **przybliżone** mnożniki; flaga `PAYOUTS_VERIFIED = False`.
Kolumny `ev_payout_per_1zl` są orientacyjne — zweryfikuj tabelę na lotto.pl i ustaw flagę
na `True`, zanim potraktujesz EV jako wynik.

## Zastrzeżenie

KENO to losowanie bez pamięci: każda liczba ma 20/70 = 28,57% szansy w każdym losowaniu,
niezależnie od historii. Ranking i zestawy służą do testowania hipotez, nie są prognozą
gwarantującą wygrane. Wartość oczekiwana zakładów KENO jest ujemna — grasz na minusie.
