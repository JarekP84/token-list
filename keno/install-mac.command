#!/bin/bash
# KenoLab - instalator dla macOS. Uruchom podwojnym klikiem w Finderze.
# Tworzy wirtualne srodowisko, instaluje zaleznosci i przygotowuje baze danych.
set -u
cd "$(dirname "$0")" || exit 1

echo "=============================================="
echo "  KenoLab - instalacja (macOS)"
echo "=============================================="
echo

# --- 1. Python 3.10+ --------------------------------------------------------
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
    major=${ver%%.*}; minor=${ver##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then PY="$cand"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "BLAD: nie znaleziono Pythona 3.10+."
  echo "Zainstaluj go poleceniem:  brew install python@3.12"
  echo "(a jesli nie masz Homebrew: https://brew.sh)"
  read -r -p "Enter zamyka okno..." _; exit 1
fi
echo "[1/4] Python: $($PY --version)"

# --- 2. libomp (wymagany przez LightGBM na macOS) ---------------------------
if command -v brew >/dev/null 2>&1; then
  if ! brew list libomp >/dev/null 2>&1; then
    echo "[2/4] Instaluje libomp (potrzebny dla LightGBM)..."
    brew install libomp || echo "      Uwaga: libomp sie nie zainstalowal - LightGBM zostanie pominiety."
  else
    echo "[2/4] libomp: obecny"
  fi
else
  echo "[2/4] Brak Homebrew - pomijam libomp (LightGBM moze nie dzialac; reszta modeli tak)."
fi

# --- 3. Wirtualne srodowisko + zaleznosci -----------------------------------
echo "[3/4] Tworze srodowisko i instaluje biblioteki (kilka minut)..."
"$PY" -m venv .venv || { echo "BLAD: nie udalo sie utworzyc .venv"; read -r -p "Enter..." _; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
if ! python -m pip install --quiet -r requirements.txt; then
  echo "BLAD: instalacja bibliotek nie powiodla sie."
  read -r -p "Enter zamyka okno..." _; exit 1
fi

# --- 4. Pierwsze pobranie danych -------------------------------------------
echo "[4/4] Pobieram historie losowan KENO..."
if python -m kenolab.cli update; then
  echo
  echo "GOTOWE. Dane pobrane."
else
  echo
  echo "Nie udalo sie pobrac oficjalnych wynikow (brak klucza API lub blokada sieci)."
  echo "Mozesz:"
  echo "  a) zdobyc darmowy klucz na https://developers.lotto.pl i uruchomic w Terminalu:"
  echo "       cd \"$(pwd)\" && source .venv/bin/activate"
  echo "       export LOTTO_API_SECRET=twoj_klucz && python -m kenolab.cli update"
  echo "  b) sprobowac archiwum:  python -m kenolab.cli update --source mirror"
  echo "  c) przetestowac program na danych losowych (NIE sa to prawdziwe wyniki):"
  echo "       python -m kenolab.cli update --source synthetic --n-synthetic 30000"
fi

echo
echo "Teraz uruchamiaj program podwojnym klikiem na pliku: start-mac.command"
read -r -p "Enter zamyka okno..." _
