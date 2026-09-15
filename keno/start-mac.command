#!/bin/bash
# KenoLab - uruchomienie dashboardu. Podwojny klik w Finderze.
set -u
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
  echo "Brak srodowiska. Uruchom najpierw: install-mac.command"
  read -r -p "Enter zamyka okno..." _; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=============================================="
echo "  KenoLab - dashboard"
echo "=============================================="
echo "Otwieram http://127.0.0.1:5000 w przegladarce."
echo "Zatrzymanie serwera: Ctrl+C (albo zamknij to okno)."
echo

# przegladarka po 2 sekundach, gdy serwer juz wstanie
( sleep 2; open "http://127.0.0.1:5000" >/dev/null 2>&1 ) &
python -m kenolab.cli dashboard
