"""KenoLab - modularny system analityczny dla polskiego KENO (20 z 70).

Moduly:
    config     - stale i konfiguracja (zakres liczb, tabela wyplat, sciezki)
    fetch      - pobieranie oficjalnych wynikow (API lotto.pl, mirror, generator syntetyczny)
    db         - warstwa SQLite + CSV, deduplikacja i walidacja
    analysis   - statystyki opisowe i testy losowosci
    models     - modele predykcyjne (kazdy zwraca 70 prawdopodobienstw)
    backtest   - walk-forward backtest + benchmarki + statystyka istotnosci
    predict    - ranking liczb na nastepne losowanie i przykladowe zestawy
    cli        - interfejs linii komend
    dashboard  - lokalny dashboard Flask z przyciskiem "Aktualizuj dane"
"""
__version__ = "1.0.0"
