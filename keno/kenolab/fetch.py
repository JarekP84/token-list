"""Pobieranie historii losowan KENO.

Trzy zrodla, probowane w kolejnosci:
  1. Oficjalne API lotto.pl (api.lotto.pl/open/v1) - wymaga naglowka 'secret'
     (darmowy klucz z https://developers.lotto.pl) LUB dziala bez klucza dla
     czesci endpointow; paginowane po 'drawDate'.
  2. Publiczne archiwum tekstowe (mirror) - format: nr,data,l1,...,l20.
  3. Generator syntetyczny (uczciwie losowy) - wylacznie do testowania
     pipeline'u offline. NIGDY nie traktuj tych danych jak prawdziwych.

Kazde zrodlo zwraca liste rekordow: dict(draw_id, date, time, numbers[20]).
"""
from __future__ import annotations

import csv
import io
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date

import requests

from .config import POOL, DRAW_SIZE, NUM_MIN, NUM_MAX, RANDOM_SEED

LOTTO_API = "https://api.lotto.pl/open/v1/lotteries/draw-results/by-gametype"
MIRRORS = [
    "https://www.mbnet.com.pl/kn.txt",     # archiwum tekstowe KENO
]
UA = {"User-Agent": "KenoLab/1.0 (+local analysis)"}


@dataclass
class Draw:
    """Jedno losowanie KENO."""
    draw_id: int
    date: str      # YYYY-MM-DD
    time: str      # HH:MM (pusty string jesli zrodlo nie podaje godziny)
    numbers: list  # 20 liczb 1-70, posortowane

    def as_row(self) -> dict:
        d = asdict(self)
        d["numbers"] = ",".join(str(n) for n in self.numbers)
        return d


# --------------------------------------------------------------------------- #
# 1. Oficjalne API lotto.pl
# --------------------------------------------------------------------------- #
def fetch_lotto_api(secret: str | None = None, date_from: str = "2014-01-01",
                    date_to: str | None = None, page_size: int = 200,
                    timeout: int = 30, max_pages: int = 100000) -> list[Draw]:
    """Sciaga wyniki KENO z oficjalnego API lotto.pl.

    secret: klucz API (lub zmienna srodowiskowa LOTTO_API_SECRET).
    Zwraca liste Draw. Rzuca requests.HTTPError przy blokadzie/braku dostepu.
    """
    secret = secret or os.environ.get("LOTTO_API_SECRET")
    headers = dict(UA)
    if secret:
        headers["secret"] = secret
    date_to = date_to or date.today().isoformat()

    out: list[Draw] = []
    page = 1
    while page <= max_pages:
        params = {
            "gameType": "Keno",
            "drawDateFrom": date_from,
            "drawDateTo": date_to,
            "sort": "drawSystemId",
            "order": "ASC",
            "size": page_size,
            "index": page,
        }
        r = requests.get(LOTTO_API, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        items = payload.get("items") or []
        if not items:
            break
        for it in items:
            out.extend(_parse_api_item(it))
        if len(items) < page_size:
            break
        page += 1
    return out


def _parse_api_item(it: dict) -> list[Draw]:
    """Rozpakowuje jeden element API (moze zawierac kilka 'results')."""
    draws: list[Draw] = []
    raw_date = it.get("drawDate") or ""
    dt = raw_date.replace("Z", "")
    d_part, _, t_part = dt.partition("T")
    for res in it.get("results", [{}]):
        nums = res.get("resultsJson") or res.get("numbers") or it.get("numbers") or []
        nums = sorted(int(n) for n in nums)
        if len(nums) != DRAW_SIZE:
            continue
        draws.append(Draw(
            draw_id=int(res.get("drawSystemId") or it.get("drawSystemId") or 0),
            date=d_part,
            time=t_part[:5],
            numbers=nums,
        ))
    return draws


# --------------------------------------------------------------------------- #
# 2. Mirror tekstowy
# --------------------------------------------------------------------------- #
def fetch_mirror(url: str | None = None, timeout: int = 60) -> list[Draw]:
    """Pobiera archiwum tekstowe 'nr,DD.MM.YYYY,l1,...,l20'."""
    urls = [url] if url else MIRRORS
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=timeout)
            r.raise_for_status()
            return parse_mirror_text(r.text)
        except Exception as e:      # noqa: BLE001 - probujemy kolejny mirror
            last_err = e
    raise RuntimeError(f"Zadny mirror nie odpowiedzial: {last_err}")


def parse_mirror_text(text: str) -> list[Draw]:
    """Parser archiwum tekstowego (tolerancyjny na separatory , i ;)."""
    out: list[Draw] = []
    for line in io.StringIO(text):
        line = line.strip().replace(";", ",")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2 + DRAW_SIZE:
            continue
        try:
            draw_id = int(parts[0])
            d = _norm_date(parts[1])
            nums = sorted(int(x) for x in parts[2:2 + DRAW_SIZE])
        except ValueError:
            continue
        out.append(Draw(draw_id, d, "", nums))
    return out


def _norm_date(s: str) -> str:
    """Normalizuje date do YYYY-MM-DD."""
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


# --------------------------------------------------------------------------- #
# 3. Generator syntetyczny (offline / testy)
# --------------------------------------------------------------------------- #
def generate_synthetic(n_draws: int = 30000, seed: int = RANDOM_SEED,
                       start: str = "2016-01-01", per_day: int = 24) -> list[Draw]:
    """Uczciwie losowa historia KENO - benchmark 'czystego losu'.

    Sluzy do weryfikacji, ze pipeline i backtest NIE wykrywaja przewagi tam,
    gdzie jej z definicji nie ma (test negatywny / sanity check).
    """
    rng = random.Random(seed)
    pool = list(range(NUM_MIN, NUM_MAX + 1))
    d0 = datetime.fromisoformat(start)
    out: list[Draw] = []
    for i in range(n_draws):
        day, slot = divmod(i, per_day)
        ts = d0 + timedelta(days=day, hours=slot)
        out.append(Draw(
            draw_id=i + 1,
            date=ts.date().isoformat(),
            time=ts.strftime("%H:%M"),
            numbers=sorted(rng.sample(pool, DRAW_SIZE)),
        ))
    return out


# --------------------------------------------------------------------------- #
# Wysokopoziomowe API
# --------------------------------------------------------------------------- #
def fetch_all(prefer: str = "auto", allow_synthetic: bool = False,
              n_synthetic: int = 30000, **kw) -> tuple[list[Draw], str]:
    """Zwraca (losowania, opis_zrodla).

    prefer: 'api' | 'mirror' | 'synthetic' | 'auto' (API -> mirror -> syntetyk).
    allow_synthetic: dopuszcza fallback do danych syntetycznych.
    """
    order = {"auto": ["api", "mirror"], "api": ["api"],
             "mirror": ["mirror"], "synthetic": []}[prefer]
    errors = []
    for src in order:
        try:
            draws = fetch_lotto_api(**kw) if src == "api" else fetch_mirror()
            if draws:
                return draws, src
            errors.append(f"{src}: pusta odpowiedz")
        except Exception as e:      # noqa: BLE001
            errors.append(f"{src}: {e}")
    if prefer == "synthetic" or allow_synthetic:
        return generate_synthetic(n_synthetic), "synthetic"
    raise RuntimeError("Nie udalo sie pobrac danych: " + " | ".join(errors))


def load_csv(path) -> list[Draw]:
    """Wczytuje lokalny CSV (kolumny: draw_id,date,time,n1..n20 lub numbers)."""
    out: list[Draw] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if "numbers" in row and row["numbers"]:
                nums = [int(x) for x in row["numbers"].split(",")]
            else:
                nums = [int(row[f"n{i}"]) for i in range(1, DRAW_SIZE + 1)]
            out.append(Draw(int(row["draw_id"]), row["date"],
                            row.get("time", "") or "", sorted(nums)))
    return out
