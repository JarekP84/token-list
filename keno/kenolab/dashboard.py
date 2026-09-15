"""Lokalny dashboard Flask.

Uruchomienie:  python -m kenolab.cli dashboard   (http://127.0.0.1:5000)

Przycisk "Aktualizuj dane" wykonuje w jednym kroku:
  1. pobranie najnowszych wynikow z lotto.pl (dedup + walidacja),
  2. rozliczenie poprzednich predykcji wzgledem prawdziwego wyniku,
  3. przeliczenie modeli i predykcje nastepnego losowania,
  4. wyswietlenie rankingu liczb i tabeli skutecznosci (zaklady 1-10).
"""
from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string, request

from . import db as store
from . import pipeline as pl
from . import tracking as tr
from .config import BET_SIZES

_state = {"busy": False, "last": None, "error": None}
_lock = threading.Lock()

TEMPLATE = """
<!doctype html><html lang="pl"><head><meta charset="utf-8">
<title>KenoLab - dashboard</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#0f1116;color:#e7e9ee}
 h1{font-size:20px} h2{font-size:16px;margin-top:28px;color:#9fd0ff}
 button{background:#2b6fd6;color:#fff;border:0;padding:10px 18px;border-radius:6px;
        font-size:15px;cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 table{border-collapse:collapse;margin-top:8px;font-size:13px}
 th,td{border:1px solid #2a2f3a;padding:4px 8px;text-align:right}
 th{background:#1a1f29;color:#9fd0ff} td:first-child,th:first-child{text-align:left}
 .num{display:inline-block;min-width:34px;margin:2px;padding:6px 8px;border-radius:6px;
      background:#1e2633;text-align:center;font-variant-numeric:tabular-nums}
 .top{background:#2b6fd6} .warn{color:#ffb86b} .ok{color:#8ce99a}
 pre{background:#171b23;padding:10px;border-radius:6px;overflow:auto;font-size:12px}
 .sets code{display:block;padding:2px 0}
</style></head><body>
<h1>KenoLab - dashboard KENO (20 z 70)</h1>
<p>Status bazy: <b>{{ n_draws }}</b> losowan | ostatnia aktualizacja: <b>{{ last_update or '-' }}</b>
   | zrodlo: <b>{{ src or '-' }}</b></p>
{% if src == 'synthetic' %}
<p class="warn">UWAGA: w bazie sa dane SYNTETYCZNE (uczciwie losowe), nie prawdziwe wyniki KENO.</p>
{% endif %}
<button id="btn" onclick="update()">Aktualizuj dane</button>
<span id="msg" style="margin-left:12px"></span>
<div id="out"></div>
<h2>Zastrzezenie</h2>
<p>KENO nie ma pamieci - kazda liczba ma 20/70 = 28,57% szansy w kazdym losowaniu.
Ranking sluzy do testowania hipotez, nie jest prognoza gwarantujaca wygrane.
Wartosc oczekiwana zakladow jest ujemna.</p>
<script>
async function update(){
  const b=document.getElementById('btn'), m=document.getElementById('msg');
  b.disabled=true; m.textContent='Pobieram wyniki i przeliczam modele...';
  try{
    const r=await fetch('/api/update',{method:'POST'});
    const j=await r.json();
    if(j.error){ m.textContent='Blad: '+j.error; }
    else { m.textContent='Gotowe ('+j.added+' nowych losowan)'; render(j); }
  }catch(e){ m.textContent='Blad: '+e; }
  b.disabled=false;
}
function tbl(rows, cols){
  if(!rows || !rows.length) return '<p>brak danych</p>';
  let h='<table><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
  for(const r of rows){ h+='<tr>'+cols.map(c=>'<td>'+fmt(r[c])+'</td>').join('')+'</tr>'; }
  return h+'</table>';
}
function fmt(v){ return (typeof v==='number' && !Number.isInteger(v))? v.toFixed(4): (v??''); }
function render(j){
  let h='<h2>Predykcja na nastepne losowanie (po losowaniu '+j.after_draw_id+')</h2>';
  for(const k of [10,15,20]){
    h+='<p><b>TOP '+k+':</b><br>'+j.ranking.slice(0,k).map(
        r=>'<span class="num top">'+r.number+'</span>').join('')+'</p>';
  }
  h+='<h2>Ranking wszystkich 70 liczb</h2>'+tbl(j.ranking,['rank','number','ensemble_z','probability']);
  h+='<h2>Przykladowe zestawy (do testow, nie gwarancja wygranej)</h2><div class="sets">'+
      j.sample_sets.map(s=>'<code>'+s.join(', ')+'</code>').join('')+'</div>';
  h+='<h2>Skutecznosc dotychczasowych predykcji (zaklady 1-10 liczb)</h2>'+
     tbl(j.accuracy,['bet_size','n_predictions','mean_hits','median_hits','max_hits',
                     'expected_random','edge','ci95_low','ci95_high','p_value']);
  h+='<h2>Ostatni wynik w bazie</h2><pre>'+JSON.stringify(j.last_draw)+'</pre>';
  h+='<h2>Walidacja danych</h2><pre>'+JSON.stringify(j.validation,null,1)+'</pre>';
  document.getElementById('out').innerHTML=h;
}
</script></body></html>
"""


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        df = store.load_frame()
        return render_template_string(
            TEMPLATE, n_draws=len(df), last_update=store.get_meta("last_update"),
            src=store.get_meta("source"))

    @app.post("/api/update")
    def api_update():
        """Aktualizuj dane -> rozlicz predykcje -> przelicz modele -> nowa predykcja."""
        with _lock:
            if _state["busy"]:
                return jsonify({"error": "Przeliczanie juz trwa"}), 409
            _state["busy"] = True
        try:
            allow_syn = request.args.get("synthetic") == "1"
            info = pl.update_data(allow_synthetic=allow_syn)
            df = store.load_frame()
            pred = pl.run_prediction(df)                   # zapisuje predykcje do bazy
            tr.settle()
            acc = tr.accuracy_table()
            payload = {
                "added": info["added"], "total": info["total"],
                "source": info["source"], "validation": info["validation"],
                "settled": info["settled_predictions"],
                "after_draw_id": pred["after_draw_id"],
                "ranking": pred["ranking"][["rank", "number", "ensemble_z", "probability"]]
                           .to_dict("records"),
                "sample_sets": pred["sample_sets"],
                "accuracy": acc.to_dict("records") if len(acc) else [],
                "last_draw": {"draw_id": int(df["draw_id"].iloc[-1]),
                              "date": df["date"].iloc[-1], "time": df["time"].iloc[-1],
                              "numbers": list(map(int, df["numbers"].iloc[-1]))},
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _state["last"] = payload
            return jsonify(payload)
        except Exception as e:                              # noqa: BLE001
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
        finally:
            _state["busy"] = False

    @app.get("/api/accuracy")
    def api_accuracy():
        tr.settle()
        acc = tr.accuracy_table()
        return jsonify(acc.to_dict("records") if len(acc) else [])

    @app.get("/api/ranking")
    def api_ranking():
        df = store.load_frame()
        if df.empty:
            return jsonify({"error": "Baza pusta"}), 400
        pred = pl.run_prediction(df, save=False)
        return jsonify(pred["ranking"].to_dict("records"))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
