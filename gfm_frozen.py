"""Die vorgerechneten Messreihen (Experimente 3 bis 5): erzeugen, laden, kontrollieren.

Die Messungen brauchen zusammen mehrere Minuten (Tausende Datensätze mit je bis zu 5 000 Containern) und sind darum eingefroren: `data/gfm_frozen.json` enthält die
Anteile über die Datensätze, `tools/gfm_make_frozen.py` erzeugt die Datei neu (`--check` prüft sie gegen die Suite). Dieselben Funktionen aus `gfm_experiments`
rechnen die kurzen Live-Wiederholungen der App."""

import json
from pathlib import Path

import gfm_constants as C
import gfm_experiments as E

ROOT = Path(__file__).resolve().parent
HINWEIS = ("erzeugt von tools/gfm_make_frozen.py; Einheit jeder Zeile: Wiederholung = ein Datensatz aus n Containern (Zufalls-Seeds aus gfm_experiments.seed_of). "
           "Anteile sind Bänder (Monte-Carlo-Standardfehler √(p(1−p)/Datensätze)), keine exakten Prozentzahlen.")


def _grid() -> dict:
    return {"link_export": [list(g) for g in C.LINK_EXPORT_GRID], "link_import": [list(g) for g in C.LINK_IMPORT_GRID],
            "schwelle": {"n": C.SCHWELLE_N, "reps": C.SCHWELLE_REPS, "schwellen": list(C.E4_SCHWELLEN)}, "modell": [list(g) for g in C.MODELL_GRID], "probe_reps": C.PROBE_REPS}


def build(log=print) -> dict:
    """Rechnet alle Messungen (mehrere Minuten) und gibt das Wörterbuch für die JSON-Datei zurück."""
    out = {"version": C.FROZEN_VERSION, "hinweis": HINWEIS, "grid": _grid()}
    rows = []
    for n, reps in C.LINK_EXPORT_GRID:
        recs = [E.link_choice_export(n, r) for r in range(reps)]
        rows.append({"n": n, **E.aggregate_link_export(recs)})
        log(f"link_export n={n} reps={reps} fertig")
    out["link_export"] = rows
    rows = []
    for n, reps in C.LINK_IMPORT_GRID:
        recs = [E.link_choice_import(n, r) for r in range(reps)]
        rows.append({"n": n, **E.aggregate_link_import(recs)})
        log(f"link_import n={n} reps={reps} fertig")
    out["link_import"] = rows

    recs = [E.schwelle_dataset(C.SCHWELLE_N, r, C.E4_SCHWELLEN) for r in range(C.SCHWELLE_REPS)]
    out["schwelle"] = {"n": C.SCHWELLE_N, "reps": C.SCHWELLE_REPS, "rows": [E.summarize_schwelle(recs, c, C.SCHWELLE_N) for c in C.E4_SCHWELLEN],
                       "probe": {"7": [recs[i][7]["du"] for i in range(C.PROBE_REPS)]}}
    log("schwelle fertig")

    modell, probe = {}, {}
    for welt, mit in (("mit", True), ("ohne", False)):
        modell[welt], probe[welt] = {}, {}
        for n, reps in C.MODELL_GRID:
            recs = [E.standgeld_dataset(n, r, mit) for r in range(reps)]
            cell = {}
            for m in ("M0", "M1"):
                for link in E.BINARY_LINKS:
                    cell[f"{m}/{link}"] = E.summarize_standgeld(E.aggregate_standgeld(recs, m, link), n)
            cell["lq"] = {link: E.lq_rejection(recs, link) for link in E.BINARY_LINKS}
            cell["aic"] = E.aic_choice_binary(recs)
            modell[welt][str(n)] = cell
            probe[welt][str(n)] = {"M0/probit": [recs[i]["M0/probit"]["du"] for i in range(C.PROBE_REPS)]}
            log(f"modell welt={welt} n={n} reps={reps} fertig")
    out["modell"] = modell
    out["probe_modell"] = probe
    return out


_CACHE: dict = {}


def load(path: Path | None = None) -> dict:
    p = path or (ROOT / C.FROZEN_FILE)
    key = str(p)
    if key not in _CACHE:
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _CACHE[key]


def check(frozen: dict | None = None, full: bool = False, log=print) -> list[str]:
    """Kontrolle der Datei gegen die Suite; Rückgabe: Liste der Abweichungen (leer = in Ordnung).
    Immer: Version, Raster und Struktur. Dazu die Einzelwerte der ersten Datensätze (Probe) neu gerechnet und mit Toleranz verglichen; mit full=True zusätzlich zwei
    ganze Zeilen der Link-Wahl (Zählungen ± 2 % der Datensätze, mindestens ± 2: die Plattformen unterscheiden sich in der letzten Stelle der Gleitkommafunktionen, und bei n = 20 liegen die AIC-Werte zweier Links oft so dicht beieinander,
    dass dieses Rauschen den Sieger wechselt; gemessen Linux gegen Windows: etwa 11 von 993 Datensätzen im Import bei n = 20)."""
    fz = frozen or load()
    problems = []
    if fz.get("version") != C.FROZEN_VERSION:
        problems.append("Version")
    if fz.get("grid") != _grid():
        problems.append("Raster der Datei passt nicht zu gfm_constants")
    if [r["n"] for r in fz.get("link_export", [])] != [g[0] for g in C.LINK_EXPORT_GRID]:
        problems.append("link_export: n")
    if [r["n"] for r in fz.get("link_import", [])] != [g[0] for g in C.LINK_IMPORT_GRID]:
        problems.append("link_import: n")
    if [r["c"] for r in fz["schwelle"]["rows"]] != list(C.E4_SCHWELLEN):
        problems.append("schwelle: Schwellen")
    for welt in ("mit", "ohne"):
        if [int(k) for k in fz["modell"][welt]] != [g[0] for g in C.MODELL_GRID]:
            problems.append(f"modell/{welt}: n")
    # Probe: die ersten Datensätze neu rechnen
    for i in range(C.PROBE_REPS):
        rec = E.schwelle_dataset(C.SCHWELLE_N, i, (7,))[7]
        if abs(rec["du"] - fz["schwelle"]["probe"]["7"][i]) > 1e-6 * abs(rec["du"]):
            problems.append(f"Probe Schwelle, Datensatz {i}")
            break
    for welt, mit in (("mit", True), ("ohne", False)):
        n = C.MODELL_GRID[0][0]
        for i in range(C.PROBE_REPS):
            rec = E.standgeld_dataset(n, i, mit, links=("probit",))["M0/probit"]
            if rec is None or abs(rec["du"] - fz["probe_modell"][welt][str(n)]["M0/probit"][i]) > 1e-6 * abs(rec["du"]):
                problems.append(f"Probe Modell {welt}, Datensatz {i}")
                break
    if full:
        for n, reps in C.LINK_EXPORT_GRID[:1]:
            recs = [E.link_choice_export(n, r) for r in range(reps)]
            agg = E.aggregate_link_export(recs)
            row = fz["link_export"][0]
            if any(abs(agg["wins"][k] - row["wins"][k]) > max(2, 0.02 * reps) for k in agg["wins"]):
                problems.append(f"link_export n={n}: Zählungen {agg['wins']} gegen Datei {row['wins']}")
        for n, reps in C.LINK_IMPORT_GRID[:1]:
            recs = [E.link_choice_import(n, r) for r in range(reps)]
            agg = E.aggregate_link_import(recs)
            row = fz["link_import"][0]
            if any(abs(agg["wins"][k] - row["wins"][k]) > max(2, 0.02 * reps) for k in agg["wins"]):
                problems.append(f"link_import n={n}: Zählungen {agg['wins']} gegen Datei {row['wins']}")
    return problems
