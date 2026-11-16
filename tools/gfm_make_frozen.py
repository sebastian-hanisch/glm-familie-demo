"""Erzeugt data/gfm_frozen.json: die vorgerechneten Messreihen der Experimente 3 bis 5 (Link-Wahl, Residualdeviance, Gruppierung, Likelihood-Quotienten-Test).

Aufruf (aus dem Repo-Ordner):  python tools/gfm_make_frozen.py            rechnet alle Messungen neu (rund 10 bis 20 Minuten) und schreibt die Datei
                               python tools/gfm_make_frozen.py --check    prüft die Datei gegen Raster und Einzelwerte (rund 20 Sekunden; mit --full zusätzlich zwei ganze Zeilen)"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gfm_constants as C  # noqa: E402
import gfm_frozen as F  # noqa: E402

if __name__ == "__main__":
    if "--check" in sys.argv:
        problems = F.check(full="--full" in sys.argv)
        print("in Ordnung" if not problems else "ABWEICHUNGEN: " + "; ".join(problems))
        sys.exit(1 if problems else 0)
    t0 = time.time()
    data = F.build(log=lambda s: print(f"[{time.time() - t0:6.0f} s] {s}", flush=True))
    path = ROOT / C.FROZEN_FILE
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path} geschrieben ({time.time() - t0:.0f} s)")
