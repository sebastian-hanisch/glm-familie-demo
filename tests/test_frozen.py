"""Die vorgerechneten Messungen (`data/gfm_frozen.json`): Struktur, innere Stimmigkeit und Neu-Rechnung.

Immer: Raster und Struktur, die Einzelwerte der ersten Datensätze (Probe) neu gerechnet, zwei ganze Zeilen der Link-Wahl neu gerechnet (Zählungen ± 2 % der Datensätze).
Mit GFM_FULL=1 rechnet der Test die gesamte Datei neu (rund 10 Minuten) und vergleicht Anteile (± 2 % der Datensätze, mindestens ± 2 Zählungen je Zelle: die Plattformen unterscheiden sich in der letzten Stelle, bei Gleichständen im AIC wechselt der Sieger)."""

import json
import os

import pytest

import gfm_constants as C
import gfm_frozen as F


@pytest.fixture(scope="module")
def fz():
    return F.load()


def test_the_file_matches_the_suite(fz):
    assert F.check(fz, full=True) == []


def test_structure_and_inner_consistency(fz):
    assert fz["version"] == C.FROZEN_VERSION and "Bänder" in fz["hinweis"]
    for key in ("link_export", "link_import"):
        for row in fz[key]:
            assert sum(row["wins"].values()) == row["used"] <= row["reps"]
            assert all(v >= 0 for v in row["invalid"].values())
    for row in fz["schwelle"]["rows"]:
        assert 0 <= row["rej_hi"] <= 1 and 0 <= row["below_lo"] <= 1 and row["rej_hi"] + row["below_lo"] <= 1 + 1e-12 and row["valid"] <= row["reps"]
    for welt in ("mit", "ohne"):
        for n, cell in fz["modell"][welt].items():
            for key, s in cell.items():
                if key == "lq":
                    assert all(0 <= v["rej"] <= 1 and v["valid"] <= v["reps"] for v in s.values())
                elif key == "aic":
                    assert sum(s["wins"].values()) == s["used"]
                else:
                    assert s["valid"] <= s["reps"] and 0 <= s["rej_du"] <= 1
                    assert ("rej_dg4" in s) == key.startswith("M0")           # die 4-Zellen-Gruppierung gibt es nur für M0 (p = 3 ≤ 4 Zellen)


def test_the_file_is_valid_json_with_utf8():
    text = (F.ROOT / C.FROZEN_FILE).read_text(encoding="utf-8")
    assert json.loads(text)["version"] == C.FROZEN_VERSION


@pytest.mark.skipif(os.environ.get("GFM_FULL") != "1", reason="mit GFM_FULL=1: alle Zeilen neu rechnen (rund 10 Minuten)")
def test_full_recomputation_matches_the_file(fz):
    new = F.build(log=lambda s: None)
    for key in ("link_export", "link_import"):
        for a, b in zip(fz[key], new[key]):
            assert all(abs(a["wins"][k] - b["wins"][k]) <= max(2, 0.02 * a["reps"]) for k in a["wins"]), (key, a["n"])
    for a, b in zip(fz["schwelle"]["rows"], new["schwelle"]["rows"]):
        assert abs(a["rej_hi"] - b["rej_hi"]) <= 2 / a["reps"] and abs(a["below_lo"] - b["below_lo"]) <= 2 / a["reps"]
    for welt in ("mit", "ohne"):
        for n in fz["modell"][welt]:
            for key in ("M0/probit", "M1/probit"):
                x, y = fz["modell"][welt][n][key], new["modell"][welt][n][key]
                assert abs(x["rej_du"] - y["rej_du"]) <= 2 / x["reps"] and abs(x["rej_dg48"] - y["rej_dg48"]) <= 2 / x["reps"]
