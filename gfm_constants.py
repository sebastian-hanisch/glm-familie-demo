"""Konstanten: Regler-Grenzen, Beschriftungen, Raster der vorgerechneten Messreihen, Farben und Presets."""

import gfm_algorithm as A

# --- Paare und Ausgänge ---------------------------------------------------------------------------------------------------------------------
OUTCOMES = {"stetig": "Verweilzeit d (stetig, Tage)", "zähl": "Standtage ⌈d⌉ (Zählung)", "binär": "Standgeld, d > 7 Tage (0/1)"}
PAIRS = (  # (Schlüssel, Ausgang, Familie, Link)
    ("normal/identity", "stetig", "normal", "identity"),
    ("gamma/log", "stetig", "gamma", "log"),
    ("gamma/inverse", "stetig", "gamma", "inverse"),
    ("poisson/log", "zähl", "poisson", "log"),
    ("poisson/identity", "zähl", "poisson", "identity"),
    ("nb2/log", "zähl", "nb2", "log"),
    ("binomial/logit", "binär", "binomial", "logit"),
    ("binomial/probit", "binär", "binomial", "probit"),
    ("binomial/cloglog", "binär", "binomial", "cloglog"),
)
PAIR_KEYS = tuple(p[0] for p in PAIRS)
FAMILY_LABEL = {"normal": "Normal", "poisson": "Poisson", "binomial": "Binomial (Bernoulli)", "gamma": "Gamma", "nb2": "NB2 (α fest)"}
LINK_LABEL = {"identity": "Identität", "log": "Log", "inverse": "Kehrwert", "logit": "Logit", "probit": "Probit", "cloglog": "cloglog"}
LINK_ADJ = {"identity": "Identitäts-Link", "log": "Log-Link", "inverse": "Kehrwert-Link", "logit": "Logit-Link", "probit": "Probit-Link", "cloglog": "cloglog-Link"}
NB2_ALPHA = 0.2                 # fester Wert von α bei NB2 in Experiment 1 und 2 (Experiment 6 schätzt α)
STANDGELD_TAGE = 7.0

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 180
HAND_DATEN = {"hand4": "Von Hand: vier Container ohne Zollhalt (eine Gruppe)", "hand9": "Von Hand: neun Container (zwei Gruppen)",
              "tabelle": "Container-Tabelle (alle Arten, n = 3 000)"}
START_OPTIONEN = {"daten": "aus den Daten (Startmittel)", "fuenf": "β₀ = (5; 0; …)", "null": "β₀ = (0; 0; …)"}
REGEL_OPTIONEN = {"fisher": "Fisher-Scoring (IRLS)", "newton": "Newton (beobachtete Information)", "beide": "beide"}
E2_AUSGANG = {"stetig": "Verweilzeit d (stetig)", "zähl": "Standtage ⌈d⌉ (Zählung)"}
E2_DATEN = {"hand9": "Von Hand: neun Container", "tabelle": "Container-Tabelle (n = 3 000)"}
E2_FAMILIEN = ("normal", "poisson", "gamma")
E2_LINKS = ("log", "identity", "inverse")
E3_WELT = {"export": "Export: Vorlauf stetig, wahr Identität", "import": "Import: nur 0/1-Merkmale, wahr Log"}
E3_N = (20, 50, 200, 1000, 5000)
E3_PUNKTE = (5.5, 15.0, 20.0)   # Vorhersagepunkte im Export (Vorlauf in Tagen); die Daten liegen in [1; 10]
E4_SCHWELLEN = (2, 3, 4, 5, 7, 10, 14, 21)
E4_N = (350, 1050, 3500)
E4_WELT = {"ohne": "ohne Kundeneffekt", "mit": "mit Kundeneffekt"}
E4_MODELL = {"M0": "M0: Zoll und Wochenende (p = 3)", "M1": "M1: mit elf Kundenmerkmalen (p = 14)"}
E5_N = (350, 1050, 3500)
E5_LINK = ("probit", "logit", "cloglog")
E6_ZEILEN = ("Separation", "nicht kanonischer Link", "zweiter Parameter (NB2-α)", "Lasso (IRLS als äußere Schleife)", "Quantilregression", "Kalibrierung",
             "gemischtes Modell")
LIVE_BUDGET = 40_000            # höchstens so viele Container je Live-Wiederholung insgesamt (Datensätze × n): rund 3 bis 8 Sekunden Rechenzeit
LIVE_REPS_MIN = 5

# --- Raster der vorgerechneten Messreihen (tools/gfm_make_frozen.py; Wiederholungen = Datensätze je Zeile) -----------------------------------
FROZEN_FILE = "data/gfm_frozen.json"
FROZEN_VERSION = 1
LINK_EXPORT_GRID = ((20, 1000), (50, 1000), (200, 1000), (1000, 500), (5000, 100))
LINK_IMPORT_GRID = ((20, 1000), (50, 1000), (200, 1000), (1000, 600), (5000, 300))
SCHWELLE_N, SCHWELLE_REPS = 1050, 600
MODELL_GRID = ((350, 400), (1050, 400), (3500, 250))
PROBE_REPS = 20                 # die ersten Datensätze, deren Einzelwerte die Datei zur Kontrolle mitführt

# --- Darstellung ----------------------------------------------------------------------------------------------------------------------------
COLORS = {"identity": "#2a6fb0", "log": "#c77700", "inverse": "#2e7d4f", "probit": "#7a3fb0", "logit": "#c77700", "cloglog": "#2e7d4f", "truth": "#5b6470",
          "fisher": "#2a6fb0", "newton": "#c4452d"}
CHART_HEIGHT = 380
CHI2_LEVEL = 0.95

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------
# Jeder Eintrag setzt die Regler, die er nennt; alle übrigen stehen auf dem Standardwert (SETTING_DEFAULTS in gfm_presets.py).
DEFAULT_PRESET = "Von Hand (Poisson, Identität)"
PRESETS = {
    DEFAULT_PRESET: {},
    "Zwei Gruppen, drei Links": {"e2_daten": "hand9", "e2_ausgang": "stetig", "e2_familie": "gamma", "e2_link": "log"},
    "Neun Paare (Container-Tabelle)": {"e1_daten": "tabelle", "e1_paar": "poisson/identity", "e1_regel": "beide", "e1_start": "daten"},
    "Link auf dem Export (Vorlauf)": {"e3_welt": "export", "e3_n": 50},
    "Link auf dem Import (0/1-Merkmale)": {"e3_welt": "import", "e3_n": 200},
    "Deviance: richtiges Modell": {"e4_c": 7, "e4_n": 1050, "e4_welt": "ohne", "e4_modell": "M0"},
    "Deviance: Schwelle 4 Tage": {"e4_c": 4, "e4_n": 1050, "e4_welt": "ohne", "e4_modell": "M0"},
    "Falsches Modell (Kunden fehlen)": {"e4_c": 7, "e4_n": 1050, "e4_welt": "mit", "e4_modell": "M0", "e5_n": 1050},
}
PRESET_HELP = {
    DEFAULT_PRESET: "Experiment 1: vier erfundene Container ohne Zollhalt (Standtage 4, 5, 6, 9; Mittel 6), Poisson mit Identitäts-Link, Start β₀ = 5. Fisher-Scoring trifft das Mittel in einem Schritt, Newton nähert sich ihm nur.",
    "Zwei Gruppen, drei Links": "Experiment 2: neun erfundene Container in zwei Gruppen (ohne und mit Zollhalt), Gamma-GLM mit Log-, Identitäts- und Kehrwert-Link. Alle drei Links geben dieselben Mittel und verschiedene Koeffizienten.",
    "Neun Paare (Container-Tabelle)": "Experiment 1 auf der Container-Tabelle mit n = 3 000: Poisson mit Identitäts-Link, Fisher und Newton nebeneinander; die Tabelle darunter zeigt alle neun Paare.",
    "Link auf dem Export (Vorlauf)": "Experiment 3: Export-Container, das Mittel der Verweilzeit ist linear im Vorlauf (wahr: Identität). n = 50, Vorhersagepunkt Vorlauf 15 Tage, außerhalb der Daten.",
    "Link auf dem Import (0/1-Merkmale)": "Experiment 3: Import-Container, nur Zoll und Wochenende, das Mittel ist multiplikativ (wahr: Log). n = 200, es gibt keinen Bereich außerhalb der Daten.",
    "Deviance: richtiges Modell": "Experiment 4: Standgeld bei mehr als 7 Tagen, das Modell ohne Kundeneffekt ist exakt richtig (Probit). n = 1 050.",
    "Deviance: Schwelle 4 Tage": "Experiment 4: dieselbe Welt, aber Standgeld ab mehr als 4 Tagen. Das Modell ist weiter exakt richtig; die Deviance verwirft es trotzdem.",
    "Falsches Modell (Kunden fehlen)": "Experimente 4 und 5: die Welt hat einen Kundeneffekt, das Modell kennt ihn nicht. n = 1 050. Welche Prüfung merkt es?",
}
assert set(PRESET_HELP) == set(PRESETS)
assert all(k in A.LINKS for k in ("identity", "log", "inverse", "logit", "probit", "cloglog"))
