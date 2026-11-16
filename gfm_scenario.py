"""Vehikel: die Container-Verweilzeit (synthetisch, erfunden) und das Handbeispiel mit neun Containern.

Nachbau der Merkmale und der Verweilzeit der `verweilzeit-demo` (nur der Teil, den diese Demo braucht, mit dem exakten SplitMix64 aus `gfm_rng.py`;
nicht bitgleich zu jener Demo, also mit eigenen Zahlen): die Verweilzeit d in Tagen ist lognormal, ln d ~ N(mu, s²), mit

  Import    ln-Median = ln 3,5 + 0,7·Zoll + 0,3·Wochenende + Kundeneffekt,   s = 0,55   (Zoll mit Wahrscheinlichkeit 0,25)
  Export    ln-Median = ln(0,6 + 0,85·Vorlauf) + 0,3·Kundeneffekt,           s = 0,25   (Vorlauf gleichverteilt in [1; 10] Tagen)
  Umschlag  ln-Median = ln(0,3 + 0,8·Vorlauf) + 0,3·Kundeneffekt,            s = 0,35   (Vorlauf gleichverteilt in [0,5; 6])
  Leer      ln-Median = ln 7 + 0,3·Wochenende + Kundeneffekt,                s = 0,90

Die zwölf Kundeneffekte sind Konstanten (Normal(0; 0,3), einmal gezogen; realisierte Standardabweichung 0,379 mit ddof = 1). Zwei Teile des Modells sind
exakt bekannt und darum die Prüfsteine der Demo: das Mittel der Verweilzeit im Export ist **linear** im Vorlauf (Faktor C_EXPORT), und das Standgeld
(Verweilzeit > 7 Tage) im Import ist ein **Probit**, exakt linear in Zoll, Wochenende und Kunde."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import special

import gfm_rng as R

# --- Modellkonstanten -------------------------------------------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
U_SD_NOMINAL = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)       # Kundeneffekt je Art (bei Export und Umschlag ein Drittel)
S_BASE = (0.55, 0.25, 0.35, 0.90)    # Streuung der ln-Verweilzeit je Art
P_ZOLL = 0.25
STANDGELD_TAGE = 7.0                 # Standgeld ab einer Verweilzeit über 7 Tagen
UK = np.array([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
               -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771])
LN35 = R.exact_log(3.5)
LN7 = R.exact_log(7.0)
# Das Mittel der Verweilzeit ist Faktor mal die Formel des ln-Medians ohne Kundeneffekt: E[exp(u·Kunde)]·exp(s²/2)
C_EXPORT = float(np.mean(np.exp(U_SCALE[1] * UK)) * math.exp(S_BASE[1] ** 2 / 2))    # 1,0294
C_IMPORT = float(np.mean(np.exp(U_SCALE[0] * UK)) * math.exp(S_BASE[0] ** 2 / 2))
TABLE_FEATURES = ("Achsenabschnitt", "Export", "Umschlag", "Leer", "Zoll", "Wochenende", "log(1 + Vorlauf)", "Rauschmerkmal")


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Der Orakel-Wert: ln-Median und Streuung der ln-Verweilzeit je Container (vektorisiert)."""
    art = np.asarray(art)
    u = UK[np.asarray(kunde)] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    return np.log(med) + u, np.take(S_BASE, art)


@dataclass(frozen=True)
class Tabelle:
    """Container-Tabelle aller Arten: Merkmale, ln-Median und Streuung (Orakel) und die Verweilzeit d."""
    art: np.ndarray
    kunde: np.ndarray
    zoll: np.ndarray
    we: np.ndarray
    vorlauf: np.ndarray
    rausch: np.ndarray
    mu: np.ndarray
    s: np.ndarray
    d: np.ndarray

    @property
    def n(self) -> int:
        return len(self.d)

    def designmatrix(self) -> np.ndarray:
        """Achsenabschnitt, Export, Umschlag, Leer, Zoll, Wochenende, log(1 + Vorlauf), Rauschmerkmal (p = 8)."""
        a = self.art
        return np.column_stack([np.ones(self.n), a == 1, a == 2, a == 3, self.zoll, self.we, np.log1p(self.vorlauf), self.rausch]).astype(float)

    def standtage(self) -> np.ndarray:
        return np.ceil(self.d)

    def standgeld(self) -> np.ndarray:
        return (self.d > STANDGELD_TAGE).astype(float)


def draw_table(n: int, seed: int) -> Tabelle:
    """n Container aller Arten (SplitMix64, Zufalls-Seed `seed`); das Rauschmerkmal hat keine Wirkung auf die Verweilzeit."""
    rng = R.SplitMix64(seed)
    art = np.zeros(n, int)
    kunde = np.zeros(n, int)
    zoll = np.zeros(n, int)
    we = np.zeros(n, int)
    vorlauf = np.zeros(n)
    rausch = np.zeros(n)
    z = np.zeros(n)
    cum = np.cumsum(ART_P)
    for i in range(n):
        u = rng.random()
        a = 3
        for j in range(3):
            if u < cum[j]:
                a = j
                break
        art[i] = a
        kunde[i] = rng.below(K_KUNDEN)
        zoll[i] = int(a == 0 and rng.random() < P_ZOLL)
        we[i] = int(rng.below(7) >= 5)
        if a == 1:
            vorlauf[i] = 1.0 + 9.0 * rng.random()
        elif a == 2:
            vorlauf[i] = 0.5 + 5.5 * rng.random()
        rausch[i] = (rng.random() - 0.5) * math.sqrt(12.0)
        z[i] = rng.normal()
    mu, s = log_median_and_s(art, kunde, zoll, we, vorlauf)
    return Tabelle(art, kunde, zoll, we, vorlauf, rausch, mu, s, np.exp(mu + s * z))


@dataclass(frozen=True)
class ImportDaten:
    zoll: np.ndarray
    we: np.ndarray
    kunde: np.ndarray
    mu: np.ndarray
    s: np.ndarray
    d: np.ndarray

    @property
    def n(self) -> int:
        return len(self.d)

    def standgeld(self, tage: float = STANDGELD_TAGE) -> np.ndarray:
        return (self.d > tage).astype(float)

    def wahre_wahrscheinlichkeit(self, tage: float = STANDGELD_TAGE) -> np.ndarray:
        """Die wahre Wahrscheinlichkeit für Standgeld ab `tage`: ein Probit, Φ((ln-Median − ln tage)/s)."""
        return special.ndtr((self.mu - R.exact_log(tage)) / self.s)


def draw_import(n: int, seed: int, kundeneffekt: bool = True) -> ImportDaten:
    """n Import-Container: Zoll, Wochenende und Kunde; ohne Kundeneffekt hängt der ln-Median nur an Zoll und Wochenende (Vergleichswelt)."""
    rng = R.SplitMix64(seed)
    zoll = np.zeros(n, int)
    we = np.zeros(n, int)
    kunde = np.zeros(n, int)
    z = np.zeros(n)
    for i in range(n):
        zoll[i] = int(rng.random() < P_ZOLL)
        we[i] = int(rng.below(7) >= 5)
        kunde[i] = rng.below(K_KUNDEN)
        z[i] = rng.normal()
    mu = LN35 + 0.7 * zoll + 0.3 * we + (UK[kunde] * U_SCALE[0] if kundeneffekt else 0.0)
    s = np.full(n, S_BASE[0])
    return ImportDaten(zoll, we, kunde, mu, s, np.exp(mu + s * z))


@dataclass(frozen=True)
class ExportDaten:
    vorlauf: np.ndarray
    kunde: np.ndarray
    mu: np.ndarray
    s: np.ndarray
    d: np.ndarray

    @property
    def n(self) -> int:
        return len(self.d)


def draw_export(n: int, seed: int) -> ExportDaten:
    """n Export-Container: Vorlauf gleichverteilt in [1; 10] Tagen und Kunde; das Mittel der Verweilzeit ist exakt linear im Vorlauf."""
    rng = R.SplitMix64(seed)
    vorlauf = np.zeros(n)
    kunde = np.zeros(n, int)
    z = np.zeros(n)
    for i in range(n):
        vorlauf[i] = 1.0 + 9.0 * rng.random()
        kunde[i] = rng.below(K_KUNDEN)
        z[i] = rng.normal()
    mu = np.log(0.6 + 0.85 * vorlauf) + UK[kunde] * U_SCALE[1]
    s = np.full(n, S_BASE[1])
    return ExportDaten(vorlauf, kunde, mu, s, np.exp(mu + s * z))


def export_mittel(vorlauf):
    """Das wahre Mittel der Verweilzeit im Export: C_EXPORT·(0,6 + 0,85·Vorlauf), exakt linear."""
    return C_EXPORT * (0.6 + 0.85 * np.asarray(vorlauf, float))


def import_mittel(zoll, we):
    """Das wahre Mittel der Verweilzeit im Import: C_IMPORT·3,5·exp(0,7·Zoll + 0,3·Wochenende), multiplikativ."""
    return C_IMPORT * 3.5 * np.exp(0.7 * np.asarray(zoll, float) + 0.3 * np.asarray(we, float))


def kunden_dummies(kunde) -> np.ndarray:
    """Elf Spalten (Kunde 1 bis 11; Kunde 0 ist die Referenz)."""
    return np.column_stack([(np.asarray(kunde) == k).astype(float) for k in range(1, K_KUNDEN)])


# --- Handbeispiel: neun erfundene Container --------------------------------------------------------------------------------------------
# vier ohne Zollhalt, fünf mit Zollhalt; Verweilzeiten so gewählt, dass alles in Brüchen aufgeht (nicht aus dem Generator gezogen)
HAND_D0 = (Fraction(16, 5), Fraction(9, 2), Fraction(51, 10), Fraction(83, 10))                          # 3,2 / 4,5 / 5,1 / 8,3 Tage
HAND_D1 = (Fraction(13, 2), Fraction(41, 5), Fraction(97, 10), Fraction(57, 5), Fraction(131, 10))      # 6,5 / 8,2 / 9,7 / 11,4 / 13,1 Tage


@dataclass(frozen=True)
class HandDaten:
    d: np.ndarray            # Verweilzeit in Tagen
    zoll: np.ndarray
    standtage: np.ndarray    # ⌈d⌉
    standgeld: np.ndarray    # d > 7

    @property
    def X(self) -> np.ndarray:
        return np.column_stack([np.ones(len(self.d)), self.zoll])

    @property
    def n(self) -> int:
        return len(self.d)


def _hand(d_list, zoll_list) -> HandDaten:
    d = np.array([float(x) for x in d_list])
    return HandDaten(d, np.array(zoll_list, float), np.array([math.ceil(x) for x in d_list], float), np.array([float(x > 7) for x in d_list]))


HAND_NEUN = _hand(HAND_D0 + HAND_D1, [0] * 4 + [1] * 5)
HAND_VIER = _hand(HAND_D0, [0] * 4)     # eine Gruppe: die vier Container ohne Zollhalt
