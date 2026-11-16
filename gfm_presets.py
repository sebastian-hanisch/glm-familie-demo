"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler der App steht in SETTING_SPECS (Seitenleiste und Hauptbereich): dann hebt seine Änderung das aktive Preset auf, und die Adresszeile spiegelt die
Konfiguration. Ein Preset setzt **alle** Regler: die genannten auf ihren Wert, die übrigen auf den Standardwert."""

import random
from dataclasses import dataclass

import streamlit as st

import gfm_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


def _bool(text) -> bool:
    return str(text).lower() in ("1", "true", "wahr", "ja")


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "e1_daten": SettingSpec("e1d", str, "hand4", options=tuple(C.HAND_DATEN)),
    "e1_paar": SettingSpec("e1p", str, "poisson/identity", options=C.PAIR_KEYS),
    "e1_regel": SettingSpec("e1r", str, "beide", options=tuple(C.REGEL_OPTIONEN)),
    "e1_start": SettingSpec("e1s", str, "fuenf", options=tuple(C.START_OPTIONEN)),
    "e1_halb": SettingSpec("e1h", _bool, False),
    "e2_daten": SettingSpec("e2d", str, "tabelle", options=tuple(C.E2_DATEN)),
    "e2_ausgang": SettingSpec("e2a", str, "zähl", options=tuple(C.E2_AUSGANG)),
    "e2_familie": SettingSpec("e2f", str, "poisson", options=C.E2_FAMILIEN),
    "e2_link": SettingSpec("e2l", str, "identity", options=C.E2_LINKS),
    "e3_welt": SettingSpec("e3w", str, "export", options=tuple(C.E3_WELT)),
    "e3_n": SettingSpec("e3n", int, 1000, options=C.E3_N),
    "e4_c": SettingSpec("e4c", int, 7, options=C.E4_SCHWELLEN),
    "e4_n": SettingSpec("e4n", int, 350, options=C.E4_N),
    "e4_welt": SettingSpec("e4w", str, "mit", options=tuple(C.E4_WELT)),
    "e4_modell": SettingSpec("e4m", str, "M0", options=tuple(C.E4_MODELL)),
    "e5_n": SettingSpec("e5n", int, 350, options=C.E5_N),
    "e5_link": SettingSpec("e5l", str, "probit", options=C.E5_LINK),
    "e6_zeile": SettingSpec("e6z", str, C.E6_ZEILEN[0], options=C.E6_ZEILEN),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key):
    return SETTING_SPECS[state_key].options


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf zulässige Auswahlwerte eingerastet (sonst ignoriert)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None and value not in spec.options:
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt alle Regler: die Werte des Presets, für nicht genannte Regler den Standardwert."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos; verglichen wird jeder Regler aus SETTING_SPECS)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
