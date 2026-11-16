"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset eindeutig erkennbar; Permalink-Spezifikationen."""

import gfm_constants as C
import gfm_presets as P


def _full(preset):
    return tuple((k, preset.get(k, spec.default)) for k, spec in P.SETTING_SPECS.items())


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_options_or_inside_the_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            if spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_is_the_first_and_equals_the_default_settings():
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET and C.PRESETS[C.DEFAULT_PRESET] == {}


def test_every_preset_is_a_different_full_configuration():
    """Sonst würden zwei Presets zugleich als aktiv erkannt: aktives_preset() meldet das erste mit gleicher Konfiguration."""
    configs = [_full(p) for p in C.PRESETS.values()]
    assert len(set(configs)) == len(configs)


def test_setting_specs_cover_every_preset_key_and_have_valid_defaults():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, key
        if spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi, key
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)   # Adressparameter eindeutig


def test_every_preset_changes_something_visible():
    base = _full({})
    for name, preset in C.PRESETS.items():
        if name != C.DEFAULT_PRESET:
            assert _full(preset) != base, name


def test_pairs_cover_nine_combinations_with_three_per_outcome():
    assert len(C.PAIRS) == 9 and len(set(C.PAIR_KEYS)) == 9
    for outcome in C.OUTCOMES:
        assert sum(1 for p in C.PAIRS if p[1] == outcome) == 3
