"""Die GLM-Familie: Link, Deviance und ein IRLS-Code - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein IRLS-Code löst neun Familie/Link-Paare auf den drei Ausgängen der Container-Verweilzeit (stetig, Standtage, Standgeld). Fisher-Scoring und Newton gehen nur beim
kanonischen Link denselben Pfad. Der Link ist aus Daten nur lesbar, wenn das Merkmal stetig ist; die ungruppierte Residualdeviance ist kein Gütetest.
Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import gfm_constants as C
import gfm_evaluation as ev
import gfm_experiments as E
import gfm_frozen as F
from gfm_layout import dunkles_design, kopf_css
from gfm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
    SETTING_SPECS,
)
from gfm_visualization import (
    build_exp1,
    build_exp3_export,
    build_exp3_frozen,
    build_exp3_import,
    build_exp4_chi2,
    build_exp4_series,
    build_exp5,
)

st.set_page_config(page_title="Die GLM-Familie – Sebastian Hanisch", layout="wide")


# --- zwischengespeicherte Rechnungen (jede Live-Rechnung unter etwa 2 Sekunden) -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def _exp1(daten, paar, regel, start, halb, seed):
    return ev.run_exp1(daten, paar, regel, start, halb, seed)


@st.cache_data(show_spinner=False)
def _ref(daten, paar, seed):
    return ev.reference_for(daten, paar, seed)


@st.cache_data(show_spinner=False)
def _nine(seed):
    return ev.nine_pairs(seed)


@st.cache_data(show_spinner=False)
def _exp2(daten, ausgang, familie, link, seed):
    return ev.run_exp2(daten, ausgang, familie, link, seed)


@st.cache_data(show_spinner=False)
def _exp3(welt, n, seed):
    return ev.run_exp3_live(welt, n, seed)


@st.cache_data(show_spinner=False)
def _exp4(c, n, welt, modell, seed):
    return ev.run_exp4_live(c, n, welt, modell, seed)


@st.cache_data(show_spinner=False)
def _alpha(seed):
    return ev.nb2_alpha_live(seed)


@st.cache_data(show_spinner=False)
def _separation():
    return ev.separation_path()


def _select(label, key, labels=None, **kw):
    opts = list(options(key))
    return st.selectbox(label, opts, key=key, format_func=(lambda o: labels[o]) if labels else str, **kw)


def _radio(label, key, labels, **kw):
    return st.radio(label, list(options(key)), key=key, format_func=lambda o: labels[o], horizontal=True, **kw)


def _fmt(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


# --- Kopf und Einleitung ------------------------------------------------------------------------------------------------------------------
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Die GLM-Familie: Link, Deviance und ein IRLS-Code")
st.markdown(
    """
**Poisson mit Log-Link, Gamma mit Kehrwert, Logit:** wer so ein Modell nennt, hat **zwei Wahlen** getroffen. Die **Familie** legt fest, mit welchem Gewicht jeder Container in die Anpassung eingeht,
der **Link** legt fest, auf welcher Skala ein Merkmal wirkt. Hier läuft **ein** IRLS-Code (iterativ gewichtete kleinste Quadrate) auf allen drei Ausgängen der Container-Verweilzeit: stetig (Tage),
gezählt (Standtage) und binär (Standgeld). Er zeigt, wann Fisher-Scoring und Newton denselben Weg gehen (nur beim **kanonischen Link**), ob man den Link aus Daten lesen kann, und ob die **Residualdeviance** prüft,
ob das Modell passt. Konzept-Demo zu Regression und verallgemeinerten linearen Modellen; der Vorgänger fragt nur nach der Varianzfunktion bei festem Log-Link,
hier sind Familie **und** Link Parameter.

Weiter unten: **💬 Kernfragen**, sechs **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert ein verallgemeinertes lineares Modell", expanded=True):
    st.markdown(
        """
1. **Familie und Link wählen.** Die Familie nennt die Varianzfunktion V(μ) (Normal: 1, Poisson: μ, Binomial: μ(1 − μ), Gamma: μ², NB2: μ + αμ²), der Link verbindet das Mittel mit dem linearen Prädiktor, g(μ) = x'β.
2. **Arbeitsantwort und Gewicht bilden.** Aus dem aktuellen β folgt für jeden Container ein Mittel μ, ein Gewicht W = μ'(η)²/V(μ) und eine Arbeitsantwort z = η + (y − μ)/μ'(η).
3. **Gewichtet kleinste Quadrate lösen:** β ← (X'WX)⁻¹ X'Wz. Das ist ein Fisher-Scoring-Schritt; Newton nimmt statt der erwarteten die beobachtete Information.
4. **Wiederholen, bis der volle Schritt unter der Toleranz liegt.** Wird der Schritt halbiert, bleibt das Kriterium am vollen Schritt, sonst „konvergiert“ der Code mit Schrittweite null.
5. **Warum das funktioniert:** jede Iteration ist eine gewichtete lineare Regression (Wedderburn 1974); beim kanonischen Link sind erwartete und beobachtete Information gleich, der Pfad ist der von Newton.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Container", *bounds("seed_input"), key="seed_input", step=1,
                           help="Legt die Container-Tabelle (Experimente 1, 2, 6) und die Datensätze der Live-Rechnungen (Experimente 3 und 4) fest. Die vorgerechneten Messungen haben eigene feste Zufalls-Seeds.")
    st.button("🎲 Neue Container ziehen", width="stretch", on_click=randomize_seed)
    st.caption("Alle weiteren Regler stehen bei ihrem Experiment.")

seed = int(seed)
fz = F.load()

# --- Kernfragen ---------------------------------------------------------------------------------------------------------------------------
st.markdown("## 💬 Kernfragen")
nine = _nine(seed)
kern = ev.kern_answers(nine, fz)
st.markdown(
    f"""
**1. Welche zwei Wahlen stecken in „Poisson mit Log-Link“?** Die Familie (hier: Varianz gleich Mittel) bestimmt das Gewicht jedes Containers, der Link (hier: Log) die Skala des Effekts. {kern[0]}

**2. Woran sieht man, welcher Link stimmt?** {kern[1]}

**3. Sagt die Residualdeviance, ob das Modell passt?** {kern[2]}
"""
)
st.markdown("---")

# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Ein Code, neun Paare")
st.caption("Fisher-Scoring (IRLS) und Newton auf denselben Daten. Zähleinheit: ein Datensatz; die Schrittzahlen sind Messwerte dieser Daten, keine Konstanten. "
           "Voreinstellung: vier erfundene Container ohne Zollhalt (Standtage 4, 5, 6, 9), Poisson mit Identitäts-Link, Start β₀ = 5.")
c1, c2, c3 = st.columns(3)
with c1:
    _select("Daten", "e1_daten", C.HAND_DATEN, help="Von Hand: erfundene Container, bei denen sich alles in Brüchen nachrechnen lässt. Die Container-Tabelle hat 3 000 Container aller Arten und acht Merkmale.")
    _select("Start", "e1_start", C.START_OPTIONEN, help="Startmittel aus den Daten (binär (y + 0,5)/2, Zählung y + 0,5, sonst y) oder ein vorgegebener Koeffizientenvektor.")
with c2:
    paar_labels = {k: f"{C.FAMILY_LABEL[f]} / {C.LINK_LABEL[lk]}  ({C.OUTCOMES[a].split(' (')[0]})" for k, a, f, lk in C.PAIRS}
    _select("Familie und Link", "e1_paar", paar_labels, help="Neun Paare, drei je Ausgang. Kanonisch sind Normal/Identität, Poisson/Log, Gamma/Kehrwert und Binomial/Logit.")
    _select("Schrittregel", "e1_regel", C.REGEL_OPTIONEN, help="Fisher-Scoring nimmt die erwartete, Newton die beobachtete Information.")
with c3:
    st.toggle("Schritthalbierung", key="e1_halb", help="Wird die Deviance größer oder das Mittel ungültig, wird der Schritt halbiert (bis zu 30 Mal). Das Konvergenzkriterium bleibt am vollen Schritt.")

res1 = _exp1(st.session_state["e1_daten"], st.session_state["e1_paar"], st.session_state["e1_regel"], st.session_state["e1_start"], bool(st.session_state["e1_halb"]), seed)
cols = st.columns(max(len(res1["fits"]), 1) * 2)
for i, (mode, f) in enumerate(res1["fits"].items()):
    name = "Fisher" if mode == "fisher" else "Newton"
    cols[2 * i].metric(f"Schritte {name}", f"{f.iterations}" + ("" if f.converged else " (nicht konvergiert)"))
    cols[2 * i + 1].metric(f"Halbierungen {name}", f"{f.halvings}")
if res1["fits"]:
    st.plotly_chart(build_exp1(res1), width="stretch", key="chart_exp1")
    with st.expander("Koeffizienten je Schritt"):
        for mode, f in res1["fits"].items():
            df = pd.DataFrame(f.path, columns=list(res1["names"]))
            df.insert(0, "Schritt", range(len(df)))
            st.markdown(f"**{'Fisher-Scoring' if mode == 'fisher' else 'Newton'}**")
            st.dataframe(df.style.format({c: "{:.6f}" for c in res1["names"]}), hide_index=True, width="stretch")
kind, text = ev.verdict_exp1(res1, _ref(st.session_state["e1_daten"], st.session_state["e1_paar"], seed))
getattr(st, kind)(text)

st.markdown("**Alle neun Paare auf der Container-Tabelle** (n = 3 000, ein Datensatz, Toleranz 10⁻¹⁰ am vollen Schritt, in Klammern 10⁻⁸; Schrittzahlen sind Bänder):")
table = pd.DataFrame([{
    "Ausgang": r["ausgang"], "Paar": f"{C.FAMILY_LABEL[r['family']]} / {C.LINK_LABEL[r['link']]}", "kanonisch": "ja" if r["canonical"] else "nein",
    "Schritte Fisher": f"{r['fisher10']} ({r['fisher8']})", "Schritte Newton": f"{r['newton10']} ({r['newton8']})",
    "Pfadunterschied Fisher/Newton": ev.sci(r['path_diff']), "Unterschied der Grenzwerte": "–" if r["limit_diff"] is None else ev.sci(r['limit_diff']),
    "Gradient der unabhängigen Likelihood": "–" if r["grad"] is None else ev.sci(r['grad']), "Zoll-Koeffizient": _fmt(r["zoll"]), "kleinstes angepasstes Mittel": _fmt(r["mu_min"]),
} for r in nine])
st.dataframe(table, hide_index=True, width="stretch")
st.markdown("---")

# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Der Link ist die Skala")
st.caption("Dieselben Daten, anderer Link oder andere Familie. Voreinstellung: Container-Tabelle (n = 3 000, ein Datensatz), Standtage, Poisson mit Identitäts-Link. "
           "Eine Häufigkeit über Datensätze steht hier nicht, nur die Anpassung eines Datensatzes.")
d1, d2, d3, d4 = st.columns(4)
with d1:
    _select("Daten", "e2_daten", C.E2_DATEN)
with d2:
    _select("Ausgang", "e2_ausgang", C.E2_AUSGANG)
with d3:
    _select("Familie (Tabelle der Links)", "e2_familie", {f: C.FAMILY_LABEL[f] for f in C.E2_FAMILIEN}, help="Die Tabelle „Drei Links“ zeigt diese Familie mit Log-, Identitäts- und Kehrwert-Link.")
with d4:
    _select("Link (Tabelle der Familien)", "e2_link", {lk: C.LINK_LABEL[lk] for lk in C.E2_LINKS}, help="Die Tabelle „Drei Familien“ zeigt diesen Link mit Normal, Poisson (Quasi) und Gamma.")
res2 = _exp2(st.session_state["e2_daten"], st.session_state["e2_ausgang"], st.session_state["e2_familie"], st.session_state["e2_link"], seed)


def _table2(rows, first, first_label):
    out = []
    for r in rows:
        row = {first_label: first(r)}
        if not r["ok"]:
            row["Zoll-Koeffizient"] = "nicht gültig"
            out.append(row)
            continue
        row["Zoll-Koeffizient"] = _fmt(r["zoll"], 4)
        row["Lesart auf der Link-Skala"] = ev.scale_text(r["link"], r["zoll"])
        if r["we"] is not None:
            row["Wochenende-Koeffizient"] = _fmt(r["we"], 4)
        if "group_means" in r:
            gm = r["group_means"]
            row["angepasste Mittel (Tage)"] = _fmt(gm[0]) + ("" if gm[1] is None else f" und {_fmt(gm[1])}")
        else:
            row["kleinstes angepasstes Mittel"] = _fmt(r["mu_min"], 3)
            row["Mittel je Art: Σμ̂/Σy − 1 (Import, Export, Umschlag, Leer; %)"] = ", ".join(f"{b:+.2f}".replace(".", ",") for b in r["bias_art"])
        out.append(row)
    return pd.DataFrame(out)


t1, t2 = st.columns(2)
with t1:
    st.markdown(f"**Drei Links** ({C.FAMILY_LABEL[st.session_state['e2_familie']]})")
    st.dataframe(_table2(res2["links"], lambda r: C.LINK_LABEL[r["link"]], "Link"), hide_index=True, width="stretch")
with t2:
    st.markdown(f"**Drei Familien** ({C.LINK_ADJ[st.session_state['e2_link']]})")
    st.dataframe(_table2(res2["families"], lambda r: C.FAMILY_LABEL[r["family"]] + (" (Quasi)" if r["family"] == "poisson" else ""), "Familie"), hide_index=True, width="stretch")
kind, text = ev.verdict_exp2(res2)
getattr(st, kind)(text)
st.markdown("---")

# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Welcher Link? Mit stetigem Merkmal ab n = 50 meist lesbar, mit 0/1-Merkmalen nicht")
st.caption("Gamma-GLM mit drei Links, Wahl nach AIC (Gamma-Likelihood mit auf die Form profilierter Wahrscheinlichkeit, Parameter Mittel + 1). Zähleinheit: Wiederholung = ein Datensatz aus n Containern. "
           "Export: Vorlauf gleichverteilt in [1; 10] Tagen, das Mittel der Verweilzeit ist exakt linear (wahr: Identität). Import: nur Zoll und Wochenende, das Mittel ist multiplikativ (wahr: Log).")
e1c, e2c = st.columns(2)
with e1c:
    _radio("Welt", "e3_welt", C.E3_WELT)
with e2c:
    st.select_slider("Container je Datensatz n", options=list(options("e3_n")), key="e3_n")
welt3, n3 = st.session_state["e3_welt"], int(st.session_state["e3_n"])
live3 = _exp3(welt3, n3, seed)
row3 = ev.frozen_link_row(welt3, n3)
mcols = st.columns(3)
for col, link in zip(mcols, E.LINKS3):
    f = live3["fits"][link]
    col.metric(f"AIC, {C.LINK_ADJ[link]}", "nicht gültig" if f is None else ev.num(f['aic'], 1))
if welt3 == "export":
    st.plotly_chart(build_exp3_export(live3), width="stretch", key="chart_exp3_live")
    dfn = row3["defined"]["inverse"] if row3 else None
    st.caption("Grau gestrichelt: das wahre Mittel. Grün hinterlegt: der Datenbereich. Außerhalb davon laufen die drei Links auseinander; beim Kehrwert-Link bricht die Kurve ab, wo der lineare Prädiktor nicht mehr positiv ist (Lücke)."
               + (f" Vorgerechnet (n = {ev.ntxt(n3)}): eine definierte Vorhersage des Kehrwert-Links bei Vorlauf 15 / 20 Tagen haben {ev.pct(dfn[1])} / {ev.pct(dfn[2])} der Datensätze." if dfn else ""))
    rt = [(link, f["ratio"]) for link, f in live3["fits"].items() if f is not None]
    st.dataframe(pd.DataFrame([{"Link": C.LINK_LABEL[l], **{f"Vorlauf {v:g} Tage": ("nicht definiert" if r != r else _fmt(r)) for v, r in zip(C.E3_PUNKTE, rs)}} for l, rs in rt]),
                 hide_index=True, width="stretch")
    st.caption("Vorhersage geteilt durch das wahre Mittel an drei Punkten (in diesem Datensatz); 1,000 wäre genau.")
else:
    st.plotly_chart(build_exp3_import(live3), width="stretch", key="chart_exp3_live")
    cd = row3["median_celldev"] if row3 else None
    st.caption("Es gibt nur vier Zellen (Zoll × Wochenende), also keinen Bereich außerhalb der Daten, in dem sich die Links trennen könnten."
               + (f" Vorgerechnet (n = {ev.ntxt(n3)}): größte Zellabweichung μ̂/μ − 1, Median, bei Identität / Log / Kehrwert {ev.num(cd['identity'])} / {ev.num(cd['log'])} / {ev.num(cd['inverse'])}." if cd else ""))
kind, text = ev.verdict_exp3(welt3, n3, live3, row3)
getattr(st, kind)(text)
rows_all = fz["link_export" if welt3 == "export" else "link_import"]
st.plotly_chart(build_exp3_frozen(rows_all, welt3, n3), width="stretch", key="chart_exp3_frozen")
st.caption("Vorgerechnet (Datensätze je Zeile: " + ", ".join(f"n = {r['n']}: {r['reps']}" for r in rows_all) + "). Anteile sind Bänder, keine exakten Prozente.")
with st.expander("Mit diesen Einstellungen live wiederholen"):
    lim = ev.live_reps_limit(n3)
    reps3 = st.slider("Datensätze", C.LIVE_REPS_MIN, lim, min(lim, 40), key=f"live3_reps_{n3}")
    if st.button("▶ Wiederholen", key="live3_go"):
        with st.spinner(f"Rechnet {reps3} Datensätze mit je {n3} Containern (etwa {max(1, reps3 * n3 // 8000)} bis {max(2, reps3 * n3 // 4000)} Sekunden) …"):
            st.session_state["live3"] = {"key": (welt3, n3, seed, reps3), "res": ev.live_repeat_exp3(welt3, n3, reps3, seed)}
    got = st.session_state.get("live3")
    if got and got["key"] == (welt3, n3, seed, reps3):
        r = got["res"]
        w = r["wins"]
        st.success(f"{r['reps']} frische Datensätze, n = {n3}: der AIC wählt Identität / Log / Kehrwert in {ev.pct(w['identity'] / r['used'])} / {ev.pct(w['log'] / r['used'])} / {ev.pct(w['inverse'] / r['used'])} "
                   f"(Band: bei {r['used']} Datensätzen ist jeder Anteil bis zu ± {ev.pct(ev.band_pp(r['used']), 0)} unsicher, drei Standardfehler).")
st.markdown("---")

# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Deviance, kein Gütetest")
st.caption("Standgeld (Verweilzeit über c Tage) bei Import-Containern; die wahre Form ist ein Probit, exakt linear in Zoll, Wochenende und Kunde, die Anpassung nimmt den wahren Link. Getestet wird die Residualdeviance gegen χ² "
           "(Niveau 5 %). Zähleinheit: Wiederholung = ein Datensatz aus n Containern.")
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.select_slider("Standgeld ab … Tagen (c)", options=list(options("e4_c")), key="e4_c")
with f2:
    st.select_slider("Container n", options=list(options("e4_n")), key="e4_n")
with f3:
    _radio("Welt", "e4_welt", C.E4_WELT, help="Mit Kundeneffekt: der ln-Median hängt vom Kunden ab. Ohne: nur Zoll und Wochenende wirken; dann ist Modell M0 exakt richtig.")
with f4:
    _radio("Modell", "e4_modell", C.E4_MODELL, help="Das Modell M1 kennt die elf Kundenmerkmale, M0 nicht.")
c4, n4, w4, m4 = int(st.session_state["e4_c"]), int(st.session_state["e4_n"]), st.session_state["e4_welt"], st.session_state["e4_modell"]
live4 = _exp4(c4, n4, w4, m4, seed)
if live4.get("ok"):
    m = st.columns(4)
    m[0].metric("Residualdeviance D (ungruppiert)", ev.num(live4['du'], 1))
    m[1].metric("D/(n − p)", _fmt(live4["du"] / live4["dfu"]))
    m[2].metric("2·mittlere Entropie der wahren p", _fmt(live4["ent2"]))
    m[3].metric("95-%-Wert von χ²", ev.num(live4['crit_hi'], 1), delta=("D darüber: Modell verworfen" if live4["du"] > live4["crit_hi"] else "D darunter: nicht verworfen"), delta_color="off")
    st.plotly_chart(build_exp4_chi2(live4), width="stretch", key="chart_exp4_chi2")
    g = live4["g48"]
    gtxt = (f"gruppiert (Zellen Zoll × Wochenende × Kunde, {g['cells']} nicht leer, Freiheitsgrade {g['df']}): D = {ev.num(g['dev'], 1)}, Anteil Zellen mit erwarteter Zahl unter 5: {ev.pct(g['small'], 0)}"
            if g["df"] >= 1 else "gruppiert: keine Freiheitsgrade (so viele Parameter wie Zellen)")
    st.caption(f"Ungruppiert: Freiheitsgrade n − p = {live4['dfu']}. {gtxt}. Beziehung ungruppiert = gruppiert − 2·(gesättigte Log-Likelihood der Zellen), Abweichung {ev.sci(live4['relation_gap'])}.")
fz4 = ev.frozen_deviance_row(c4, n4, w4, m4)
kind, text = ev.verdict_exp4(live4, fz4)
getattr(st, kind)(text)
st.markdown("**Vorgerechnet: Welt ohne Kundeneffekt, Modell M0 exakt richtig, n = 1 050, Standgeld ab c Tagen**")
st.plotly_chart(build_exp4_series(fz["schwelle"]["rows"]), width="stretch", key="chart_exp4_series")
st.caption(f"{fz['schwelle']['reps']} Datensätze je Zeile, alle Schwellen am selben Datensatz. Die ungruppierte Deviance geteilt durch n − p folgt der doppelten mittleren Entropie der wahren Wahrscheinlichkeiten: "
           "gemessen ist diese Gleichheit auf höchstens 0,005 in allen acht Zeilen; die Deutung ist, dass die Häufigkeit des Ereignisses bestimmt, ob der Wert über oder unter dem χ²-Wert liegt, und nicht die Passung.")
with st.expander("Mit diesen Einstellungen live wiederholen"):
    lim4 = ev.live_reps_limit(n4)
    reps4 = st.slider("Datensätze", C.LIVE_REPS_MIN, lim4, min(lim4, 40), key=f"live4_reps_{n4}")
    if st.button("▶ Wiederholen", key="live4_go"):
        with st.spinner(f"Rechnet {reps4} Datensätze mit je {n4} Containern (etwa {max(1, reps4 * n4 // 8000)} bis {max(2, reps4 * n4 // 4000)} Sekunden) …"):
            st.session_state["live4"] = {"key": (c4, n4, w4, m4, seed, reps4), "res": ev.live_repeat_exp4(c4, n4, w4, m4, reps4, seed)}
    got = st.session_state.get("live4")
    if got and got["key"] == (c4, n4, w4, m4, seed, reps4):
        r = got["res"]
        if r["valid"]:
            st.success(f"{r['valid']} gültige Anpassungen von {r['reps']} frischen Datensätzen: ungruppierte Deviance über dem 95-%-Wert in {ev.pct(r['rej_du'])}, unter dem 5-%-Wert in {ev.pct(r['below_du'])}; "
                       f"mittleres D/(n − p) {_fmt(r['du_per_df'])}, 2·mittlere Entropie {_fmt(r['ent2'])}.")
        else:
            st.warning("Keine gültige Anpassung in diesen Datensätzen.")
st.markdown("---")

# --- Experiment 5 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 5: Was ein Test leisten kann: gruppiert und Likelihood-Quotient")
st.caption("Standgeld ab 7 Tagen, vorgerechnet; Zähleinheit: Wiederholung = ein Datensatz aus n Import-Containern, nur Datensätze mit gültiger Anpassung (Koeffizienten unter 15). Niveau 5 %.")
g1, g2 = st.columns(2)
with g1:
    st.select_slider("Container n", options=list(options("e5_n")), key="e5_n")
with g2:
    _radio("Link der Anpassung", "e5_link", {lk: C.LINK_LABEL[lk] for lk in C.E5_LINK}, help="Der wahre Link ist Probit; Logit und cloglog sind falsche Links bei richtigem Modell.")
n5, l5 = int(st.session_state["e5_n"]), st.session_state["e5_link"]
rows5 = ev.exp5_rows(n5, l5)
st.plotly_chart(build_exp5(rows5, n5), width="stretch", key="chart_exp5")
tab5 = []
for r in rows5:
    s = r["s"]
    tab5.append({"Szenario": r["name"], "gültig (von)": f"{s['valid']} (von {s['reps']})", "ungruppiert": ev.pct(s["rej_du"]), "Pearson": ev.pct(s["rej_xu"]),
                 "gruppiert, 48 Zellen": "–" if s["rej_dg48"] is None else ev.pct(s["rej_dg48"]), "gruppiert, 4 Zellen": ev.pct(s["rej_dg4"]) if "rej_dg4" in s else "–",
                 "Likelihood-Quotient": "–" if r["lq"] is None else ev.pct(r["lq"]["rej"]), "Zellen mit erwarteter Zahl < 5 (48 Zellen)": ev.pct(s["small48"], 0)})
st.dataframe(pd.DataFrame(tab5), hide_index=True, width="stretch")
kind, text = ev.verdict_exp5(rows5, n5, l5)
getattr(st, kind)(text)
st.markdown("---")

# --- Experiment 6 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 6: Wo der Rahmen reißt")
st.caption("Die GLM-Rechnung (gewichtete kleinste Quadrate in jedem Schritt) gilt für die neun Paare und, als äußere Schleife, für gestrafte Anpassungen. Hier steht, wo sie nicht reicht. Live gerechnet sind nur die ersten drei Zeilen.")
_select("Zeile", "e6_zeile", help="Wählt die Zeile, die darunter erklärt wird.")
rows6 = ev.exp6_table()
sel = st.session_state["e6_zeile"]
st.dataframe(pd.DataFrame(rows6), hide_index=True, width="stretch")
sep = _separation()
alpha = _alpha(seed)
if sel == "Separation":
    st.dataframe(pd.DataFrame([{"Schritte": r["k"], "Koeffizient von x": _fmt(r["beta1"], 2), "Deviance": _fmt(r["deviance"], 4)} for r in sep]), hide_index=True, width="stretch")
    st.caption("Vier Container, x = −2, −1, 1, 2, y = 0, 0, 1, 1 (vollständig getrennt): ohne Schritthalbierung wächst der Koeffizient mit jedem Schritt; der Wert, bei dem man aufhört, hängt an der Abbruchregel. Ab |η| = 30 greift die Rechengrenze.")
elif sel == "zweiter Parameter (NB2-α)":
    st.dataframe(pd.DataFrame([{"Verfahren": "Profil-Außenschleife (IRLS innen)", "α̂": _fmt(alpha["alpha_profile"], 5)},
                               {"Verfahren": f"abwechselnde Schätzung ({alpha['outer']} Außenschritte)", "α̂": _fmt(alpha["alpha_alternating"], 5)}]), hide_index=True, width="stretch")
    st.caption(f"Standtage der Container-Tabelle (Mittel {_fmt(alpha['mean'], 2)}, Varianz {_fmt(alpha['var'], 2)}). Das IRLS schätzt α nicht; es löst die Score-Gleichung für β bei festem α.")
elif sel == "nicht kanonischer Link":
    pi = next(r for r in nine if r["key"] == "poisson/identity")
    st.caption(f"Poisson mit Identitäts-Link (nicht kanonisch), Standtage der Container-Tabelle: Fisher-Scoring {pi['fisher10']} Schritte, Newton {pi['newton10']} Schritte (Toleranz 10⁻¹⁰); Experiment 1 zeigt alle neun Paare.")
else:
    st.caption("Für diese Zeile steht hier nur die Einordnung mit Quelle und Stufe, keine eigene Rechnung dieser Demo.")
kind, text = ev.verdict_exp6(sep, alpha)
getattr(st, kind)(text)
st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wo steht es |
|---|---|---|
| **Familie und Link stehen fest** | Beide sind Wahlen mit Folgen: der Link ändert die Skala des Effekts und die Vorhersage außerhalb der Daten, die Familie die Gewichtung. Aus zwei Gruppenmitteln ist der Link nicht wählbar. | Experimente 2 und 3 |
| **Das Mittelwertmodell passt zur Link-Skala** | Auf dieser Tabelle ist das Mittel der Verweilzeit nicht exakt log-linear; je Familie bleibt eine andere Abweichung. Der AIC wählt einen Link, nicht das richtige Modell. | Experimente 2 und 3 |
| **Die Residualdeviance prüft das Modell** | Beim Logit ist sie bei 0/1-Ausgang eine Funktion der angepassten Wahrscheinlichkeiten; auch beim Probit folgt sie in dieser Demo der mittleren Entropie (D/(n − p) gleich 2·H̄ auf höchstens 0,005), nicht der Passung. | Experiment 4 |
| **Gruppieren macht daraus einen Test** | Nur bei genügend großen Zellen; bei kleinen Zellen hält der Test sein Niveau nicht. Ein falscher Link (Logit statt Probit) ist kaum zu trennen. | Experiment 5 |
| **IRLS gilt immer** | Nicht bei Separation, nicht für einen zweiten Parameter ohne Außenschleife, nicht für Quantilregression, Kalibrierung und gemischte Modelle. | Experiment 6 |
| **Die Daten sind die des Vehikels** | Alle Zahlen gelten für synthetische Container (erfundene Annahmen), zwölf feste Kundeneffekte und die gewählten Parameter, nicht für Daten eines Hafens. | README |
"""
)
st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Beobachtungen $y_i$ ($i = 1,\dots,n$) und Merkmale $x_i$; gesucht ist $\beta$. Der Rest ist Notation.

| Symbol | Bedeutung |
|---|---|
| $\mu_i$, $\eta_i = x_i^\top\beta$ | Mittel von $y_i$ und linearer Prädiktor |
| $g$, $h = g^{-1}$ | Link und Umkehrfunktion, $\mu_i = h(\eta_i)$, $\mu'_i = h'(\eta_i)$ |
| $V(\mu)$, $\phi$ | Varianzfunktion der Familie und Dispersion, $\operatorname{Var}(y_i) = \phi\,V(\mu_i)$ |
| $W$, $U$ | Gewichtsmatrix und Score |
| $D$, $X^2$ | Residualdeviance (Summe) und Pearson-Statistik |

**Modell.** $g(\mu_i) = x_i^\top\beta$, $\operatorname{Var}(y_i) = \phi\,V(\mu_i)$. Familien dieser Demo: Normal $V = 1$, Poisson $V = \mu$, Binomial (Bernoulli) $V = \mu(1-\mu)$, Gamma $V = \mu^2$, NB2 mit festem $\alpha$: $V = \mu + \alpha\mu^2$.

**IRLS (Fisher-Scoring).** Mit $W_i = \mu_i'^2 / V(\mu_i)$, Score $U = X^\top\big((y-\mu)\,\mu'/V\big)$ und Arbeitsantwort $z_i = \eta_i + (y_i - \mu_i)/\mu'_i$:

$$\beta \leftarrow \beta + (X^\top W X)^{-1} U \;=\; (X^\top W X)^{-1} X^\top W z .$$

**Newton.** Die beobachtete Information ersetzt $W_i$ durch

$$W_i - (y_i-\mu_i)\,\frac{\mu''_i\,V(\mu_i) - \mu_i'^2\,V'(\mu_i)}{V(\mu_i)^2} .$$

Beim **kanonischen Link** (Normal/Identität, Binomial/Logit, Poisson/Log, Gamma/Kehrwert) ist dieser Zusatz null: erwartete und beobachtete Information sind gleich, der Pfad ist derselbe.

**Einheitsdeviance** (Summe $= D$, Vorgewicht 1, Dispersion nicht eingerechnet):

$$d_N = (y-\mu)^2,\quad d_P = 2\Big[y\ln\tfrac{y}{\mu} - (y-\mu)\Big],\quad d_B = 2\Big[y\ln\tfrac{y}{\mu} + (1-y)\ln\tfrac{1-y}{1-\mu}\Big],$$

$$d_\Gamma = 2\Big[-\ln\tfrac{y}{\mu} + \tfrac{y-\mu}{\mu}\Big],\quad d_{NB2} = 2\Big[y\ln\tfrac{y}{\mu} - \big(y+\tfrac1\alpha\big)\ln\tfrac{1+\alpha y}{1+\alpha\mu}\Big].$$

Pearson: $X^2 = \sum_i (y_i-\mu_i)^2 / V(\mu_i)$. AIC $= -2\ell + 2\,(\text{Zahl der Parameter})$, bei Gamma mit $+1$ für die auf die Form profilierte Likelihood und bei NB2 mit $+1$ für $\alpha$.

**Bernoulli-Deviance und Entropie.** Mit $\hat p_i = \mu_i$ ist $D = -2\sum_i\big[y_i\ln\hat p_i + (1-y_i)\ln(1-\hat p_i)\big]$. Beim Logit gilt $D = -2\sum_i\big[\hat p_i\hat\eta_i + \ln(1-\hat p_i)\big]$: $D$ hängt nur von den angepassten Wahrscheinlichkeiten ab, nicht von den $y_i$. Sind die $\hat p_i$ die wahren, ist $D/n$ nahe $2\bar H$ mit der mittleren Entropie $\bar H = \tfrac1n\sum_i H(p_i)$, $H(p) = -p\ln p - (1-p)\ln(1-p)$.

**Gruppiert und ungruppiert.** Die ungruppierte Deviance nimmt „jeder Container eine eigene Wahrscheinlichkeit“ als gesättigtes Modell (Log-Likelihood 0), die gruppierte „jede Zelle gleicher Merkmale eine eigene Wahrscheinlichkeit“ mit Log-Likelihood $\ell_{\mathrm{sat}}$ der Zellen. Es gilt $D_{\text{ungruppiert}} = D_{\text{gruppiert}} - 2\,\ell_{\mathrm{sat}}$.

**Likelihood-Quotienten-Test.** Für ein eingebettetes Modell ist $D_{M0} - D_{M1}$ gegen $\chi^2_{p_1 - p_0}$ zu halten.

| Konvention | Wert in dieser Demo |
|---|---|
| Deviance | Summe, nicht Mittel |
| Start | Startmittel aus den Daten (binär $(y+0{,}5)/2$, Zählung $y+0{,}5$, sonst $y$), das Handbeispiel $\mu_0 = y$ |
| Konvergenz | Betragsmaximum des vollen Schritts unter $10^{-10}$ |
| Logit | Kappung $\lvert\eta\rvert \le 30$ wie in R, eine Rechengrenze |
| Probit, cloglog | Wahrscheinlichkeiten auf $[10^{-15};\,1 - 10^{-15}]$ gekappt |
| Zellen für „gruppiert“ | Zoll × Wochenende × Kunde (48) bzw. Zoll × Wochenende (4) |

**Literatur.** Wedderburn, R. W. M. (1974): Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method, Biometrika 61 (Einleitung: jede Iteration ist eine gewichtete lineare Regression). Hastie, Tibshirani, Friedman: The Elements of Statistical Learning, Abschnitt 4.4.1 (Seiten 120 und 121: IRLS beim Logit gleich Newton, Schrittweitenhalbierung, Konvergenz nicht garantiert).
James, Witten, Hastie, Tibshirani: An Introduction to Statistical Learning with Applications in R, 2. Auflage, Abschnitt 4.6.3 (Link-Funktion). Was McCullagh/Nelder (1989) zu Deviance und Dispersion sagen, ist hier nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html)."
)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})
