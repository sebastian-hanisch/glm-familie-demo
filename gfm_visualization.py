"""Plotly-Abbildungen (alle Achsen fest, `fixedrange`, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import gfm_constants as C

LINK_NAME = C.LINK_LABEL
MODE_NAME = {"fisher": "Fisher-Scoring (IRLS)", "newton": "Newton"}


def _fix(fig, height=C.CHART_HEIGHT, **kw):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin={"l": 55, "r": 15, "t": 40, "b": 50}, legend={"orientation": "h", "y": -0.2}, **kw)
    return fig


def build_exp1(res: dict) -> go.Figure:
    """Abstand der Koeffizienten zum Grenzwert und Betragsmaximum des Scores je Schritt (logarithmisch)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abstand zum Grenzwert", "Größter Betrag des Scores"), horizontal_spacing=0.12)
    for mode, f in res["fits"].items():
        col = C.COLORS[mode]
        d = res["distance"].get(mode)
        if d is not None:
            fig.add_trace(go.Scatter(x=list(range(len(d))), y=np.maximum(d, 1e-17), mode="lines+markers", name=MODE_NAME[mode], line={"color": col}, legendgroup=mode), row=1, col=1)
        if len(f.scores):
            fig.add_trace(go.Scatter(x=list(range(len(f.scores))), y=np.maximum(f.scores, 1e-17), mode="lines+markers", name=MODE_NAME[mode], line={"color": col},
                                     legendgroup=mode, showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title_text="Schritt", rangemode="tozero")
    return _fix(fig)


def build_exp3_export(live: dict) -> go.Figure:
    v, d = live["data"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=d, mode="markers", name="Container (Verweilzeit)", marker={"color": "rgba(150,150,150,0.5)", "size": 5}))
    fig.add_trace(go.Scatter(x=live["grid"], y=live["truth"], mode="lines", name="wahres Mittel (linear)", line={"color": C.COLORS["truth"], "dash": "dash", "width": 3}))
    for link, f in live["fits"].items():
        if f is not None:
            fig.add_trace(go.Scatter(x=live["grid"], y=f["curve"], mode="lines", name=f"Gamma, {C.LINK_ADJ[link]}", line={"color": C.COLORS[link]}))
    fig.add_vrect(x0=1, x1=10, fillcolor="rgba(46,125,79,0.08)", line_width=0, annotation_text="Datenbereich", annotation_position="top left")
    fig.update_xaxes(title_text="Vorlauf (Tage)", range=[0, 25])
    fig.update_yaxes(title_text="Verweilzeit (Tage)", range=[0, 60])
    return _fix(fig)


def build_exp3_import(live: dict) -> go.Figure:
    labels = ["kein Zoll, Wochentag", "kein Zoll, Wochenende", "Zoll, Wochentag", "Zoll, Wochenende"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=live["truth"], name="wahres Mittel", marker_color=C.COLORS["truth"], opacity=0.55))
    for link, f in live["fits"].items():
        if f is not None:
            fig.add_trace(go.Scatter(x=labels, y=f["cell_mu"], mode="markers", name=f"Gamma, {C.LINK_ADJ[link]}", marker={"color": C.COLORS[link], "size": 12, "symbol": "diamond"}))
    fig.update_yaxes(title_text="mittlere Verweilzeit (Tage)", rangemode="tozero")
    return _fix(fig)


def build_exp3_frozen(rows: list[dict], welt: str, n_now: int) -> go.Figure:
    fig = go.Figure()
    xs = [f"n = {r['n']}" for r in rows]
    for link in ("identity", "log", "inverse"):
        fig.add_trace(go.Bar(x=xs, y=[100 * r["wins"][link] / r["used"] for r in rows], name=f"AIC wählt {LINK_NAME[link]}", marker_color=C.COLORS[link]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Anteil der Datensätze (%)", range=[0, 100])
    fig.add_annotation(x=f"n = {n_now}", y=104, text="▼ eingestellt", showarrow=False)
    return _fix(fig)


def build_exp4_chi2(live: dict) -> go.Figure:
    df = live["dfu"]
    sd = (2 * df) ** 0.5
    x = np.linspace(max(0.0, df - 5 * sd), df + 5 * sd, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=stats.chi2.pdf(x, df), mode="lines", name=f"χ² mit {df} Freiheitsgraden", line={"color": C.COLORS["truth"]}))
    fig.add_vline(x=live["crit_hi"], line={"color": "#c4452d", "dash": "dash"}, annotation_text="95-%-Wert", annotation_position="top right")
    fig.add_vline(x=live["crit_lo"], line={"color": "#2a6fb0", "dash": "dash"}, annotation_text="5-%-Wert", annotation_position="top left")
    fig.add_vline(x=live["du"], line={"color": "#2e7d4f", "width": 3}, annotation_text="Deviance dieses Datensatzes", annotation_position="bottom right")
    fig.update_xaxes(title_text="Residualdeviance (ungruppiert)")
    fig.update_yaxes(title_text="Dichte", rangemode="tozero")
    return _fix(fig)


def build_exp4_series(rows: list[dict]) -> go.Figure:
    """D/(n − p) und 2·mittlere Entropie je Standgeld-Schwelle, dazu Verwerfungsanteile."""
    cs = [f"{r['c']} Tage" for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("D/(n − p) gegen 2·mittlere Entropie", "Verwerfung bei richtigem Modell (%)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=cs, y=[r["du_per_df"] for r in rows], mode="lines+markers", name="D/(n − p)", line={"color": "#2a6fb0"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=cs, y=[r["ent2"] for r in rows], mode="lines+markers", name="2·mittlere Entropie der wahren p", line={"color": "#c77700", "dash": "dot"}), row=1, col=1)
    fig.add_hline(y=1.0, line={"color": C.COLORS["truth"], "dash": "dash"}, row=1, col=1, annotation_text="1 (erwartet bei Passung?)")
    fig.add_trace(go.Bar(x=cs, y=[100 * r["rej_hi"] for r in rows], name="über dem 95-%-Wert", marker_color="#c4452d"), row=1, col=2)
    fig.add_trace(go.Bar(x=cs, y=[100 * r["below_lo"] for r in rows], name="unter dem 5-%-Wert", marker_color="#2a6fb0"), row=1, col=2)
    fig.add_trace(go.Bar(x=cs, y=[100 * r["rej_g4"] for r in rows], name="gruppiert (4 Zellen) über dem 95-%-Wert", marker_color="#2e7d4f"), row=1, col=2)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Standgeld ab …")
    return _fix(fig)


def build_exp5(rows: list[dict], n: int) -> go.Figure:
    tests = [("rej_du", "ungruppiert"), ("rej_xu", "Pearson (ungruppiert)"), ("rej_dg48", "gruppiert, 48 Zellen"), ("rej_dg4", "gruppiert, 4 Zellen")]
    fig = go.Figure()
    names = [r["name"].split(" (")[0] for r in rows]
    for key, label in tests:
        ys = [None if r["s"].get(key) is None else 100 * r["s"][key] for r in rows]
        fig.add_trace(go.Bar(x=names, y=ys, name=label))
    ys = [None if r["lq"] is None else 100 * r["lq"]["rej"] for r in rows]
    fig.add_trace(go.Bar(x=names, y=ys, name="Likelihood-Quotienten-Test M0 gegen M1", marker_color="#2e7d4f"))
    fig.add_hline(y=5, line={"color": C.COLORS["truth"], "dash": "dash"}, annotation_text="Niveau 5 %")
    fig.update_yaxes(title_text="Anteil verworfen (%)", range=[0, 100])
    fig.update_layout(barmode="group")
    return _fix(fig, height=C.CHART_HEIGHT + 60)
