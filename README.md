# Die GLM-Familie: Link, Deviance und ein IRLS-Code – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-glm-familie-demo.streamlit.app/)** -->

Konzept-Demo zu **Regression und verallgemeinerten linearen Modellen** (GLM) auf der Container-Verweilzeit (synthetisch, erfunden), Vorgänger in der Kette der Annahmen ist
[`zaehldaten-demo`](https://github.com/sebastian-hanisch/zaehldaten-demo): dort ist der Log-Link gesetzt und nur die Varianzfunktion wird getauscht (Poisson, Quasi-Poisson, NB2). Hier wird die gesetzte Annahme aufgehoben:
**Familie und Link sind zwei Wahlen mit sichtbaren Folgen.** Ein IRLS-Code löst neun Familie/Link-Paare auf den drei Ausgängen der Container-Tabelle (stetig: Verweilzeit; gezählt: Standtage;
binär: Standgeld ab 7 Tagen), und die Frage „passt das Modell?“ wird mit der Residualdeviance gestellt, die sie im Container-Vehikel bei 0/1-Ausgang ohne Gruppierung nicht beantwortet (Zähleinheit: Datensatz).
Nachbar-Demos: [`logit-separation-demo`](https://github.com/sebastian-hanisch/logit-separation-demo) (Logit, Existenz des Schätzers, Firth), [`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo)
(Standgeld als Entscheidung mit Kosten), [`modellwahl-demo`](https://github.com/sebastian-hanisch/modellwahl-demo) (AIC, BIC, Kreuzvalidierung für Polynomgrade), [`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo)
(Gewichte aus der Streuung statt aus dem Mittel), [`kollinearitaet-auswahl-demo`](https://github.com/sebastian-hanisch/kollinearitaet-auswahl-demo) (gestrafte Anpassungen), [`cluster-gemischt-demo`](https://github.com/sebastian-hanisch/cluster-gemischt-demo)
(gemischte Modelle), [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo) (der Generator, den diese Demo nachbaut).

## Warum dieses Problem

Wer „Poisson mit Log-Link“ oder „Gamma mit Kehrwert“ sagt, hat zwei Wahlen getroffen. Die **Familie** legt über die Varianzfunktion fest, mit welchem Gewicht jeder Container in die Anpassung eingeht;
der **Link** legt fest, auf welcher Skala ein Merkmal wirkt und was das Modell außerhalb der Daten vorhersagt. Drei Kernfragen stehen in der App vor den Experimenten: (1) Welche zwei Wahlen stecken in „Poisson mit Log-Link“?
(2) Woran sieht man, welcher Link stimmt? (3) Sagt die Residualdeviance, ob das Modell passt?

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die Zahlen dieser README stammen aus **dieser** Demo (eigener Generator mit dem exakten SplitMix64, nicht bitgleich zur `verweilzeit-demo`) und stehen in den Tests; die erste Rechnung mit `numpy`-Zufall hatte andere Datensätze und liegt
nur in einigen Zahlen daneben. Was sich verschoben hat:

- **Schrittzahlen sind Messwerte eines Datensatzes.** Poisson mit Identitäts-Link: Fisher-Scoring braucht auf der Tabelle mit Zufalls-Seed 180 hier 103 Schritte (Toleranz 10⁻¹⁰), die erste Rechnung hatte 45; über 20 Tabellen (Zufalls-Seeds 300 bis 319)
  liegt der Wert zwischen 35 und 157, Newton braucht 6 oder 7. Die Aussage „Fisher braucht ein Mehrfaches von Newton“ gilt als Band, nicht als Zahl. Die Zahlen der ersten Rechnung stehen in `out_rn_v18_glm_familie.txt` und `out_rn_v2.txt` (Messdateien, nicht Teil dieses Repos).
- **Fisher-Scoring konvergiert bei Poisson/Identität nicht immer.** In Tabellen mit n = 250 (20 Instanzen, Zufalls-Seeds 300 bis 319) brauchten vier von zwanzig mehr als 5 000 Schritte (Toleranz 10⁻¹¹; die Zählung gilt für dieses Schrittbudget und diese Toleranz) (im Beispiel Zufalls-Seed 300 liegt das kleinste angepasste Mittel bei 0,47); Newton konvergierte in allen 20 (Tests: `test_eigenschaften_glm_familie`, `test_claims`: `test_b11_fisher_creeps_in_small_tables`).
- **Ungruppierte Deviance bei fehlendem Kundeneffekt:** die erste Rechnung fand 2,2 / 4,0 / 9,5 % Verwerfung bei n = 350 / 1 050 / 3 500; hier sind es 2,2 / 5,0 / 13,2 % (250 Datensätze bei n = 3 500, Standardfehler 2,1 Prozentpunkte). Die Aussage, die ungruppierte Deviance erkenne das fehlende Kundenmerkmal bei n = 350 und 1 050 nicht (Verwerfung 2,2 % und 5,0 % bei Niveau 5 %), bleibt.
- **Link-Wahl im Export:** AIC wählt den wahren Identitäts-Link hier in 81,3 / 91,5 / 99,9 % bei n = 20 / 50 / 200 (erste Rechnung 80,1 / 90,8 / 99,8 %); im Import den wahren Log-Link in 14,4 % bei n = 50 (erste Rechnung 14,7 %).
- **Das 49,2-%-Ergebnis** einer früheren Messung (Bernoulli, ungruppiert, n = 200, p = 2, logit p = −0,5 + x mit x ~ N(0; 1)) gilt nur für diese Konfiguration; mit dem Zufall dieser Demo (300 Datensätze) liegt der Anteil bei 52,3 %.
  Im Container-Vehikel liegt die ungruppierte Deviance bei richtigem Modell und Schwelle 7 Tage in 53,0 % der Datensätze unter dem 5-%-Wert, bei Schwelle 4 Tage in 100,0 % über dem 95-%-Wert (Experiment 4); die Deutung (nicht gemessen) ist, dass die Häufigkeit des Ereignisses entscheidet, nicht die Passung.
- **Vorhersagepunkt und Schwelle sind keine Regler mehr, wo sie ins Leere liefen:** der Vorhersagepunkt im Import hätte keinen Bereich außerhalb der Daten gehabt; die Vorhersage steht an drei festen Punkten (Vorlauf 5,5 / 15 / 20 Tage).
- **Wiederholungen:** die vorgerechneten Messungen haben 400 / 400 / 250 Datensätze (Experiment 5) statt 600 / 600 / 400, um die Rechenzeit klein zu halten; die Bänder sind entsprechend breiter.

## Modell

Beobachtung y mit Mittel μ und Varianz φ·V(μ); linearer Prädiktor η = x'β; Link g(μ) = η. IRLS-Schritt (Fisher-Scoring): W = μ'²/V(μ), Arbeitsantwort z = η + (y − μ)/μ', β ← (X'WX)⁻¹X'Wz; gleichbedeutend β ← β + (X'WX)⁻¹U mit dem Score U.
**Newton** nimmt die beobachtete statt der erwarteten Information. Beim **kanonischen Link** (Normal/Identität, Binomial/Logit, Poisson/Log, Gamma/Kehrwert) sind beide gleich, der Pfad ist derselbe. Familien: Normal, Poisson, Binomial (Bernoulli), Gamma, NB2 mit festem α.
Die Einheitsdeviance (Summe = Deviance D) und die Konventionen (Deviance ist eine Summe, ungruppiert gegen gruppiert, Start, Konvergenz am vollen Schritt, AIC mit +1 Parameter bei Gamma und NB2) stehen im Expander „📐 Mathematische Formulierung“.

**Vehikel (alle Zahlen gelten für dieses Vehikel und diese Parameter).** Container-Verweilzeit, lognormal, synthetisch: ln d ~ N(μ, s²); Import: ln-Median = ln 3,5 + 0,7·Zoll + 0,3·Wochenende + Kundeneffekt, s = 0,55; Export: ln-Median = ln(0,6 + 0,85·Vorlauf) + 0,3·Kundeneffekt, s = 0,25;
Umschlag und Leer analog (siehe `gfm_scenario.py`); zwölf feste Kundeneffekte (realisierte Standardabweichung 0,379). Standtage sind ⌈d⌉, Standgeld ist d > 7 Tage. Zwei Teile sind exakt bekannt und darum die Prüfsteine: das Mittel im **Export** ist linear im Vorlauf
(Faktor 1,0294), das Standgeld im **Import** ist ein Probit, exakt linear in Zoll, Wochenende und Kunde. Die drei Ausgänge sind Vergröberungen einer Zufallsgröße, keine unabhängigen Aufgaben.

## Methodik

Sechs Experimente, jedes mit Verdikt: (1) ein Code für neun Paare, Fisher gegen Newton, mit unabhängiger Referenz; (2) derselbe Datensatz, anderer Link oder andere Familie; (3) Link-Wahl mit dem AIC auf Export (stetiges Merkmal, wahr: Identität) und Import (0/1-Merkmale, wahr: Log);
(4) Residualdeviance beim Standgeld, Schwelle in Tagen; (5) gruppierte Deviance, Pearson und Likelihood-Quotienten-Test; (6) wo die Rechnung nicht mehr gilt. Die Experimente 3 bis 5 sind **vorgerechnet** (`data/gfm_frozen.json`, erzeugt von `tools/gfm_make_frozen.py`, rund 10 Minuten) und in der App so
gekennzeichnet; jedes hat daneben eine Live-Rechnung auf einem Datensatz und, mit Knopf und Zeitangabe, eine kurze Live-Wiederholung. **Zähleinheit überall: Wiederholung = ein Datensatz aus n Containern**, wo nicht „Container“ steht. Anteile über Datensätze sind Bänder mit Monte-Carlo-Fehler.
Zufall nur aus dem exakten SplitMix64 (`gfm_rng.py`, Kopie der geprüften Vorlage des Portfolios).

## Befunde (gemessen, keine Behauptungen)

Alle Messreihen: Container der Demo, Zähleinheit wie angegeben, Zufalls-Seeds in `gfm_experiments.seed_of`. „Test“ nennt die Datei und, wo vorhanden, die Testfunktion, die die Zahl prüft (exakt nur Brüche und geschlossene Formeln, sonst Bänder).

### Handbeispiel (neun erfundene Container, in Brüchen nachgerechnet)

Vier Container ohne Zollhalt (Verweilzeit 3,2 / 4,5 / 5,1 / 8,3 Tage), fünf mit Zollhalt (6,5 / 8,2 / 9,7 / 11,4 / 13,1 Tage); Standtage ⌈d⌉ = 4 / 5 / 6 / 9 und 7 / 9 / 10 / 12 / 14; Standgeld (d > 7) 0 / 0 / 0 / 1 und 0 / 1 / 1 / 1 / 1.

| Befund | Wert | Test |
|---|---|---|
| H1 Logit auf dem Standgeld: erster IRLS-Schritt von β = 0 | exakt (−1; 11/5) | `test_algorithm`: `test_logit_first_step_…` |
| H2 Grenzwert in geschlossener Form; Fisher-Pfad = Newton-Pfad | β = (ln 1/3; ln 12) = (−1,098612; 2,484907); Pfadunterschied unter 10⁻¹²; 5 bis 10 Schritte | `test_logit_limit_in_closed_form_…` |
| H3 Deviance des Logit-Modells | D = 10 ln 5 − 6 ln 3 = 9,5027054, **dieselbe für jede Verteilung der Ereignisse**, die 1 von 4 und 4 von 5 behält, und gleich 2·Σ H(p̂) | `test_bernoulli_deviance_is_…` |
| H4 Poisson/Identität auf den Standtagen: Gruppenmittel | 6 und 52/5; Grenzwert β = (6; 22/5), also 4,4 Tage mehr durch Zoll | `test_poisson_identity_group_means_…` |
| H5 Start μ₀ = y: Schritt 1, Schritt 2 | harmonisches Mittel je Gruppe (720/131 und 6300/641), dann das arithmetische (6; 4,4) | `test_poisson_identity_group_means_…` |
| H6 eine Gruppe (4, 5, 6, 9), Start β₀ = 5 | Fisher-Scoring trifft 6 in **einem** Schritt; Newton-Fehler 1, 1/6, 1/216, 1/279 936 (e_(k+1) = e_k²/6) und erreicht 6 in exakter Rechnung nie (in Gleitkomma ist der Fehler nach wenigen Schritten bis auf Rundung null) | `test_one_group_start_five_…` |
| H7 Gamma, drei Links, zwei Gruppen | alle drei Links geben die Gruppenmittel 211/40 = 5,275 und 489/50 = 9,78; Koeffizienten Log (1,6630; 0,6174), Identität (5,275; 4,505), Kehrwert (0,18957; −0,08732) | `test_gamma_two_groups_…` |
| H8 AIC bei zwei Gruppen | gleich bei allen drei Links: aus zwei Gruppenmitteln ist der Link nicht wählbar | `test_eigenschaften_glm_familie`: `test_aic_of_two_links_…` |

### Experiment 1: ein Code, neun Paare (Container-Tabelle, n = 3 000, Zufalls-Seed 180, ein Datensatz)

Schritte bei Toleranz 10⁻¹⁰ (in Klammern 10⁻⁸); Schrittzahlen sind Messwerte dieses Datensatzes, Bänder in den Tests. „Referenz“: der Gradient der unabhängigen Likelihood (`scipy.stats`) am IRLS-Ergebnis liegt bei höchstens 2,2·10⁻⁶ (Band im Test: unter 10⁻⁵; die Plattformen unterscheiden sich im Rauschen der Differenzen).

| Ausgang | Paar | kanonisch | Fisher | Newton | Pfadunterschied | Zoll-Koeffizient |
|---|---|---|---|---|---|---|
| stetig | Normal/Identität | ja | 1 (1) | 1 (1) | 0 | +5,242 |
| stetig | Gamma/Log | nein | 10 (8) | 6 (5) | 0,21 | +0,764 |
| stetig | Gamma/Kehrwert | ja | 7 (7) | 7 (7) | 1,2·10⁻¹⁵ | −0,115 |
| Standtage | Poisson/Log | ja | 5 (5) | 5 (5) | 0 | +0,705 |
| Standtage | Poisson/Identität | nein | **103** (82) | **7** (6) | 2,8 | +5,214 |
| Standtage | NB2/Log (α = 0,2) | nein | 7 (6) | 4 (4) | 1,8·10⁻² | +0,705 |
| Standgeld | Logit | ja | 9 (8) | 9 (8) | 1,7·10⁻¹³ | +1,755 |
| Standgeld | Probit | nein | 12 (11) | 8 (8) | 0,61 | +1,064 |
| Standgeld | cloglog | nein | 12 (11) | 8 (8) | 0,50 | +1,418 |

Test: `test_claims`: `test_b1_nine_pairs_table`. Bei den vier kanonischen Paaren ist der Fisher-Pfad der Newton-Pfad (größter Unterschied 1,7·10⁻¹³), bei den fünf anderen nicht; alle Grenzwerte sind gleich (größter Unterschied 4,1·10⁻¹¹).
Eine unabhängige Minimierung (BFGS mit `jac="3-point"`, drei Startpunkte) landet in den vier in der App geprüften Paaren (Poisson/Identität, Gamma/Kehrwert, Logit, NB2/Log) höchstens 10⁻⁶ daneben (`test_b12_reference_bands`); bei Gamma/Kehrwert, nahe dem ungültigen Bereich, ist die Referenz selbst ungenau, darum prüft der Orakel-Test dort den Gradienten und den Zielfunktionswert statt der Parameter.
**Über 20 Tabellen** (Zufalls-Seeds 300 bis 319, n = 3 000) braucht Fisher-Scoring bei Poisson/Identität 35 bis 157 Schritte (Median 53), Newton 6 oder 7 (Test: `test_b2_fisher_steps_over_twenty_tables`).

### Experiment 2: der Link ist die Skala (Tabelle mit Zufalls-Seed 180, ein Datensatz)

- Standtage, Poisson: der Zoll-Koeffizient ist +0,7048 beim Log-Link (Faktor 2,024 auf das Mittel) und +5,2142 Tage bei der Identität; beim Kehrwert −0,0976. Das kleinste angepasste Mittel ist 1,212 (Log) und 0,288 (Identität).
- Verweilzeit d, Log-Link, drei Familien: Zoll-Koeffizient 0,7613 / 0,7623 / 0,7638 und **Wochenende 0,2016 / 0,1780 / 0,1536** (Normal / Poisson-Quasi / Gamma). Mittel je Art im Training, Σμ̂/Σy − 1 in % (Import, Export, Umschlag, Leer):
  Normal +0,38 / −1,27 / +0,64 / +0,50; Poisson 0,00 überall; Gamma −0,46 / +2,18 / −1,39 / −0,50. Die Familie legt fest, welche Container das Mittel anpassen; das Modell ist nicht exakt log-linear, darum bleibt je Familie eine andere Abweichung (Deutung, nicht gemessen).
- Normal/Identität auf d (OLS) sagt negative Verweilzeiten voraus: kleinstes angepasstes Mittel −1,366 Tage.
- Zwei Gruppen (Handbeispiel): alle drei Links geben dieselben Mittel und verschiedene Koeffizienten (H7).

Test: `test_b3_scale_and_family`.

### Experiment 3: Link-Wahl mit dem AIC (vorgerechnet, 1 000 Datensätze je Zeile, wo nicht anders angegeben)

AIC mit der auf die Form profilierten Gamma-Likelihood (Parameter Mittel + 1); gewählt wird unter den Datensätzen mit mindestens einer gültigen Anpassung.
**Export** (Vorlauf gleichverteilt in [1; 10] Tagen, wahr: Identität). Vorhersage geteilt durch das wahre Mittel, Median über die Datensätze, bei Vorlauf 5,5 / 15 / 20 Tagen; der Kehrwert-Link hat bei Vorlauf 15 und 20 nur in höchstens 3,2 % der Datensätze eine definierte Vorhersage (η > 0; Spalte „Kehrwert definiert“):

| n (Datensätze) | AIC wählt Identität / Log / Kehrwert | Vorlauf 5,5: Identität / Log / Kehrwert | Vorlauf 15 | Vorlauf 20 | Kehrwert definiert bei 15 / 20 | Log über 25 % zu hoch bei 20 |
|---|---|---|---|---|---|---|
| 20 (1 000) | 81,3 % / 16,6 % / 2,1 % | 1,000 / 0,906 / 0,845 | 0,999 / 1,992 / 5,770 | 0,997 / 3,738 / 4,088 | 3,2 % / 0,1 % | 99,6 % |
| 50 (1 000) | 91,5 % / 8,5 % / 0,0 % | 0,995 / 0,901 / 0,844 | 0,994 / 2,010 / 14,425 | 0,995 / 3,789 / nicht definiert | 0,4 % / 0,0 % | 100,0 % |
| 200 (1 000) | 99,9 % / 0,1 % / 0,0 % | 0,998 / 0,903 / 0,845 | 0,999 / 2,016 / nicht definiert | 0,999 / 3,810 / nicht definiert | 0,0 % / 0,0 % | 100,0 % |
| 1 000 (500) | 100,0 % / 0,0 % / 0,0 % | 1,000 / 0,904 / 0,846 | 0,999 / 2,025 / nicht definiert | 1,000 / 3,824 / nicht definiert | 0,0 % / 0,0 % | 100,0 % |
| 5 000 (100) | 100,0 % / 0,0 % / 0,0 % | 0,999 / 0,903 / 0,846 | 0,999 / 2,020 / nicht definiert | 0,999 / 3,819 / nicht definiert | 0,0 % / 0,0 % | 100,0 % |

**Import** (nur Zoll und Wochenende, 0/1-Merkmale, wahr: Log; es gibt keinen Bereich außerhalb der Daten). Größte Abweichung μ̂/μ − 1 über die vier Zellen, Median:

| n (Datensätze) | AIC wählt Log / Identität / Kehrwert | ungültige Anpassungen (Identität / Log / Kehrwert) | größte Zellabweichung (Identität / Log / Kehrwert) |
|---|---|---|---|
| 20 (1 000) | 14,2 % / 50,9 % / 34,9 % | 16 / 7 / 17 | 0,417 / 0,416 / 0,455 |
| 50 (1 000) | 14,4 % / 44,2 % / 41,4 % | 1 / 0 / 10 | 0,268 / 0,262 / 0,300 |
| 200 (1 000) | 27,9 % / 40,2 % / 31,9 % | 0 / 0 / 0 | 0,161 / 0,133 / 0,170 |
| 1 000 (600) | 62,5 % / 23,3 % / 14,2 % | 0 / 0 / 0 | 0,109 / 0,062 / 0,114 |
| 5 000 (300) | 93,3 % / 5,7 % / 1,0 % | 0 / 0 / 0 | 0,091 / 0,029 / 0,091 |

Der Log-Link liegt im Datenbereich des Exports in der Mitte um rund 10 % daneben (0,90 bei Vorlauf 5,5), am Rand des Datenbereichs anders (nachgerechnet, nicht in der vorgerechneten Datei, Median über 100 Datensätze mit n = 5 000: bei Vorlauf 1 etwa das 1,4-Fache, bei Vorlauf 10 etwa das 1,2-Fache), bei Vorlauf 15 beim 2,0-Fachen und bei Vorlauf 20 beim 3,8-Fachen; im Export wählt der AIC mit einem stetigen Merkmal ab n = 50 meist den wahren Link (91,5 %). Bei 0/1-Merkmalen wählt er bei n = 50 häufiger einen falschen Link als den wahren (14,4 % gegen 85,6 %).
Aussage mit Design: **stetiges Merkmal gegen 0/1-Merkmale, Gamma-GLM, ein Merkmal bzw. zwei, ohne Kunden im Modell**. Tests: `test_b4_link_choice_export`, `test_b5_link_choice_import` (Anteile als Band aus drei Standardfehlern, Datei und Neu-Rechnung zweier Zeilen in `test_frozen`).

### Experiment 4: die Residualdeviance beim Standgeld (vorgerechnet; n = 1 050 Import-Container, 600 Datensätze je Zeile, Welt ohne Kundeneffekt, Modell mit Zoll und Wochenende **exakt richtig**)

Alle Schwellen werden an demselben Datensatz gerechnet. Die Anteile Standgeld (Spalte 2) gelten für den Generator dieser Demo; andere Demos zu den Containern bauen ihren Generator eigenständig nach und nennen deshalb ihre eigene Rate. Test gegen χ² auf dem Niveau 5 %; „D/(n − p)“ ist die ungruppierte Deviance durch die Freiheitsgrade, „2·H̄“ das Doppelte der mittleren Entropie der wahren Wahrscheinlichkeiten.

| Schwelle c (Tage) | Anteil Standgeld | 2·H̄ | D/(n − p) | D über dem 95-%-Wert | D unter dem 5-%-Wert | gruppiert (4 Zellen) über dem 95-%-Wert | gültige Datensätze |
|---|---|---|---|---|---|---|---|
| 2 | 0,902 | 0,582 | 0,583 | 0,0 % | 100,0 % | 4,3 % | 539 |
| 3 | 0,735 | 1,028 | 1,029 | 7,7 % | 0,0 % | 4,2 % | 600 |
| 4 | 0,568 | 1,191 | 1,191 | **100,0 %** | 0,0 % | 4,5 % | 600 |
| 5 | 0,431 | 1,167 | 1,168 | 99,8 % | 0,0 % | 5,0 % | 600 |
| 7 | 0,247 | 0,925 | 0,926 | 0,0 % | 53,0 % | 7,0 % | 600 |
| 10 | 0,114 | 0,568 | 0,566 | 0,0 % | 100,0 % | 6,5 % | 600 |
| 14 | 0,045 | 0,287 | 0,285 | 0,0 % | 100,0 % | 7,2 % | 600 |
| 21 | 0,011 | 0,092 | 0,097 | 0,0 % | 100,0 % | 5,2 % | 427 |

Bei **richtigem** Modell verwirft die ungruppierte Deviance in 2 von 8 Zeilen (Schwelle 4 und 5 Tage: 100,0 % und 99,8 %); in 5 von 8 Zeilen liegt sie in mindestens 53,0 % der Datensätze **unter** dem 5-%-Wert (in vier davon, bei 2, 10, 14 und 21 Tagen, in 100,0 %); nur eine Zeile (3 Tage) liegt nahe dem Niveau (7,7 % über dem 95-%-Wert).
D/(n − p) ist in allen acht Zeilen gleich 2·H̄ auf höchstens 0,005 (gemessen). Deutung (nicht gemessen): die Häufigkeit des Ereignisses bestimmt, ob der Wert über oder unter dem χ²-Wert liegt, die Passung geht nicht ein (Test: `test_b6_threshold_series`).
Bei der Schwelle 21 Tage sind nur 427 von 600 Datensätzen gültig (zu wenige Ereignisse: Trennung oder keine Konvergenz); die Zahlen sind auf Datensätze mit Schätzer bedingt.

### Experiment 5: was ein Test leisten kann (vorgerechnet; Standgeld ab 7 Tagen; Probit-Link, gültige Datensätze)

M0 = Achsenabschnitt, Zoll, Wochenende (p = 3); M1 = zusätzlich elf Kundenmerkmale (p = 14). Zellen: Zoll × Wochenende × Kunde (48) bzw. Zoll × Wochenende (4); Anteil Zellen mit erwarteter Zahl unter 5 in Klammern (48 Zellen).

| n (Datensätze) | Welt / Modell | ungruppiert über 95-%-Wert | gruppiert, 48 Zellen | gruppiert, 4 Zellen | Likelihood-Quotienten-Test M0 gegen M1 (11 Freiheitsgrade) |
|---|---|---|---|---|---|
| 350 (400) | mit Kundeneffekt, M0 (**falsch**) | 2,2 % | 100,0 % (100 %) | 5,0 % | 100,0 % (303 gültig) |
| 1 050 (400) | mit Kundeneffekt, M0 (**falsch**) | 5,0 % | 100,0 % (43 %) | 5,5 % | 100,0 % (397 gültig) |
| 3 500 (250) | mit Kundeneffekt, M0 (**falsch**) | 13,2 % | 100,0 % (2 %) | 5,6 % | 100,0 % (250 gültig) |
| 350 | mit Kundeneffekt, M1 (richtig); 303 gültig | 0,0 % | 11,6 % (97 %) | – | – |
| 1 050 | mit Kundeneffekt, M1 (richtig); 397 gültig | 0,0 % | 9,6 % (69 %) | – | – |
| 3 500 | mit Kundeneffekt, M1 (richtig) | 0,0 % | 8,8 % (23 %) | – | – |
| 350 | ohne Kundeneffekt, M0 (richtig) | 0,0 % | 22,5 % (100 %) | 4,0 % | 6,9 % (394 gültig) |
| 1 050 | ohne Kundeneffekt, M0 (richtig) | 0,0 % | 11,0 % (59 %) | 6,8 % | 4,0 % (400 gültig) |
| 3 500 | ohne Kundeneffekt, M0 (richtig) | 0,0 % | 6,0 % (6 %) | 6,8 % | 5,6 % (250 gültig) |

Die ungruppierte Deviance erkennt das falsche Modell (Kunden fehlen) bei n = 350 und 1 050 nicht (2,2 % und 5,0 %, bei Niveau 5 %), bei n = 3 500 in 13,2 %; die gruppierte Deviance mit 48 Zellen und der Likelihood-Quotienten-Test erkennen es in allen gültigen Datensätzen (bei n = 350 sind nur 303 von 400 gültig, bei n = 1 050 sind es 397 von 400; gemessen für diesen Kundeneffekt: zwölf feste Effekte, Streuung 0,379).
Die gruppierte Deviance hält ihr Niveau nur bei großen Zellen (richtiges Modell, 48 Zellen: 22,5 / 11,0 / 6,0 % bei 100 / 59 / 6 % Zellen mit erwarteter Zahl unter 5); mit den **groben** Zellen Zoll × Wochenende (4 Zellen) sieht sie den fehlenden Kundeneffekt nicht (Verwerfung 5,0 / 5,5 / 5,6 %, gleich dem Niveau 5 %).
Der Likelihood-Quotienten-Test hält sein Niveau (6,9 / 4,0 / 5,6 % bei richtigem Modell; Band um 5 %: ± 3,3 / 3,3 / 4,1 Prozentpunkte, drei Standardfehler).
**Falscher Link bei richtigem Modell (M1, Welt mit Kundeneffekt):** gruppierte Deviance (48 Zellen) mit Logit 11,9 / 9,8 / 12,4 %, mit cloglog 19,3 / 35,0 / 82,4 % (n = 350 / 1 050 / 3 500; Probit 11,6 / 9,6 / 8,8 %): bei Logit statt Probit liegt der Unterschied in den drei gemessenen Zellen bei höchstens 3,6 Prozentpunkten (12,4 gegen 8,8 % bei n = 3 500; zum Vergleich: der Standardfehler eines Anteils bei 400 oder 250 Datensätzen liegt bei 1,5 bis 2 Prozentpunkten), bei cloglog ab n = 1 050 bei 25,4 und mehr (35,0 gegen 9,6 %).
**AIC-Wahl zwischen Probit, Logit und cloglog** (M1, Welt mit Kundeneffekt, wahr: Probit): 52,8 / 66,0 / 82,0 % Probit bei n = 350 / 1 050 / 3 500. Bei n = 350 fehlen 97 von 400 Datensätzen (M1: Trennung oder keine Konvergenz), bei n = 1 050 drei; die Zahlen sind auf Datensätze mit Schätzer bedingt.
Tests: `test_b7_grouping_and_lr`, `test_b8_wrong_link`.

### Weitere Befunde

- **Die Entropie-Deutung gilt auch in einer anderen Konfiguration** (`test_b9_entropy_formula`): Bernoulli mit logit p = −0,5 + x, x ~ N(0; 1), n = 200, p = 2: die Gauß-Hermite-Quadratur gibt 2·H̄ = 1,1635 und (2nH̄ − p)/(n − p) = 1,1651; 300 simulierte Datensätze geben ein mittleres D/(n − p) von 1,170 und 52,3 % über dem 95-%-Wert (Band 40 bis 65 %).
- **NB2-α** braucht eine Außenschleife: auf den Standtagen der Tabelle (Mittel 6,221, Varianz 43,50) geben Profil-Likelihood und abwechselnde Schätzung beide α̂ = 0,1766 (Unterschied unter 10⁻³ relativ; `test_b10_alpha_and_separation`).
- **Separation** (Logit, x = −2, −1, 1, 2; y = 0, 0, 1, 1, ohne Schritthalbierung): der Koeffizient von x ist nach 1 / 2 / 3 / 5 / 10 / 20 Schritten 1,20 / 2,02 / 2,91 / 4,84 / 9,82 / 19,82 und wächst weiter; nach 40 Schritten steht dort 35,39, weil die Rechengrenze |η| ≤ 30 greift (keine Eigenschaft der Daten).
- **Vehikel-Prüfsteine** (`test_oracle_vehikel`): Export-Mittel gleich C·(0,6 + 0,85·Vorlauf) (Σd/Σ Formelmittel 1,000 ± 0,006 bei 40 000 Containern, je Vorlauf-Fünftel ± 0,02); Standgeld-Probit trifft alle 14 Koeffizienten innerhalb von 4,5 Standardfehlern (30 000 Container).

### Rechenregeln und Hinweise zu den Zahlen

- Schrittzahlen, Anteile und Verhältnisse sind **Bänder**, keine Konstanten (Monte-Carlo-Standardfehler √(p(1 − p)/Datensätze), Unterschiede zwischen Zufalls-Seeds); die Tests prüfen Bänder oder die Zählung „k von m“. Exakt sind nur Brüche und geschlossene Formeln.
- Plattform: der Zufall ist bitgleich (SplitMix64 mit exakten Funktionen), die Gleitkomma-Lösungen (BLAS, `numpy.exp`) haben Toleranz 10⁻⁹ bis 10⁻¹²; Zählungen aus Neu-Rechnung gelten ± 2 Datensätze, bei der Link-Wahl ± 2 % der Datensätze: im Import bei n = 20 wechselt der AIC-Sieger zwischen Identität und Kehrwert bei etwa 11 von 993 Datensätzen, wenn man dieselbe Rechnung unter Linux statt unter Windows ausführt (die Zeile 14,2 % / 50,9 % / 34,9 % wird dort zu 14,1 % / 49,8 % / 36,1 %; Gleichstände zwischen fast gleichen Anpassungen).
- **Gültig** zählt eine Anpassung, die konvergierte und deren Koeffizienten unter 15 blieben (sonst Trennung oder Rechengrenze); die Zahl ungültiger Anpassungen steht in den Tabellen.
- Jede Zahl der Tabellen steht als Band im Test der Spalte „Test“ oder im Abschnitt; nicht im Test stehen: die Rechenzeit der App (erster Lauf rund 3 Sekunden auf dem Entwicklungsrechner, AppTest, einmal gemessen) und die in „Ehrliche Grenzen“ genannte frühere Messung des Lasso.

## Quellen

Nur als gelesen zitiert: Wedderburn (1974), Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method, Biometrika 61 (Einleitung: jede Iteration der 1972 von Nelder und Wedderburn definierten Klasse ist eine gewichtete lineare Regression);
Hastie, Tibshirani, Friedman, The Elements of Statistical Learning, Abschnitt 4.4.1 (IRLS beim Logit gleich Newton, Schrittweitenhalbierung, „convergence is never guaranteed“); James, Witten, Hastie, Tibshirani, An Introduction to Statistical Learning (2. Auflage), Abschnitt 4.6.3
(Link-Funktion); ein ETH-Seminarskript zu GLM (Studierendenskript, geringe Autorität: bei kanonischem Link sind Newton und Fisher-Scoring gleich); Van Calster u. a. (2019), Calibration: the Achilles heel of predictive analytics, BMC Medicine 17 (raten vom Hosmer-Lemeshow-Test ab; die hier gruppierte Deviance
bildet **natürliche** Zellen nach den Merkmalen, nicht künstliche Gruppen nach p̂, ist also nicht dasselbe). Nelder und Wedderburn (1972) liegt nur als Abstract vor; was McCullagh und Nelder (1989) zu Deviance und Dispersion schreiben, ist **nicht nachgelesen**, die Aussagen dazu sind eigene Rechnung dieser Demo.
Die Aussage „Deviance und R² sind keine Modellprüfung“ ist nicht aus einer einzelnen Quelle belegt, sondern hier als Messung geführt. Nichts davon ist wissenschaftlich neu: IRLS, kanonischer Link und Deviance sind Lehrbuchstoff; neu ist nur die Vorführbarkeit mit Zähleinheit auf einem wachsenden Beispiel.

## Ehrliche Grenzen

- **Ein synthetisches Vehikel.** Alle Zahlen gelten für diese erfundenen Container, zwölf feste Kundeneffekte und die gewählten Parameter; sie sind keine Aussage über einen Hafen. Die Form („richtig“ und „falsch“) ist nur im Export-Mittel und im Standgeld-Probit des Imports bekannt; deshalb stehen die Experimente 3 bis 5 dort.
- **Der Link ist nur mit stetigem Merkmal lesbar.** Bei 0/1-Merkmalen wählt der AIC bei n = 50 in 85,6 % der Datensätze einen falschen Link; auch der richtige Link hat dort keine Vorhersage außerhalb. Die größte Zellabweichung liegt im Median bei n = 50 für alle drei Links zwischen 0,262 und 0,300, bei n = 5 000 zwischen 0,029 und 0,091.
- **Die Link-Wahl nutzt AIC mit profilierter Form**; die Konvention anderer Software für den Gamma-AIC (zum Beispiel R mit Deviance/n als Dispersion) kann andere Werte geben. Kunden im Modell der Link-Wahl, andere Formen (fehlender Vorlauf, Wechselwirkungen, die Zählform aus der Aufrundung) und andere Zellen für die Gruppierung (Hosmer-Lemeshow, Pearson mit Momentkorrekturen) sind **nicht gemessen**.
- **Auch die Alternativen haben Grenzen:** die gruppierte Deviance hält das Niveau bei kleinen Zellen nicht, und Logit statt Probit unterscheidet sich in der gruppierten Deviance in den drei gemessenen Zellen um höchstens 3,6 Prozentpunkte, bei Standardfehlern von 1,5 bis 2 Punkten (Experiment 5); der Likelihood-Quotienten-Test braucht ein eingebettetes Modell, das man schon kennt.
- **Eine Tabelle für die neun Paare.** Schrittzahlen und Koeffizienten sind Messwerte eines Datensatzes; die Verteilung über Wiederholungen ist nur für Poisson/Identität (20 Tabellen) gemessen.
- **Lasso, Quantilregression, Kalibrierung, gemischte Modelle** stehen nur als Einordnung (Experiment 6); die Aussage „IRLS als äußere Schleife beim Lasso-Logit“ stützt sich auf eine frühere Messung (IRLS außen, Koordinatenabstieg innen, gleich der direkten Minimierung bis auf 7,1·10⁻⁸), die nicht Teil dieser Demo ist und hier nicht neu gerechnet wurde (Zahl nach Bericht dieser Messung). Quasi-Likelihood mit eigener Varianzfunktion, Tweedie, Nullüberschuss und GLM mit Zufallseffekten sind nicht gemessen.
- **Die Rechenzeit der vorgerechneten Messungen** (rund 10 Minuten) bleibt aus der App heraus; die Live-Wiederholung rechnet nur wenige Datensätze und ist darum ein Band mit großer Unsicherheit, die die App mitnennt.

## Verifikation

- **Kern:** IRLS gegen Handrechnung auf den neun erfundenen Containern in Brüchen (`tests/test_algorithm.py`), Randfälle (`tests/test_edge_cases.py`: n = p, n < p, konstante und kollineare Spalten, y = 0, Gamma mit y ≤ 0, ungültiges Mittel bei Identität und Kehrwert, Separation, Schrittbudget, Schritthalbierung, leere Zellen).
- **Orakel** (`tests/test_oracle_glm_familie.py`): alle neun Paare gegen eine unabhängige Minimierung der `scipy.stats`-Likelihood (BFGS mit `jac="3-point"`, eigene Umkehrfunktionen der Links) und den Gradienten dieser Likelihood am IRLS-Ergebnis; scikit-learn (`LogisticRegression`, `PoissonRegressor`, `GammaRegressor`) und `lstsq`; Einheitsdeviancen gleich 2·(ℓ_gesättigt − ℓ) aus `scipy.stats`; ungruppiert = gruppiert − 2·ℓ_gesättigt; Gamma-Profil-Likelihood gegen `gamma.fit`; NB2-Likelihood gegen `nbinom`; α durch Außenschleife gegen abwechselnde Schätzung; Likelihood-Quotienten-Statistik gegen die Differenz zweier Log-Likelihoods. **Vehikel** (`tests/test_oracle_vehikel.py`): SplitMix64 gegen die veröffentlichte Folge, ln-Median und Streuung gegen die Formeln, Export-Mittel linear, Standgeld-Probit exakt.
- **Eigenschaften** (`tests/test_eigenschaften_glm_familie.py`, je 20 Instanzen): kanonischer Link gleich Pfad, nicht kanonisch verschieden mit gleichem Grenzwert, Score null am Grenzwert, Residuensumme null bei kanonischem Link mit Achsenabschnitt, Vertauschen, Verdoppeln, Vorgewicht, Offset, Skalierung, Deviance nie negativ und nie steigend bei zusätzlichem Merkmal, NB2 mit α → 0 gleich Poisson.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README als Band oder Zählung; **Vorgerechnetes** (`tests/test_frozen.py`): Raster und Einzelwerte der Datei, dazu zwei ganze Zeilen neu gerechnet (Zählungen ± 2 % der Datensätze).
- **Oberfläche:** Verdikte als reine Funktionen (`tests/test_verdicts.py`), Presets (`tests/test_presets.py`), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf, `tests/test_layout.py`), AppTest-Smoke (`tests/test_app.py`), Footer, Laufzeit-Importe.
- **Testzahl:** 208 Tests (`pytest tests/ --collect-only -q`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Kopf, Kernfragen, sechs Experimente, Grenzen, Formeln) |
| `gfm_algorithm.py` | Kern: Familien, Links, IRLS (Fisher-Scoring und Newton, Schritthalbierung, Offset, Gewichte) |
| `gfm_constants.py` | Regler-Grenzen, Beschriftungen, Raster der vorgerechneten Messungen, Farben, Schnellstart-Szenarien |
| `gfm_evaluation.py` | Rechnungen der Experimente und Verdikte (reine Funktionen) |
| `gfm_experiments.py` | Messreihen: ein Datensatz je Wiederholung, Anteile über Datensätze (vorgerechnet und live) |
| `gfm_frozen.py` | vorgerechnete Messungen erzeugen, laden, kontrollieren |
| `gfm_inference.py` | Log-Likelihoods, Deviance (gruppiert, ungruppiert), AIC, Likelihood-Quotienten-Test, NB2-α |
| `gfm_layout.py` | Fester Kopf (Stylesheet) |
| `gfm_presets.py` | Permalink, Schnellstart (setzt alle Regler, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `gfm_reference.py` | unabhängige Referenz: `scipy.stats`-Likelihood, BFGS mit zentralen Differenzen, Gradientenprobe |
| `gfm_rng.py` | SplitMix64 mit exakten `normal()` und `expovariate()` (Kopie der geprüften Vorlage des Portfolios) |
| `gfm_scenario.py` | Container-Generator (Nachbau, Export, Import, Tabelle) und Handbeispiel |
| `gfm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/gfm_frozen.json` | vorgerechnete Messungen der Experimente 3 bis 5 |
| `tools/gfm_make_frozen.py` | erzeugt die Datei neu; `--check` prüft sie gegen die Suite |
| `requirements.txt` | Laufzeit: streamlit, numpy, scipy, pandas, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Gegenprobe) |
| `tests/` | Kern, Orakel, Eigenschaften, Randfälle, Claims, Vorgerechnetes, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Quasi-Likelihood mit eigener Varianzfunktion, Tweedie, Nullüberschuss, GLM mit Zufallseffekten**: eigene Nachbar-Demos oder nicht gemessen.
- **Hosmer-Lemeshow-Test** (künstliche Gruppen nach p̂; abgeraten von Van Calster u. a.) und eine **Standtage-Zeile der Deviance-Falle** (Überdispersion verwürfe das Poisson-Modell zwar zu Recht, aber aus dem falschen Grund; nicht gemessen).
- **Separation, Existenz des Schätzers, Firth**: Gegenstand von `logit-separation-demo`; hier nur die Zeile in Experiment 6. **Entscheidung mit Kosten, Rekalibrierung**: `kalibrierung-schwelle-demo`. **Standardfehler, Überdeckung bei Überdispersion**: `zaehldaten-demo`.
- **Das Zwei-Punkte-Beispiel** (zwei erfundene Punkte, Identität gegen Log): der Gedanke steht im Handbeispiel mit zwei Gruppen und im Export-Experiment mit Zahlen aus dem Generator.
- **PDF-Export** und eine Live-Wiederholung von Experiment 5 (die Zeilen brauchen mehrere Minuten): die Datei ist vorgerechnet und gekennzeichnet.
- **Kunden im Modell der Link-Wahl und andere Zellen der Gruppierung**: nicht gemessen (Abschnitt „Ehrliche Grenzen“).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/gfm_make_frozen.py --check      # die vorgerechnete Datei gegen die Suite prüfen
python tools/gfm_make_frozen.py              # sie neu erzeugen (rund 10 Minuten)
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html).
