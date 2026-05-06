---
title: "Unterstützter Modellumfang und Systemgrenzen der Netzimport-Pipeline"
subtitle: "diffpf – aktueller Import- und Modellierungsumfang"
date: "Stand: 2026-05-04"
---

# Unterstützter Modellumfang und Systemgrenzen der Netzimport-Pipeline

> Dieses Dokument beschreibt den aktuell unterstützten Modellumfang der Netzimport-Pipeline sowie die wesentlichen Systemgrenzen.  
> Der Fokus liegt auf stationären, symmetrischen AC-Leistungsflussnetzen mit statischer Topologie und einem bewusst begrenzten pandapower-nahen Importumfang.

---

## 1. Unterstützte Netzelemente

Die aktuelle Netzimport-Pipeline unterstützt stationäre AC-Netze mit folgendem Modellumfang:

| Netzelement | Unterstützte Abbildung |
|---|---|
| `ext_grid` | Externe Netzanbindung als Slack-Knoten mit vorgegebenem Spannungsbetrag und Spannungswinkel |
| `bus` | Elektrische Netzknoten der internen Netzrepräsentation |
| `load` | Feste Wirk- und Blindleistungsentnahme, einschließlich Skalierungsfaktor |
| `sgen` | Feste Wirk- und Blindleistungseinspeisung |
| `gen` | Vereinfachte Generatorabbildung ohne vollständige Blindleistungs- und Spannungsregelungslogik |
| `line` | Leitung im Pi-Ersatzschaltbild mit Serienimpedanz und Queradmittanz |
| `trafo` | Zweiwicklungs-Transformator im Pi-Ersatzschaltbild, einschließlich Übersetzungsverhältnis und Phasenverschiebung |
| `shunt` | Konstante Admittanz am jeweiligen Netzknoten |
| Bus-Bus-Schalter | Geschlossene Bus-Bus-Schalter als topologische Bus-Fusion |
| Leitungsschalter | Einfache Aktivierung oder Deaktivierung von Leitungen innerhalb des unterstützten Switch-Modells |

---

## 2. Anforderungen an verarbeitbare Netze

Ein Netz sollte die folgenden Eigenschaften erfüllen, damit es mit der aktuellen Pipeline und dem numerischen Kern verarbeitet werden kann:

### Netzmodell

- Das Netz beschreibt einen stationären, symmetrischen AC-Leistungsfluss.
- Es existiert genau ein Slack-Knoten beziehungsweise eine externe Netzanbindung.
- Die Netzstruktur ist während der Berechnung statisch.
- Diskrete Umschaltungen oder Regelzustände werden nicht innerhalb des Solvers verändert.

### Elektrische Betriebsmittel

- Lasten und Einspeisungen lassen sich als feste P/Q-Injektionen oder als vereinfachte Generatorrepräsentation abbilden.
- Leitungen sind physikalisch konsistent parametriert.
- Transformatoren sind als Zweiwicklungs-Transformatoren modellierbar.
- Shunts lassen sich als konstante Wirk- und Blindadmittanzen in die Y-Bus-Matrix einbringen.

### Datenqualität und Parametrierung

- Bus-Zuordnungen und Leitungsendpunkte sind vollständig und eindeutig.
- Es liegen gültige Basisgrößen für die Per-Unit-Umrechnung vor.
- Leitungen und Transformatoren besitzen keine unzulässigen Nullimpedanzen.
- Die Eingabedaten enthalten keine aktiven Elemente, die eine nicht unterstützte Controllerlogik oder diskrete Betriebszustandswechsel erfordern.

---

## 3. Nicht unterstützter Modellumfang

Die folgenden Modellbestandteile sind im aktuellen Entwicklungsstand nicht Teil des unterstützten Scopes:

| Nicht unterstützter Bestandteil | Einordnung |
|---|---|
| Vollständige pandapower-Kompatibilität | Der Import ist bewusst scope-begrenzt und nicht als vollständiger pandapower-Ersatz ausgelegt |
| Blindleistungsgrenzen von Generatoren | Kein Q-Limit-Enforcement |
| PV-PQ-Umschaltung | Keine automatische Umschaltung bei Verletzung von Blindleistungsgrenzen |
| Vollständige PV-Bus-Semantik | Keine vollständige spannungsregelnde Generatorlogik |
| Controller und Schutzlogiken | Keine diskreten Regel- oder Schutzmechanismen |
| Dreiwicklungs-Transformatoren | `trafo3w` wird nicht unterstützt |
| `xward` / `ward` | Nicht Teil des aktuellen Modellumfangs |
| `impedance` | Nicht Teil des aktuellen Modellumfangs |
| Gleichstromleitungen | `dcline` wird nicht unterstützt |
| Offene Leitungsenden | Keine detaillierte Modellierung offener Leitungsenden |
| Unsymmetrische oder dreiphasige Lastflüsse | Der aktuelle Kern behandelt symmetrische AC-Leistungsflüsse |
| Großskalige industrielle Netzberechnung | Nicht Ziel des aktuellen Proof-of-Concept-Umfangs |

---

## 4. Kurzfazit

Die Netzimport-Pipeline ist für klar strukturierte, stationäre AC-Netze mit statischer Topologie ausgelegt.  
Sie unterstützt die wesentlichen Betriebsmittel kleiner bis mittlerer Verteil- und Demonstrationsnetze, insbesondere Busse, Leitungen, Lasten, Einspeiser, Zweiwicklungs-Transformatoren, Shunts und einfache Switch-Logik.

Der aktuelle Scope ist bewusst begrenzt. Ziel ist keine vollständige Nachbildung aller pandapower-Funktionalitäten, sondern eine robuste, nachvollziehbare und differenzierbare Netzrepräsentation für kontrollierte AC-Power-Flow-Experimente.
