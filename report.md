# Information Extraction Pipeline — Airport & Passport Domain

## 1. Objective

Build an end-to-end information extraction (IE) system over airport / passport
domain text that performs: POS tagging, named entity recognition (general +
domain-specific), relation extraction, event extraction, and temporal ordering
of events — surfaced through a Streamlit app with a notebook-developed backend.

## 2. Domain & Data

**Domain:** passenger journey narratives — passport/visa issuance, check-in,
security screening, immigration clearance, boarding, departure, arrival, and
customs — the kind of text found in immigration logs, travel itineraries, or
passenger-service notes.

**Corpus:** a synthetic sample corpus (`sample_data/sample_corpus.txt`) was
authored for this project rather than using real passport images or genuine
personal data, since passports contain sensitive personally identifiable
information. The app accepts any user-provided text of the same style
(pasted, uploaded, or the bundled sample), so it generalizes beyond the demo
corpus.

## 3. Pipeline Architecture

```
raw text
   │
   ▼
spaCy tokenizer + tagger + parser  (POS tagging, dependency parse)
   │
   ▼
EntityRuler (domain patterns) → spaCy statistical NER  (overwrite_ents=True)
   │                                        │
   ▼                                        ▼
generic entities                 domain entities: PASSPORT_NO, FLIGHT_NO,
(PERSON, GPE, ORG,                GATE, TERMINAL, SEAT, VISA_TYPE,
 DATE, TIME, NORP)                 DOCUMENT, AIRPORT
   │
   ▼
name-coreference cleanup (repeated bare first/last names → PERSON)
   │
   ├──► Relation extraction  (dependency-rule matching over subj–verb–obj)
   ├──► Event extraction     (trigger lexicon + nearby DATE/TIME + participants)
   ├──► Wikidata enrichment  (wbsearchentities REST lookup, offline fallback)
   │
   ▼
Event list ──► Temporal graph (NetworkX DiGraph) ──► topological sort ──► timeline
```

### 3.1 POS Tagging
Uses spaCy's built-in statistical tagger (`en_core_web_sm`): `token.pos_`
(universal tag set) and `token.tag_` (fine-grained Penn Treebank tag), plus
`token.dep_` / `token.head`, which relation extraction depends on directly.

### 3.2 Named Entity Recognition
spaCy's default NER handles generic classes well (PERSON, GPE, ORG, DATE,
TIME, NORP) but misses or mislabels domain vocabulary — e.g. a flight number
like `AI202` was tagged `PRODUCT`, and `Gate 14` was tagged `ORG` in initial
testing. A custom **EntityRuler**, inserted *before* the statistical NER
component with `overwrite_ents=True`, adds high-precision regex/token
patterns for:

| Label | Examples |
|---|---|
| `PASSPORT_NO` | K1234567 |
| `FLIGHT_NO` | AI202, 6E345 |
| `GATE` | Gate 14 |
| `TERMINAL` | Terminal 3 |
| `SEAT` | Seat 22A |
| `VISA_TYPE` | tourist visa, business visa |
| `DOCUMENT` | passport, visa, boarding pass |
| `AIRPORT` | Indira Gandhi International Airport |

A lightweight **name-coreference heuristic** re-labels a bare, single-token
mention (e.g. a second, unqualified "Ravi") as `PERSON` if it matches a
name-part of an already-recognized multi-token `PERSON` entity, correcting a
common statistical-NER error on repeated references.

### 3.3 Relation Extraction
Rule-based, using the dependency parse: for each verb whose lemma appears in
a small domain lexicon (`hold → HOLDS_DOCUMENT`, `board → BOARDS`,
`depart → DEPARTS_FROM`, `arrive → ARRIVES_AT`, `issue → ISSUED_BY`,
`check → CHECKS_IN_AT`, `clear → CLEARED_AT`, etc.), the nominal subject
(`nsubj`) is taken as the source argument and the direct/prepositional object
as the target. Each argument is mapped onto the entity span that contains it
where possible. Any other subject–verb–object triple is captured generically
as `RELATED_TO` to preserve recall on relations outside the domain lexicon.

### 3.4 Event Extraction
A trigger-word lexicon maps surface phrases to canonical event types:
`PASSPORT_ISSUANCE`, `VISA_ISSUANCE`, `CHECK_IN`, `SECURITY_SCREENING`,
`IMMIGRATION_CLEARANCE`, `BOARDING`, `DEPARTURE`, `ARRIVAL`, `CUSTOMS`,
`BAGGAGE_CLAIM`. For every sentence containing a trigger, the pipeline
attaches the nearest `DATE`/`TIME` entity (parsed to a Python `datetime` via
`dateutil` where possible) and all other in-sentence entities as
participants.

### 3.5 Temporal Ordering
Events are placed into a **NetworkX** directed graph where edge `A → B`
means "A happens before B", built from two signal sources:

1. **Absolute timestamps** — events with a successfully parsed time are
   sorted and chained chronologically.
2. **Discourse cues** — *after / following / then* (forward edge) and
   *before / prior to* (backward edge) link adjacent event sentences when no
   timestamp is available.
3. **Narrative-order fallback** — when neither signal applies, sentence order
   is used, since travel narratives are normally recounted chronologically.

The graph is topologically sorted to produce a single best-effort event
timeline (falling back to sentence order if the cues create a cycle).

### 3.6 Wikidata Enrichment
`GPE`, `ORG`, `NORP`, and `AIRPORT` entities are looked up against Wikidata's
public `wbsearchentities` REST API, returning a QID, canonical URL, and short
description — grounding extracted entities in a structured knowledge base
(e.g. confirming "Dubai" resolves to the city, not the emirate). A small
offline fallback dictionary keeps the pipeline usable without internet
access; the deployed Streamlit app uses live lookups by default.

## 4. System Design (Notebook → Module → App)

- **`backend/pipeline.ipynb`** — the pipeline was developed and validated
  stage-by-stage here, with inline output at each step (POS table, entity
  table, relation table, event table, temporal graph plot, Wikidata lookup
  demo).
- **`backend/pipeline.py`** — the finalized, importable version of the same
  logic (`run_pipeline(text)` orchestrates all stages), used directly by the
  app for fast, cached execution (no notebook kernel dependency at runtime).
- **`app.py`** — Streamlit frontend with tabs for annotated text, POS table,
  entity table (Wikidata-enriched, filterable, downloadable as CSV),
  relation table, event table, and an interactive temporal graph +
  chronological timeline, plus this report.

## 5. Deliverables Produced

| Deliverable | Location |
|---|---|
| Annotated text output | "Annotated Text" tab in the app (color-coded entities + POS subscripts); also `render_annotated_html()` in `pipeline.py` |
| Entity & relation tables | "Entities" / "Relations" tabs, downloadable as CSV |
| Event timeline / temporal graph | "Temporal Graph" tab — ordered list + NetworkX visualization, downloadable PNG |
| Pipeline report | this document |
| Backend development notebook | `backend/pipeline.ipynb` |

## 6. Limitations & Future Work

- **Statistical NER errors** on unfamiliar names/tokens are only partially
  corrected (single-mention person coreference); a transformer-based model
  (`en_core_web_trf`) or Hugging Face NER model would likely reduce this
  further at the cost of speed.
- **Relation extraction recall** is bounded by the dependency-parse rules and
  domain verb lexicon; a fine-tuned relation-extraction model (e.g. via
  Hugging Face `transformers`) could capture relations expressed without a
  direct verb link.
- **Temporal ordering** relies on explicit timestamps and a small discourse-
  cue lexicon; more sophisticated temporal-relation classification (e.g.
  TimeML/TLINK-style models) would handle implicit or overlapping event
  timing more robustly.
- **Wikidata enrichment** is a single top-match lookup; ambiguous entity
  names (e.g. common city names) would benefit from context-aware entity
  linking rather than a bare string search.
