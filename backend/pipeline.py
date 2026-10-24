"""
Airport / Passport Domain Information Extraction Pipeline
===========================================================
Stages implemented:
    1. POS Tagging              -> spaCy statistical tagger
    2. Named Entity Recognition -> spaCy NER + domain EntityRuler (regex/pattern based)
    3. Relation Extraction      -> dependency-parse rule matching + domain templates
    4. Event Extraction         -> trigger-lexicon + argument (entity) attachment
    5. Temporal Ordering        -> absolute-time parsing + discourse cue graph (NetworkX)
    6. Wikidata Enrichment      -> SPARQL / REST lookups for GPE, ORG, NORP entities

This module is the "production" version of the logic that is developed and
tested interactively in backend/pipeline.ipynb. The Streamlit app
(app.py) imports this module directly.
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from dateutil import parser as dateparser
import pandas as pd
import networkx as nx

# ----------------------------------------------------------------------------
# 1. MODEL LOADING + DOMAIN ENTITY RULER
# ----------------------------------------------------------------------------

_NLP_CACHE = {}


def load_nlp(model_name: str = "en_core_web_sm"):
    """Load (and cache) a spaCy pipeline with a domain EntityRuler inserted
    before the statistical NER component so that high-precision regex/
    pattern matches for airport & passport vocabulary win over generic
    labels (e.g. flight numbers being tagged PRODUCT, gates tagged ORG)."""
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]

    nlp = spacy.load(model_name)

    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(DOMAIN_PATTERNS)

    _NLP_CACHE[model_name] = nlp
    return nlp


# Domain-specific EntityRuler patterns (token-based + regex-in-pattern form).
# Label scheme (custom, in addition to spaCy defaults PERSON/GPE/ORG/DATE/NORP/TIME):
#   PASSPORT_NO, FLIGHT_NO, GATE, TERMINAL, SEAT, VISA_TYPE, DOCUMENT,
#   NATIONALITY (falls back to NORP if not matched), AIRPORT
DOMAIN_PATTERNS = [
    # Passport number: sequence of 1-2 letters followed by 6-8 digits (common ICAO format)
    {"label": "PASSPORT_NO", "pattern": [{"TEXT": {"REGEX": r"^[A-PR-WYa-pr-wy][0-9]{6,8}$"}}]},
    {"label": "PASSPORT_NO", "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z]{2}[0-9]{6,7}$"}}]},

    # Flight numbers e.g. AI202, 6E345, EK501 (airline code has >=1 letter, so bare
    # years like "2024" never match: either letter+letter+digits or digit+letter+digits)
    {"label": "FLIGHT_NO", "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z][A-Za-z0-9][- ]?\d{2,4}$"}}]},
    {"label": "FLIGHT_NO", "pattern": [{"TEXT": {"REGEX": r"^\d[A-Za-z][- ]?\d{2,4}$"}}]},

    # Gate / Terminal / Seat
    {"label": "GATE", "pattern": [{"LOWER": "gate"}, {"TEXT": {"REGEX": r"^[A-Z0-9]{1,3}$"}}]},
    {"label": "TERMINAL", "pattern": [{"LOWER": "terminal"}, {"TEXT": {"REGEX": r"^[A-Z0-9]{1,2}$"}}]},
    {"label": "SEAT", "pattern": [{"LOWER": "seat"}, {"TEXT": {"REGEX": r"^\d{1,2}[A-K]$"}}]},

    # Visa types
    {"label": "VISA_TYPE", "pattern": [{"LOWER": {"IN": ["tourist", "business", "transit", "student", "work"]}}, {"LOWER": "visa"}]},

    # Travel documents
    {"label": "DOCUMENT", "pattern": [{"LOWER": "boarding"}, {"LOWER": "pass"}]},
    {"label": "DOCUMENT", "pattern": [{"LOWER": "passport"}]},
    {"label": "DOCUMENT", "pattern": [{"LOWER": "visa"}]},
    {"label": "DOCUMENT", "pattern": [{"LOWER": "e-ticket"}]},

    # Airports (name + code pattern, e.g. "Indira Gandhi International Airport (DEL)")
    {"label": "AIRPORT", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "international"}, {"LOWER": "airport"}]},
    {"label": "AIRPORT", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": "airport"}]},
]


# ----------------------------------------------------------------------------
# 2. POS TAGGING
# ----------------------------------------------------------------------------

def pos_tag(doc) -> pd.DataFrame:
    rows = [{
        "token": tok.text,
        "lemma": tok.lemma_,
        "pos": tok.pos_,
        "tag": tok.tag_,
        "dep": tok.dep_,
        "head": tok.head.text,
    } for tok in doc if not tok.is_space]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. NAMED ENTITY RECOGNITION
# ----------------------------------------------------------------------------

def _resolve_repeated_person_mentions(doc):
    """Lightweight name-coreference heuristic: statistical NER often mis-tags
    a bare first/last name on second mention (e.g. 'Ravi' alone tagged
    PRODUCT after 'Ravi Kumar' was correctly tagged PERSON earlier). If a
    single-token entity's text matches a name-part of an already-seen
    multi-token PERSON entity, relabel it PERSON. Returns a new list of
    entities (does not mutate doc.ents in place beyond a fresh Span list)."""
    known_name_parts = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent) > 1:
            known_name_parts.update(part.text for part in ent)

    new_ents = []
    for ent in doc.ents:
        if len(ent) == 1 and ent.text in known_name_parts and ent.label_ != "PERSON":
            new_ents.append(Span(doc, ent.start, ent.end, label="PERSON"))
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc


def extract_entities(doc) -> pd.DataFrame:
    doc = _resolve_repeated_person_mentions(doc)
    rows = [{
        "text": ent.text,
        "label": ent.label_,
        "start_char": ent.start_char,
        "end_char": ent.end_char,
        "sentence": ent.sent.text.strip(),
    } for ent in doc.ents]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. RELATION EXTRACTION (dependency-rule based)
# ----------------------------------------------------------------------------

# Domain relation verbs -> canonical relation label
RELATION_VERBS = {
    "hold": "HOLDS_DOCUMENT", "carry": "HOLDS_DOCUMENT", "present": "PRESENTS_DOCUMENT",
    "board": "BOARDS", "depart": "DEPARTS_FROM", "arrive": "ARRIVES_AT",
    "issue": "ISSUED_BY", "check": "CHECKS_IN_AT", "screen": "SCREENED_AT",
    "clear": "CLEARED_AT", "travel": "TRAVELS_TO", "fly": "FLIES_ON",
    "land": "LANDS_AT", "belong": "NATIONALITY_OF", "assign": "ASSIGNED",
}


def _find_entity_span(doc, token) -> Optional[Span]:
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent
    return None


def extract_relations(doc) -> pd.DataFrame:
    """Rule-based relation extraction: for each verb whose lemma is in our
    domain relation lexicon, pull the nominal subject (nsubj/nsubjpass) as
    the source argument and the direct object / prepositional object /
    attribute as the target argument, then map each to the entity span
    (if any) that contains it. Also captures generic SVO triples for any
    verb (labelled RELATED_TO) to keep recall reasonable."""
    rows = []
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ != "VERB":
                continue
            subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            objs = [c for c in tok.children if c.dep_ in ("dobj", "attr", "oprd")]
            preps = [c for c in tok.children if c.dep_ == "prep"]
            for p in preps:
                objs += [c for c in p.children if c.dep_ == "pobj"]

            if not subj or not objs:
                continue

            relation = RELATION_VERBS.get(tok.lemma_.lower(), "RELATED_TO")

            for s in subj:
                for o in objs:
                    s_ent = _find_entity_span(doc, s)
                    o_ent = _find_entity_span(doc, o)
                    rows.append({
                        "subject": s_ent.text if s_ent else s.text,
                        "subject_type": s_ent.label_ if s_ent else s.pos_,
                        "relation": relation,
                        "verb": tok.lemma_,
                        "object": o_ent.text if o_ent else o.text,
                        "object_type": o_ent.label_ if o_ent else o.pos_,
                        "sentence": sent.text.strip(),
                    })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. EVENT EXTRACTION
# ----------------------------------------------------------------------------

EVENT_TRIGGERS = {
    "PASSPORT_ISSUANCE": ["issued", "issuance"],
    "VISA_ISSUANCE": ["granted", "visa issued", "visa granted"],
    "CHECK_IN": ["check-in", "checked in", "check in", "checking in"],
    "SECURITY_SCREENING": ["security screening", "screened", "security check"],
    "IMMIGRATION_CLEARANCE": ["immigration", "passport control", "clearance", "cleared"],
    "BOARDING": ["boarding", "boarded", "board"],
    "DEPARTURE": ["departure", "departed", "took off", "depart"],
    "ARRIVAL": ["arrival", "arrived", "landed", "arrive"],
    "CUSTOMS": ["customs"],
    "BAGGAGE_CLAIM": ["baggage claim", "collected baggage", "collected his bag", "collected her bag"],
}

TIME_RE = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s?(am|pm|AM|PM)?)\b|"
    r"\b(\d{1,2}\s?(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s?\d{2,4})\b|"
    r"\b((january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s\d{1,2},?\s\d{2,4})\b",
    re.IGNORECASE,
)

CUE_BEFORE = {"before", "prior to", "ahead of"}
CUE_AFTER = {"after", "following", "once", "subsequently", "then", "later"}


@dataclass
class Event:
    event_type: str
    trigger: str
    sentence: str
    sent_index: int
    time_expr: Optional[str] = None
    time_parsed: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)


def extract_events(doc) -> List[Event]:
    events = []
    for sent_idx, sent in enumerate(doc.sents):
        sent_text_lower = sent.text.lower()
        for event_type, triggers in EVENT_TRIGGERS.items():
            for trig in triggers:
                if trig in sent_text_lower:
                    # time expression: prefer spaCy DATE/TIME ents in this sentence
                    time_expr, time_parsed = None, None
                    for ent in sent.ents:
                        if ent.label_ in ("DATE", "TIME"):
                            time_expr = ent.text
                            try:
                                time_parsed = dateparser.parse(ent.text, fuzzy=True, default=datetime(2024, 1, 1))
                            except Exception:
                                time_parsed = None
                            break
                    participants = [ent.text for ent in sent.ents if ent.label_ not in ("DATE", "TIME")]
                    events.append(Event(
                        event_type=event_type,
                        trigger=trig,
                        sentence=sent.text.strip(),
                        sent_index=sent_idx,
                        time_expr=time_expr,
                        time_parsed=time_parsed,
                        participants=participants,
                    ))
                    break  # one match per event_type per sentence
    return events


def events_to_df(events: List[Event]) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_type": e.event_type,
        "trigger": e.trigger,
        "time_expr": e.time_expr,
        "time_parsed": e.time_parsed,
        "participants": ", ".join(e.participants),
        "sentence": e.sentence,
        "sent_index": e.sent_index,
    } for e in events])


# ----------------------------------------------------------------------------
# 6. TEMPORAL ORDERING
# ----------------------------------------------------------------------------

def build_temporal_graph(events: List[Event], doc) -> nx.DiGraph:
    """Builds a directed graph of events ordered in time.
    Edge (A -> B) means "A happens before B".
    Two signal sources:
      (a) absolute parsed timestamps -> chronological ordering
      (b) discourse cue words (before/after/then) linking adjacent event
          sentences -> relative ordering when timestamps are missing
    Falls back to narrative (sentence) order when neither signal exists,
    since airport/travel narratives are usually told in chronological order.
    """
    G = nx.DiGraph()
    for i, e in enumerate(events):
        node_id = f"{i}:{e.event_type}"
        G.add_node(node_id, **{
            "event_type": e.event_type,
            "sentence": e.sentence,
            "time_expr": e.time_expr,
            "time_parsed": e.time_parsed.isoformat() if e.time_parsed else None,
            "sent_index": e.sent_index,
        })

    nodes = list(G.nodes(data=True))

    # (a) chronological edges from parsed absolute times
    timed = [(nid, d) for nid, d in nodes if d["time_parsed"]]
    timed.sort(key=lambda x: x[1]["time_parsed"])
    for (n1, _), (n2, _) in zip(timed, timed[1:]):
        G.add_edge(n1, n2, reason="chronological_time")

    # (b) discourse cue edges + fallback narrative order between consecutive events
    for (n1, d1), (n2, d2) in zip(nodes, nodes[1:]):
        sent_text = d2["sentence"].lower()
        if any(cue in sent_text for cue in CUE_AFTER):
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, reason="discourse_cue_after")
        elif any(cue in sent_text for cue in CUE_BEFORE):
            if not G.has_edge(n2, n1):
                G.add_edge(n2, n1, reason="discourse_cue_before")
        elif not G.has_edge(n1, n2) and not G.has_edge(n2, n1):
            G.add_edge(n1, n2, reason="narrative_order")

    return G


def temporal_order(G: nx.DiGraph) -> List[str]:
    """Topologically sorts the event graph into a single best-effort timeline.
    If the graph has a cycle (conflicting cues), falls back to sentence order."""
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return sorted(G.nodes(), key=lambda n: G.nodes[n]["sent_index"])


# ----------------------------------------------------------------------------
# 7. WIKIDATA ENRICHMENT
# ----------------------------------------------------------------------------

import requests

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

# Small offline fallback so the app still works without internet access
# (e.g. sandboxed / air-gapped demo environments).
OFFLINE_WIKIDATA_FALLBACK = {
    "India": {"qid": "Q668", "description": "country in South Asia"},
    "Indira Gandhi International Airport": {"qid": "Q1377199", "description": "international airport serving Delhi, India"},
    "Delhi": {"qid": "Q1353", "description": "capital territory of India"},
    "Dubai": {"qid": "Q612", "description": "city in the United Arab Emirates"},
    "United Arab Emirates": {"qid": "Q878", "description": "country in the Middle East"},
    "Emirates": {"qid": "Q334528", "description": "airline based in Dubai, UAE"},
    "Air India": {"qid": "Q1928580", "description": "flag carrier airline of India"},
    "Indian": {"qid": "Q668", "description": "relating to India (nationality/NORP)"},
}


def wikidata_lookup(entity_text: str, timeout: float = 4.0) -> Dict:
    """Looks up an entity string on Wikidata: searches for a matching item,
    then fetches its description. Falls back to a small local dictionary if
    the network call fails (no internet, rate-limited, etc.)."""
    try:
        resp = requests.get(WIKIDATA_SEARCH_URL, params={
            "action": "wbsearchentities", "search": entity_text,
            "language": "en", "format": "json", "limit": 1,
        }, timeout=timeout)
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if results:
            top = results[0]
            return {
                "entity": entity_text,
                "qid": top.get("id"),
                "wikidata_url": f"https://www.wikidata.org/wiki/{top.get('id')}",
                "description": top.get("description", ""),
                "source": "live",
            }
    except Exception:
        pass

    fallback = OFFLINE_WIKIDATA_FALLBACK.get(entity_text)
    if fallback:
        return {
            "entity": entity_text,
            "qid": fallback["qid"],
            "wikidata_url": f"https://www.wikidata.org/wiki/{fallback['qid']}",
            "description": fallback["description"],
            "source": "offline_fallback",
        }
    return {"entity": entity_text, "qid": None, "wikidata_url": None, "description": None, "source": "not_found"}


def enrich_entities_with_wikidata(entities_df: pd.DataFrame, labels_to_enrich=("GPE", "ORG", "NORP", "AIRPORT")) -> pd.DataFrame:
    if entities_df.empty:
        return entities_df.assign(qid=None, wikidata_url=None, wikidata_description=None)

    unique_targets = entities_df[entities_df["label"].isin(labels_to_enrich)]["text"].unique()
    cache = {text: wikidata_lookup(text) for text in unique_targets}

    def _map(row):
        if row["text"] in cache:
            info = cache[row["text"]]
            return pd.Series([info["qid"], info["wikidata_url"], info["description"]])
        return pd.Series([None, None, None])

    entities_df = entities_df.copy()
    entities_df[["qid", "wikidata_url", "wikidata_description"]] = entities_df.apply(_map, axis=1)
    return entities_df


# ----------------------------------------------------------------------------
# 8. ANNOTATED TEXT (HTML rendering, POS + NER combined)
# ----------------------------------------------------------------------------

ENTITY_COLORS = {
    "PERSON": "#ffd6a5", "GPE": "#caffbf", "ORG": "#9bf6ff", "DATE": "#bdb2ff",
    "TIME": "#bdb2ff", "NORP": "#ffc6ff", "PASSPORT_NO": "#ff6b6b",
    "FLIGHT_NO": "#4cc9f0", "GATE": "#f9c74f", "TERMINAL": "#f9844a",
    "SEAT": "#90be6d", "VISA_TYPE": "#f94144", "DOCUMENT": "#577590",
    "AIRPORT": "#43aa8b",
}


def render_annotated_html(doc) -> str:
    """Renders the document as HTML with entities highlighted (color-coded
    by label) and a POS tag shown as a subscript under each token."""
    html_parts = []
    ent_by_start = {ent.start: ent for ent in doc.ents}
    skip_until = -1
    for tok in doc:
        if tok.i < skip_until:
            continue
        if tok.i in ent_by_start:
            ent = ent_by_start[tok.i]
            color = ENTITY_COLORS.get(ent.label_, "#e0e0e0")
            html_parts.append(
                f'<span style="background:{color};border-radius:4px;padding:1px 4px;margin:0 1px;" '
                f'title="{ent.label_}">{ent.text}'
                f'<sub style="font-size:0.65em;color:#333;"> [{ent.label_}]</sub></span>'
            )
            skip_until = ent.end
        else:
            html_parts.append(
                f'{tok.text}<sub style="font-size:0.6em;color:#888;">/{tok.pos_}</sub>'
            )
        html_parts.append(" " if tok.whitespace_ else "")
    return "<div style='line-height:2.4;font-family:sans-serif;'>" + "".join(html_parts) + "</div>"


# ----------------------------------------------------------------------------
# 9. FULL PIPELINE ORCHESTRATION
# ----------------------------------------------------------------------------

def run_pipeline(text: str, model_name: str = "en_core_web_sm", enrich_wikidata: bool = True) -> Dict:
    nlp = load_nlp(model_name)
    doc = nlp(text)

    pos_df = pos_tag(doc)
    ent_df = extract_entities(doc)
    if enrich_wikidata:
        ent_df = enrich_entities_with_wikidata(ent_df)
    rel_df = extract_relations(doc)
    events = extract_events(doc)
    event_df = events_to_df(events)
    graph = build_temporal_graph(events, doc)
    order = temporal_order(graph)
    annotated_html = render_annotated_html(doc)

    return {
        "doc": doc,
        "pos_df": pos_df,
        "entities_df": ent_df,
        "relations_df": rel_df,
        "events_df": event_df,
        "events": events,
        "temporal_graph": graph,
        "temporal_order": order,
        "annotated_html": annotated_html,
    }
