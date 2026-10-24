# Airport & Passport Information Extraction System

An end-to-end information extraction pipeline (POS tagging → NER → relation
extraction → event extraction → temporal ordering, with Wikidata entity
grounding) for airport/passport-domain text, wrapped in a Streamlit app.

## Project layout

```
airport_ie_system/
├── app.py                     # Streamlit frontend
├── report.md                  # pipeline write-up (also rendered inside the app)
├── requirements.txt
├── backend/
│   ├── pipeline.ipynb          # backend developed & validated step-by-step here
│   └── pipeline.py             # consolidated module imported by app.py
├── sample_data/
│   └── sample_corpus.txt       # synthetic demo text (no real passport data)
└── outputs/                    # generated artifacts (e.g. temporal_graph.png)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the backend notebook (optional — see how the pipeline was built)

```bash
jupyter notebook backend/pipeline.ipynb
```

## Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

In the app you can:
- Paste text, upload a `.txt` file, or use the bundled sample corpus.
- Toggle live Wikidata enrichment on/off (needs internet; falls back to an
  offline dictionary otherwise).
- Browse annotated text, POS tags, entities, relations, events, and the
  event timeline / temporal graph, each downloadable as CSV/PNG.

## Notes on Wikidata access

This project was developed in a network-restricted sandbox, so live Wikidata
lookups were not exercised end-to-end there — `backend/pipeline.py` includes
a small offline fallback dictionary for exactly that situation. When you run
`streamlit run app.py` on a machine with normal internet access, live lookups
against `https://www.wikidata.org/w/api.php` work automatically.
