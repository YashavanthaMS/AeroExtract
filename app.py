import sys
import os
import io

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from pipeline import run_pipeline  # noqa: E402

# ----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AeroExtract | Airport IE System", layout="wide", page_icon="🛂")

# Injecting the dashboard theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #0B1020;
    }
    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #253149;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
        color: #E6EDF7;
    }
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {
        color: #9AA8BD;
    }
    input, textarea, [data-baseweb="select"] > div {
        background-color: #151F32 !important;
        border-color: #2B3952 !important;
        color: #E6EDF7 !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #253149;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        border-color: #253149;
        border-radius: 12px;
    }
    .header-card {
        background: linear-gradient(135deg, #17243A 0%, #111A2B 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid #2B3952;
        padding: 30px;
        text-align: center;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2);
        margin-bottom: 28px;
        margin-top: 10px;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        padding-bottom: 5px;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #F3F6FC;
        line-height: 1.3;
    }
    .highlight {
        background: linear-gradient(120deg, #A5B4FC, #67E8F9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .sub-title {
        font-size: 1.25rem;
        color: #A9B7CC;
        margin-top: 8px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Vibrant Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #121A2B;
        border: 1px solid #253149;
        border-radius: 10px;
        padding: 0 18px;
        font-weight: 600;
        color: #A9B7CC;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1A2740;
        color: #DCE5F5;
    }
    .stTabs [aria-selected="true"] {
        background: #26345A !important;
        color: white !important;
        border-color: #6678C8 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6678D8 0%, #5969C2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 650;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 5px 16px rgba(79, 96, 190, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(79, 96, 190, 0.4);
        color: white;
    }
    .stButton>button:focus {
        border-color: #A5B4FC;
        color: white;
        box-shadow: 0 0 0 2px rgba(139, 156, 255, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Custom Dashboard Card Function
# ----------------------------------------------------------------------------
def custom_metric_card(title, value, gradient, icon):
    html = f"""
    <div style="background: {gradient}; padding: 20px; border-radius: 16px; 
                color: white; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.1); 
                transition: transform 0.3s; cursor: pointer;" 
         onmouseover="this.style.transform='translateY(-5px)'" 
         onmouseout="this.style.transform='translateY(0px)'">
        <div style="font-size: 2rem; margin-bottom: 5px;">{icon}</div>
        <h3 style="margin: 0; font-size: 1.1rem; font-weight: 500; opacity: 0.9;">{title}</h3>
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{value}</h1>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>✈️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #DD2476; margin-top: 0;'>AeroExtract</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center;'>Intelligent extraction pipeline for border control and aviation text.</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Pipeline Tweaks")
    enrich = st.toggle(
        "🌍 Enable Wikidata Enrichment", value=True,
        help="Looks up GPE/ORG/NORP/AIRPORT entities on Wikidata. Falls back to offline dict if API fails."
    )
    
    st.markdown("<hr style='border-top: 2px dashed #bbb;'>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Under the Hood", expanded=True):
        st.markdown(
            "🔹 **spaCy** (POS, NER, Dependency)\n\n"
            "🔹 **EntityRuler** (Custom patterns)\n\n"
            "🔹 **Rule-based** Event Extraction\n\n"
            "🔹 **NetworkX** (Temporal Graphs)"
        )

# ----------------------------------------------------------------------------
# Main Layout: Glassmorphism Header & Input
# ----------------------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <h1 class="main-title">🛂 <span class="highlight">AeroExtract</span> Intelligence</h1>
    <p class="sub-title">Automated NLP & Insights for Border Control Data</p>
</div>
""", unsafe_allow_html=True)

with st.container():
    input_mode = st.radio("✨ Choose your data source:", ["Paste text", "Upload .txt file", "Use sample corpus"], horizontal=True)

    text = ""
    if input_mode == "Paste text":
        text = st.text_area("Source Text", value="", height=180,
                            placeholder="e.g. Ravi Kumar, an Indian national, holds passport number K1234567...")
    elif input_mode == "Upload .txt file":
        uploaded = st.file_uploader("Upload a .txt document", type=["txt"])
        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="ignore")
    else:
        SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "sample_corpus.txt")
        default_text = ""
        if os.path.exists(SAMPLE_PATH):
            with open(SAMPLE_PATH) as f:
                default_text = f.read()
        text = default_text
        st.text_area("Sample corpus (editable)", value=text, height=180, key="sample_display")
        text = st.session_state.get("sample_display", text)

# Action Area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.write("") # Spacer
    run_clicked = st.button("🚀 Analyze Text Pipeline", use_container_width=True, disabled=not text.strip())

if not text.strip():
    st.info("👈 Please provide some text or select the sample corpus to begin the magic.")
    st.stop()

if not run_clicked:
    st.stop()

# ----------------------------------------------------------------------------
# Run Pipeline
# ----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

with st.status("🔮 Weaving NLP Magic...", expanded=True) as status:
    st.write("🟢 Running POS Tagging & Dependency Parsing...")
    st.write("🟡 Extracting Colorful Named Entities...")
    st.write("🟠 Mapping Relations & Events...")
    st.write("🟣 Generating Temporal Topology...")
    try:
        result = run_pipeline(text, enrich_wikidata=enrich)
        status.update(label="✨ Analysis Complete!", state="complete", expanded=False)
    except Exception as e:
        status.update(label=f"❌ Pipeline Failed: {e}", state="error", expanded=True)
        st.stop()

# ----------------------------------------------------------------------------
# Colorful Dashboard Metrics
# ----------------------------------------------------------------------------
ent_count = len(result["entities_df"]) if not result["entities_df"].empty else 0
rel_count = len(result["relations_df"]) if not result["relations_df"].empty else 0
evt_count = len(result["events_df"]) if not result["events_df"].empty else 0
graph_nodes = len(result["temporal_graph"].nodes) if result["temporal_graph"] else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    custom_metric_card("Entities Found", ent_count, "linear-gradient(135deg, #253B63 0%, #1D2C49 100%)", "🏷️")
with m2:
    custom_metric_card("Relations Mapped", rel_count, "linear-gradient(135deg, #1E4661 0%, #19354D 100%)", "🔗")
with m3:
    custom_metric_card("Events Detected", evt_count, "linear-gradient(135deg, #245343 0%, #1A3C34 100%)", "📅")
with m4:
    custom_metric_card("Graph Nodes", graph_nodes, "linear-gradient(135deg, #49385F 0%, #302B50 100%)", "🕸️")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Results Tabs
# ----------------------------------------------------------------------------
tabs = st.tabs([
    "🎨 Annotated Text", "🔤 POS Tags", "🏷️ Entities Table", "🔗 Relations",
    "📅 Events", "⏳ Temporal Graph", "📄 Report"
])

# --- Annotated text ---
with tabs[0]:
    st.markdown("### ✨ Visual Entity Highlighting")
    st.markdown("Hover over the colorful spans to view labels. Gray text beneath indicates Part-of-Speech tags.")
    
    with st.container(border=True):
        st.markdown(result["annotated_html"], unsafe_allow_html=True)
    
    st.markdown("##### 🎨 Color Legend")
    legend_items = " &nbsp; ".join(
        f"<span style='background:{c}; padding: 6px 12px; border-radius: 20px; font-size: 0.9em; font-weight: bold; color: #333; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>{lbl}</span>"
        for lbl, c in {
            "PERSON": "#FFD6A5", "GPE": "#CAFFBF", "ORG": "#9BF6FF", "DATE/TIME": "#BDB2FF",
            "NORP": "#FFC6FF", "PASSPORT_NO": "#FF6B6B", "FLIGHT_NO": "#4CC9F0",
            "GATE": "#F9C74F", "TERMINAL": "#F9844A", "AIRPORT": "#43AA8B",
        }.items()
    )
    st.markdown(f"<div style='margin-top: 10px;'>{legend_items}</div>", unsafe_allow_html=True)

# --- POS tags ---
with tabs[1]:
    st.markdown("### 🔤 Part-of-Speech Data")
    st.dataframe(result["pos_df"], use_container_width=True, height=350)

# --- Entities ---
with tabs[2]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🏷️ Extracted Entities")
    ent_df = result["entities_df"]
    
    if ent_df.empty:
        st.info("No entities detected in the provided text.")
    else:
        with col2:
            label_filter = st.multiselect("Filter Labels", sorted(ent_df["label"].unique()),
                                           default=sorted(ent_df["label"].unique()), label_visibility="collapsed")
        
        st.dataframe(ent_df[ent_df["label"].isin(label_filter)], use_container_width=True, height=350)
        
        if "qid" in ent_df.columns and ent_df["qid"].notna().any():
            st.success("🌍 Entities successfully enriched via Wikidata API!")
            
        st.download_button("📥 Download Entities CSV", ent_df.to_csv(index=False), file_name="entities.csv", mime="text/csv")

# --- Relations ---
with tabs[3]:
    st.markdown("### 🔗 Entity Relations")
    rel_df = result["relations_df"]
    if rel_df.empty:
        st.info("No relations detected.")
    else:
        st.dataframe(rel_df, use_container_width=True, height=350)
        st.download_button("📥 Download Relations CSV", rel_df.to_csv(index=False), file_name="relations.csv", mime="text/csv")

# --- Events ---
with tabs[4]:
    st.markdown("### 📅 Extracted Events")
    event_df = result["events_df"]
    if event_df.empty:
        st.info("No events detected.")
    else:
        st.dataframe(event_df, use_container_width=True, height=350)
        st.download_button("📥 Download Events CSV", event_df.to_csv(index=False), file_name="events.csv", mime="text/csv")

# --- Temporal graph ---
with tabs[5]:
    st.markdown("### ⏳ Temporal Topology Graph")
    order = result["temporal_order"]
    G = result["temporal_graph"]

    if not order:
        st.info("No events to order chronologically.")
    else:
        col_list, col_graph = st.columns([1, 2])
        
        with col_list:
            st.markdown("#### 📜 Chronological Sequence")
            for i, node in enumerate(order, 1):
                data = G.nodes[node]
                time_str = f" • <span style='color: #A5B4FC;'>`{data['time_expr']}`</span>" if data.get("time_expr") else ""
                st.markdown(
                    f"<div style='background: #151F32; padding: 12px 14px; border: 1px solid #253149; border-left: 3px solid #818CF8; margin-bottom: 10px; border-radius: 10px;'>"
                    f"<b>{i}. {data['event_type']}</b>{time_str}<br>"
                    f"<span style='font-size: 0.9em; color: #A9B7CC;'>{data['sentence']}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )

        with col_graph:
            st.markdown("#### 🕸️ Directed Graph")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            fig.patch.set_facecolor('#0B1020')
            ax.set_facecolor('#121A2B')
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(False)
            
            pos = nx.spring_layout(G, k=1.5, seed=42)
            labels = {n: G.nodes[n]["event_type"] for n in G.nodes}
            
            node_colors = plt.cm.rainbow([i/max(1, len(G.nodes)) for i in range(len(G.nodes))])
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, edgecolors="#DCE5F5", linewidths=1.5, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color="#71809A", arrowsize=25, arrowstyle="->", width=1.6, connectionstyle="arc3,rad=0.1", ax=ax)
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="#111827", font_weight="bold", ax=ax)
            
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", transparent=False)
            st.download_button("📥 Download Stunning Graph (PNG)", buf.getvalue(), file_name="temporal_graph.png", mime="image/png")

# --- Report ---
with tabs[6]:
    st.markdown("### 📄 Execution Report")
    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("`report.md` not found alongside `app.py`.")