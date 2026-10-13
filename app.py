import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as bg

# -----------------------------------------------------------------------------
# 1. PAGE & THEME CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ILD Graph-RAG & k-NN Diagnostic Assistant (by Dr.A)",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #1E222D; padding: 12px; border-radius: 8px; border: 1px solid #2E3440; }
    .card { background-color: #1E222D; padding: 18px; border-radius: 10px; border: 1px solid #3B4252; margin-bottom: 15px; }
    .badge-interstitial { background-color: #1D4ED8; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .badge-alveolar { background-color: #047857; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .badge-rare { background-color: #6D28D9; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 2026 ERS/ATS KNOWLEDGE GRAPH & REFERENCE COHORT ENGINE
# -----------------------------------------------------------------------------
class ILDDiagnosticEngine2026:
    """
    Hybrid Graph-RAG and k-NN Vector Search Engine derived from Diagn Interv Radiol 2026; 32(5):643-654.
    Taxonomy encompasses Interstitial Patterns, Alveolar Filling Disorders, and Rare/Unclassified Disorders.
    """
    
    FEATURE_KEYS = [
        "ggo", "reticulation", "honeycombing", "consolidation",
        "airway_nodules", "crazy_paving", "cysts_calc", "fat_attenuation"
    ]

    def __init__(self):
        self.graph = nx.DiGraph()
        self.cohort = []
        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self._build_knowledge_graph()
        self._seed_reference_cohort()
        self._fit_knn()

    def _build_knowledge_graph(self):
        # 1. Primary Classification Categories
        categories = ["Interstitial Patterns", "Alveolar Filling Disorders", "Rare and Unclassified Disorders"]
        for cat in categories:
            self.graph.add_node(cat, type="Category")

        # 2. Subtypes / Disease Entities from Literature Table 1
        diseases = {
            # Interstitial Patterns
            "Usual Interstitial Pneumonia (UIP)": (
                "Interstitial Patterns",
                "Basal & subpleural fibrosis, honeycombing, traction bronchiectasis. Histopathologic/radiologic hallmark of IPF."
            ),
            "Non-Specific Interstitial Pneumonia (NSIP)": (
                "Interstitial Patterns",
                "Diffuse GGOs +/- fine reticulation, traction bronchiectasis in fibrotic subtype, lower-lobe predominance, possible subpleural sparing."
            ),
            "Bronchiolocentric Interstitial Pneumonia (BIP)": (
                "Interstitial Patterns",
                "Peribronchiolar reticulation, centrilobular nodules, tree-in-bud opacities; air trapping on expiratory imaging."
            ),
            "Diffuse Alveolar Damage (DAD)": (
                "Interstitial Patterns",
                "Diffuse GGOs and patchy consolidation with dependent predominance reflecting acute lung injury."
            ),

            # Alveolar Filling Disorders
            "Organizing Pneumonia (OP)": (
                "Alveolar Filling Disorders",
                "Peripheral or peribronchovascular consolidations, perilobular or arcade-like opacities, reversed halo sign."
            ),
            "Respiratory Bronchiolitis-ILD (RB-ILD)": (
                "Alveolar Filling Disorders",
                "Ill-defined centrilobular GGO nodules and patchy GGOs, often upper-lobe predominant in smokers."
            ),
            "Alveolar Macrophage Pneumonia (AMP)": (
                "Alveolar Filling Disorders",
                "Diffuse GGOs with basal predominance, mild reticulation, occasional small cysts (formerly DIP)."
            ),
            "Eosinophilic Pneumonia (AEP/CEP)": (
                "Alveolar Filling Disorders",
                "AEP: diffuse bilateral GGOs/consolidation. CEP: peripheral upper-lobe consolidations ('photographic negative' of pulmonary edema)."
            ),
            "Pulmonary Alveolar Proteinosis (PAP)": (
                "Alveolar Filling Disorders",
                "Crazy-paving pattern (GGOs with superimposed smooth interlobular/intralobular septal thickening)."
            ),
            "Lipoid Pneumonia (LP)": (
                "Alveolar Filling Disorders",
                "Exogenous/endogenous lipid accumulation presenting as consolidations or nodules with fat attenuation (< -30 HU)."
            ),

            # Rare and Unclassified Disorders
            "Pulmonary Alveolar Microlithiasis (PAM)": (
                "Rare and Unclassified Disorders",
                "Diffuse sand-like calcified micronodules with subpleural involvement and black pleura sign."
            ),
            "Pleuroparenchymal Fibroelastosis (PPFE)": (
                "Rare and Unclassified Disorders",
                "Dense subpleural upper-lobe fibroelastosis, volume loss, platythorax."
            ),
            "Lymphoid Interstitial Pneumonia (LIP)": (
                "Rare and Unclassified Disorders",
                "GGOs, interlobular septal thickening, and scattered thin-walled cysts."
            ),
            "Pulmonary Amyloidosis": (
                "Rare and Unclassified Disorders",
                "Pulmonary nodules +/- calcification, tracheobronchial wall thickening, and interstitial infiltration."
            ),
            "Unclassifiable ILD": (
                "Rare and Unclassified Disorders",
                "Mixed or overlapping patterns, often combining fibrotic changes with OP-like features."
            )
        }

        for disease, (cat, desc) in diseases.items():
            self.graph.add_node(disease, type="Disease", description=desc)
            self.graph.add_edge(cat, disease, relationship="INCLUDES")

        # 3. Radiologic Signs & Associations
        signs = [
            ("Usual Interstitial Pneumonia (UIP)", "Subpleural Basal Honeycombing", 1.0),
            ("Usual Interstitial Pneumonia (UIP)", "Subpleural Sparing", -0.9),
            ("Non-Specific Interstitial Pneumonia (NSIP)", "Subpleural Sparing", 0.92),
            ("Bronchiolocentric Interstitial Pneumonia (BIP)", "Tree-in-Bud Opacities", 0.95),
            ("Bronchiolocentric Interstitial Pneumonia (BIP)", "Centrilobular Nodules", 0.90),
            ("Organizing Pneumonia (OP)", "Reversed Halo Sign", 0.92),
            ("Pulmonary Alveolar Proteinosis (PAP)", "Crazy-Paving Pattern", 0.96),
            ("Lipoid Pneumonia (LP)", "Fat Attenuation (< -30 HU)", 1.00),
            ("Pulmonary Alveolar Microlithiasis (PAM)", "Black Pleura Sign", 0.94),
            ("Pleuroparenchymal Fibroelastosis (PPFE)", "Platythorax / Apical Fibrosis", 0.91)
        ]

        for disease, sign, weight in signs:
            self.graph.add_node(sign, type="Sign")
            self.graph.add_edge(disease, sign, weight=weight)

    def _seed_reference_cohort(self):
        # 8D Reference Cohort Vectors:
        # [GGO, Reticulation, Honeycombing, Consolidation, AirwayNodules, CrazyPaving, Cysts, FatAttenuation]
        self.cohort = [
            {"id": "REF-UIP-01", "diagnosis": "Usual Interstitial Pneumonia (UIP)", "category": "Interstitial Patterns", "vector": [10, 50, 35, 0, 0, 0, 0, 0]},
            {"id": "REF-UIP-02", "diagnosis": "Usual Interstitial Pneumonia (UIP)", "category": "Interstitial Patterns", "vector": [5, 45, 40, 0, 0, 0, 0, 0]},
            {"id": "REF-NSIP-01", "diagnosis": "Non-Specific Interstitial Pneumonia (NSIP)", "category": "Interstitial Patterns", "vector": [60, 30, 0, 5, 0, 0, 0, 0]},
            {"id": "REF-NSIP-02", "diagnosis": "Non-Specific Interstitial Pneumonia (NSIP)", "category": "Interstitial Patterns", "vector": [45, 35, 0, 0, 0, 0, 0, 0]},
            {"id": "REF-BIP-01", "diagnosis": "Bronchiolocentric Interstitial Pneumonia (BIP)", "category": "Interstitial Patterns", "vector": [20, 30, 0, 10, 40, 0, 0, 0]},
            {"id": "REF-DAD-01", "diagnosis": "Diffuse Alveolar Damage (DAD)", "category": "Interstitial Patterns", "vector": [50, 10, 0, 40, 0, 0, 0, 0]},
            {"id": "REF-OP-01", "diagnosis": "Organizing Pneumonia (OP)", "category": "Alveolar Filling Disorders", "vector": [20, 0, 0, 65, 10, 0, 0, 0]},
            {"id": "REF-RB-01", "diagnosis": "Respiratory Bronchiolitis-ILD (RB-ILD)", "category": "Alveolar Filling Disorders", "vector": [40, 5, 0, 0, 35, 0, 0, 0]},
            {"id": "REF-AMP-01", "diagnosis": "Alveolar Macrophage Pneumonia (AMP)", "category": "Alveolar Filling Disorders", "vector": [65, 15, 0, 0, 0, 0, 10, 0]},
            {"id": "REF-CEP-01", "diagnosis": "Eosinophilic Pneumonia (AEP/CEP)", "category": "Alveolar Filling Disorders", "vector": [30, 0, 0, 60, 0, 0, 0, 0]},
            {"id": "REF-PAP-01", "diagnosis": "Pulmonary Alveolar Proteinosis (PAP)", "category": "Alveolar Filling Disorders", "vector": [25, 0, 0, 0, 0, 75, 0, 0]},
            {"id": "REF-LP-01", "diagnosis": "Lipoid Pneumonia (LP)", "category": "Alveolar Filling Disorders", "vector": [15, 0, 0, 55, 0, 0, 0, 1]},
            {"id": "REF-PAM-01", "diagnosis": "Pulmonary Alveolar Microlithiasis (PAM)", "category": "Rare and Unclassified Disorders", "vector": [5, 10, 0, 0, 0, 0, 85, 0]},
            {"id": "REF-PPFE-01", "diagnosis": "Pleuroparenchymal Fibroelastosis (PPFE)", "category": "Rare and Unclassified Disorders", "vector": [0, 60, 0, 20, 0, 0, 0, 0]},
            {"id": "REF-LIP-01", "diagnosis": "Lymphoid Interstitial Pneumonia (LIP)", "category": "Rare and Unclassified Disorders", "vector": [35, 20, 0, 0, 0, 0, 40, 0]}
        ]

    def _fit_knn(self):
        X = np.array([c["vector"] for c in self.cohort])
        self.knn.fit(X)

    def evaluate_case(self, input_vector: list, active_signs: list, k: int = 5) -> dict:
        distances, indices = self.knn.kneighbors([input_vector], n_neighbors=min(k, len(self.cohort)))
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            c = self.cohort[idx]
            sim = 1.0 / (1.0 + dist)
            matches.append({
                "id": c["id"],
                "diagnosis": c["diagnosis"],
                "category": c["category"],
                "similarity": sim,
                "distance": float(dist),
                "vector": c["vector"]
            })

        # Calculate primary candidate & path weights
        primary_diag = matches[0]["diagnosis"]
        category = list(self.graph.predecessors(primary_diag))[0]
        desc = self.graph.nodes[primary_diag].get("description", "")

        # Sign-based score adjustments
        sign_adjustments = {}
        for diag in self.graph.nodes:
            if self.graph.nodes[diag].get("type") == "Disease":
                score = 0.0
                for sign in active_signs:
                    if self.graph.has_edge(diag, sign):
                        score += self.graph[diag][sign]["weight"]
                if score != 0.0:
                    sign_adjustments[diag] = score

        # Strict physics/density overrides
        if input_vector[7] == 1:
            primary_diag = "Lipoid Pneumonia (LP)"
            category = "Alveolar Filling Disorders"
            desc = "Exogenous or endogenous lipid accumulation presenting with fat attenuation (< -30 HU)."

        return {
            "primary_diagnosis": primary_diag,
            "category": category,
            "description": desc,
            "knn_matches": matches,
            "sign_adjustments": sign_adjustments
        }

@st.cache_resource
def get_engine():
    return ILDDiagnosticEngine2026()

engine = get_engine()

# -----------------------------------------------------------------------------
# 3. INTERACTIVE DASHBOARD SIDEBAR & CONTROLS
# -----------------------------------------------------------------------------
st.title("🫁 ILD Graph-RAG & k-NN Diagnostic Assistant")
st.caption("Derived from 2026 ERS/ATS Classification Guidelines (Diagn Interv Radiol 2026; 32(5):643–654)")

with st.sidebar:
    st.header("⚙️ Configuration")
    k_value = st.slider("k-Nearest Neighbors (k)", min_value=3, max_value=7, value=5, step=2)
    
    st.subheader("📋 Clinical Presets")
    preset = st.selectbox(
        "Load Representative Pattern",
        ["Custom", "Classic UIP / IPF", "Fibrotic NSIP", "Bronchiolocentric (BIP)", "Crazy-Paving (PAP)", "Lipoid Pneumonia (LP)", "Microlithiasis (PAM)"]
    )

    # Preset Value Maps
    default_vals = {
        "Custom": [20, 35, 0, 10, 35, 0, 0, 0],
        "Classic UIP / IPF": [5, 45, 35, 0, 0, 0, 0, 0],
        "Fibrotic NSIP": [55, 30, 0, 5, 0, 0, 0, 0],
        "Bronchiolocentric (BIP)": [20, 30, 0, 10, 40, 0, 0, 0],
        "Crazy-Paving (PAP)": [25, 0, 0, 0, 0, 75, 0, 0],
        "Lipoid Pneumonia (LP)": [15, 0, 0, 55, 0, 0, 0, 1],
        "Microlithiasis (PAM)": [5, 10, 0, 0, 0, 0, 85, 0]
    }
    vals = default_vals[preset]

# Main 2-Column Interface
col_inputs, col_results = st.columns([1, 1.2])

with col_inputs:
    st.markdown("### 1. Quantitative Pattern Involvement (%)")
    
    ggo = st.slider("Ground-Glass Opacity (GGO)", 0, 100, vals[0])
    retic = st.slider("Fine/Coarse Reticulation", 0, 100, vals[1])
    honeycomb = st.slider("Honeycombing Cysts", 0, 100, vals[2])
    consolidation = st.slider("Parenchymal Consolidation", 0, 100, vals[3])
    airway_nodules = st.slider("Centrilobular / Tree-in-Bud Nodules", 0, 100, vals[4])
    crazy_paving = st.slider("Crazy-Paving (GGO + Septal Thickening)", 0, 100, vals[5])
    cysts_calc = st.slider("Thin-Walled Cysts / Micro-calcifications", 0, 100, vals[6])

    st.markdown("### 2. Ancillary & Quantitative Features")
    fat_att = st.checkbox("Fat Attenuation Present (< -30 HU)", value=bool(vals[7]))
    
    active_signs = st.multiselect(
        "Observed Radiological Signs",
        options=[
            "Subpleural Sparing",
            "Tree-in-Bud Opacities",
            "Centrilobular Nodules",
            "Reversed Halo Sign",
            "Crazy-Paving Pattern",
            "Black Pleura Sign",
            "Platythorax / Apical Fibrosis"
        ]
    )

    fat_flag = 1 if fat_att else 0
    input_vector = [ggo, retic, honeycomb, consolidation, airway_nodules, crazy_paving, cysts_calc, fat_flag]
    total_pct = sum(input_vector[:7])
    
    if total_pct > 100:
        st.warning(f"⚠️ Total pattern volume ({total_pct}%) exceeds 100%. Adjust sliders for accurate proportional tracking.")
    else:
        st.caption(f"Total Monitored Parenchymal Volume: **{total_pct}%**")

with col_results:
    st.markdown("### 3. Diagnostic Reasoning & Traversal")
    
    eval_result = engine.evaluate_case(input_vector, active_signs, k=k_value)
    
    # Category badge color mapping
    badge_class = "badge-interstitial"
    if eval_result["category"] == "Alveolar Filling Disorders":
        badge_class = "badge-alveolar"
    elif eval_result["category"] == "Rare and Unclassified Disorders":
        badge_class = "badge-rare"

    st.markdown(f"""
        <div class="card">
            <span style="color: #9CA3AF; font-size: 0.85rem; font-weight: 600;">PRIMARY DIAGNOSTIC CONSIDERATION</span>
            <h2 style="color: #60A5FA; margin-top: 4px; margin-bottom: 8px;">{eval_result['primary_diagnosis']}</h2>
            <span class="{badge_class}">{eval_result['category']}</span>
            <p style="margin-top: 12px; font-size: 0.92rem; color: #D1D5DB; line-height: 1.5;">{eval_result['description']}</p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABULAR RESULTS & LATENT SCATTER PLOT
    # -------------------------------------------------------------------------
    tab_knn, tab_scatter, tab_graph = st.tabs(["📊 k-NN Cohort Matches", "🌌 2D Vector Space Scatter", "🕸️ Knowledge Graph"])

    with tab_knn:
        df_knn = pd.DataFrame(eval_result["knn_matches"])
        df_knn["similarity_pct"] = (df_knn["similarity"] * 100).round(1).astype(str) + "%"
        df_display = df_knn[["id", "diagnosis", "category", "similarity_pct", "distance"]].rename(columns={
            "id": "Cohort ID",
            "diagnosis": "Matched Disease Entity",
            "category": "Taxonomy Category",
            "similarity_pct": "Vector Similarity",
            "distance": "Euclidean Distance"
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        if eval_result["sign_adjustments"]:
            st.markdown("#### Path Confidence Adjustments")
            for diag, adj in eval_result["sign_adjustments"].items():
                st.info(f"**{diag}**: Path weight adjusted by **+{adj:.2f}** based on active radiologic signs.")

    with tab_scatter:
        # Simple 2D projection using PCA-like linear combination across key feature sets
        cohort_vectors = [c["vector"] for c in engine.cohort] + [input_vector]
        labels = [c["diagnosis"] for c in engine.cohort] + ["CURRENT PATIENT"]
        cats = [c["category"] for c in engine.cohort] + ["Patient"]
        ids = [c["id"] for c in engine.cohort] + ["PATIENT"]

        X = np.array(cohort_vectors)
        # Projection Axes: X = Fibrotic vs Inflammatory, Y = Alveolar/Cystic vs Reticular
        x_proj = (X[:, 1]*1.1 + X[:, 2]*1.5) - (X[:, 0]*0.8 + X[:, 3]*1.2 + X[:, 6]*0.5)
        y_proj = (X[:, 0]*1.2 + X[:, 3]*1.4 + X[:, 5]*1.5) - (X[:, 1]*0.8 + X[:, 2]*0.9)

        df_plot = pd.DataFrame({
            "x": x_proj,
            "y": y_proj,
            "Diagnosis": labels,
            "Category": cats,
            "ID": ids,
            "Size": [10 if i < len(engine.cohort) else 20 for i in range(len(labels))]
        })

        fig = px.scatter(
            df_plot, x="x", y="y", color="Category", hover_data=["ID", "Diagnosis"],
            size="Size", title="2D Latent Pattern Vector Space Projection",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_graph:
        st.markdown("#### 2026 ERS/ATS Diagnostic Taxonomy Map")
        st.code("""
        [Interstitial Patterns]
        ├── Usual Interstitial Pneumonia (UIP)
        ├── Non-Specific Interstitial Pneumonia (NSIP)
        ├── Bronchiolocentric Interstitial Pneumonia (BIP)
        └── Diffuse Alveolar Damage (DAD)

        [Alveolar Filling Disorders]
        ├── Organizing Pneumonia (OP)
        ├── Respiratory Bronchiolitis-ILD (RB-ILD)
        ├── Alveolar Macrophage Pneumonia (AMP)
        ├── Eosinophilic Pneumonia (AEP/CEP)
        ├── Pulmonary Alveolar Proteinosis (PAP)
        └── Lipoid Pneumonia (LP)

        [Rare and Unclassified Disorders]
        ├── Pulmonary Alveolar Microlithiasis (PAM)
        ├── Pleuroparenchymal Fibroelastosis (PPFE)
        ├── Lymphoid Interstitial Pneumonia (LIP)
        ├── Pulmonary Amyloidosis
        └── Unclassifiable ILD
        """, language="text")