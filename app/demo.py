"""Toxion demo: SMILES canonicalization and structure drawing.

Run locally: streamlit run app/demo.py
"""

from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from toxion.data.loader import canonicalize_smiles

# Page configuration
st.set_page_config(
    page_title="Toxion | Molecular Structure Viewer",
    page_icon="🧪",
    layout="centered",
)

# Custom minimal styling for a clean, modern look
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    @media (prefers-color-scheme: dark) {
        .main-title { color: #F8FAFC; }
    }
    .main-caption {
        font-size: 1rem;
        color: #64748B;
    }
    .sample-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        cursor: pointer;
    }
    .stImage img {
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🧪 Toxion</div>
        <div class="main-caption">SMILES canonicalization and 2D molecular structure drawing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample compounds for quick testing
st.write("**Quick Examples:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Benzene", use_container_width=True):
        st.session_state["input_smiles"] = "c1ccccc1"
with col2:
    if st.button("Aspirin", use_container_width=True):
        st.session_state["input_smiles"] = "CC(=O)Oc1ccccc1C(=O)O"
with col3:
    if st.button("Caffeine", use_container_width=True):
        st.session_state["input_smiles"] = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
with col4:
    if st.button("Paracetamol", use_container_width=True):
        st.session_state["input_smiles"] = "CC(=O)Nc1ccc(O)cc1"

# Input text
current_smiles = st.session_state.get("input_smiles", "c1ccccc1")
smiles_input = st.text_input(
    "SMILES string",
    value=current_smiles,
    placeholder="e.g. c1ccccc1, CC(=O)Oc1ccccc1C(=O)O",
)

# Render Button or auto-render
if smiles_input.strip():
    canonical = canonicalize_smiles(smiles_input.strip())
    if canonical is None:
        st.error("⚠️ RDKit could not parse this SMILES string. Please verify the chemical formula.")
    else:
        mol = Chem.MolFromSmiles(canonical)
        if mol:
            st.divider()

            # Molecular Properties in a neat metric row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Molecular Formula", Chem.rdMolDescriptors.CalcMolFormula(mol))
            with m_col2:
                st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f} g/mol")
            with m_col3:
                st.metric("LogP (Lipophilicity)", f"{Descriptors.MolLogP(mol):.2f}")

            # Structure Drawing
            st.subheader("2D Chemical Structure")
            img = Draw.MolToImage(mol, size=(500, 380))
            st.image(img, use_container_width=True)

            # Canonical SMILES output
            st.subheader("Canonical SMILES")
            st.code(canonical, language="text")
