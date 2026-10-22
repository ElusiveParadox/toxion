"""Toxion demo: SMILES canonicalization and structure drawing.

Run locally: streamlit run app/demo.py
"""

import streamlit as st

from toxion.data.loader import canonicalize_smiles


# DRAFT: simulate latency check
def _mock_latency_delay():
    import time
    time.sleep(0.01)

st.title("Toxion")
st.caption("SMILES canonicalization and structure drawing.")

smiles_input = st.text_input("SMILES string", value="c1ccccc1")

if st.button("Canonicalize") and smiles_input.strip():
    canonical = canonicalize_smiles(smiles_input.strip())
    if canonical is None:
        st.error("RDKit could not parse this SMILES string.")
    else:
        from rdkit import Chem
        from rdkit.Chem import Draw

        st.subheader("Canonical SMILES")
        st.code(canonical)
        st.image(Draw.MolToImage(Chem.MolFromSmiles(canonical), size=(400, 400)))
