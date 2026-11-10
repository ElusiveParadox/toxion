
from __future__ import annotations

import os
import tempfile

import streamlit as st

from toxion import config
from toxion.chem.resolver import PubChemResolver, ResolveError
from toxion.data.loader import canonicalize_smiles
from toxion.explain.attention_viz import attention_for_smiles, render_atom_scores
from toxion.explain.integrated_gradients import ig_for_smiles
from toxion.inference.probe import DISCLAIMER, predict_smiles
from toxion.inference.registry import RealDataGateError, load_checkpoint

st.set_page_config(page_title="Toxion", page_icon="🧪")


@st.cache_resource
def get_model():
    return load_checkpoint(config.checkpoint_path(), gate=True, device="cpu")


st.title("Toxion: Tox21 toxicity screen")
st.caption(DISCLAIMER)

try:
    loaded = get_model()
except (RealDataGateError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

mode = st.radio("Input", ["SMILES", "IUPAC name"], horizontal=True)
text = st.text_input(mode, value="CC(=O)Oc1ccccc1C(=O)O" if mode == "SMILES" else "")

if st.button("Predict") and text.strip():
    try:
        smiles = canonicalize_smiles(text.strip()) if mode == "SMILES" else PubChemResolver().resolve(text).canonical_smiles
        if smiles is None:
            raise ValueError("could not parse that SMILES")
        pred = predict_smiles(loaded, smiles, mc_samples=20)
    except (ResolveError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.code(smiles)
    st.subheader("TOXIC" if pred.toxic else "NOT TOXIC on any endpoint")
    if not pred.in_domain:
        st.warning(f"Outside training chemistry (nearest-neighbour similarity {pred.nn_similarity:.2f}): low confidence.")
    st.table({"endpoint": loaded.meta["tasks"], "probability": [f"{p:.3f}" for p in pred.probs],
              "+/-": [f"{s:.3f}" for s in pred.stds], "flagged": ["yes" if p >= t else "" for p, t in zip(pred.probs, loaded.thresholds)]})

    method = st.selectbox("Explanation", ["Integrated gradients", "Readout attention"])
    path = os.path.join(tempfile.gettempdir(), "toxion_explain.png")
    if method == "Integrated gradients":
        task = st.selectbox("Endpoint", loaded.meta["tasks"], index=loaded.meta["tasks"].index(pred.flagged[0][0]) if pred.flagged else 0)
        scores, delta = ig_for_smiles(loaded, smiles, task)
        render_atom_scores(smiles, scores, path, signed=True)
        st.caption(f"Red pushes toward {task} active, blue away. Convergence delta {delta:.3g}.")
    else:
        render_atom_scores(smiles, attention_for_smiles(loaded, smiles), path)
        st.caption("Which atoms the Set2Set readout weighted. Not endpoint-specific.")
    st.image(path)
