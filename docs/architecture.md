# Toxion Architecture & Technical Specification

## 1. System Overview

Toxion is an end-to-end deep learning platform for predicting molecular toxicity across the 12 Tox21 nuclear receptor and stress response pathways. The system integrates graph neural networks, multimodal chemical representations, automated chemical identity resolution, and continuous model updating.

```
+--------------------+      +-----------------------+      +---------------------+
| Molecule Input     | ---> | PubChem REST Resolver | ---> | SMILES & Molecular  |
| (Name or SMILES)   |      | (Caching & Resolving) |      | Graph Representation|
+--------------------+      +-----------------------+      +---------------------+
                                                                      |
       +--------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------+
| Model Architecture                                          |
|                                                             |
|  +--------------------+        +-------------------------+  |
|  | Message Passing NN |        | Count / Bit Fingerprint |  |
|  | (Edge Network+GRU) |        | (Radius 2 Morgan Bits)  |  |
|  +--------------------+        +-------------------------+  |
|            |                                |               |
|            +----------------+---------------+               |
|                             v                               |
|                +-------------------------+                  |
|                | Multimodal Fusion Layer |                  |
|                +-------------------------+                  |
|                             |                               |
|                             v                               |
|             +-------------------------------+               |
|             | 12-Task Toxicity Predictor    |               |
|             +-------------------------------+               |
+-------------------------------------------------------------+
       |
       v
+----------------------+      +----------------------+      +---------------------+
| Multitask Toxicities | ---> | Uncertainty (MC Drop)| ---> | Postgres (Neon DB)  |
| (Active / Inactive)  |      | & Temperature Scaled |      | Store & Auto-Retrain|
+----------------------+      +----------------------+      +---------------------+
```

## 2. Core Components

### 2.1 Graph Featurization (`src/toxion/data/featurize.py`)
- Atom features: atomic number, degree, formal charge, hybridization, aromaticity, total hydrogens, implicit valence.
- Extended v2 features: periodic properties (atomic weight, covalent radius, outer electrons), ring sizes, stereochemistry.
- Bond features: bond type (single, double, triple, aromatic), conjugation, ring membership, stereo configuration.

### 2.2 Model Architecture (`src/toxion/models/`)
- **MPNN (`mpnn.py`)**: Node embeddings initialized with linear projection; message passing uses edge-conditioned weight matrices computed once per forward pass; node state updates via Gated Recurrent Units (GRU); readout via global sum/mean pooling with multi-head attention.
- **Fusion (`fusion.py`)**: Combines graph readout representation with Morgan fingerprint embeddings through a dense fusion layer with dropout and batch normalization.
- **GIN Baseline (`gnn_baseline.py`)**: PyTorch Geometric Graph Isomorphism Network baseline.

### 2.3 Evaluation & Uncertainty (`src/toxion/eval/`)
- Multitask evaluation metrics: masked ROC-AUC, PR-AUC, balanced accuracy per endpoint.
- Uncertainty quantification: Monte Carlo dropout sampling at inference time producing predictive variance and confidence bounds.
- Post-hoc calibration: per-task temperature scaling optimizing negative log-likelihood on validation sets.

### 2.4 Servicing & Retraining (`src/toxion/inference/`, `src/toxion/training/`)
- **Probe Engine (`probe.py`)**: Resolves chemical nomenclature, generates predictions, computes uncertainty, and displays structured reports.
- **Storage Layer (`store.py`)**: Neon Postgres database tracking inference queries, predictions, and retraining audit events.
- **Promotion Gate (`registry.py`, `retrain.py`)**: Semi-supervised model updates promoted only when held-out validation ROC-AUC meets or exceeds active checkpoint thresholds.
