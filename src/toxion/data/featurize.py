
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "NumHeteroatoms",
]
N_FP_BITS = 2048
ATOM_TYPES = ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I", "P"]
N_HYBRIDIZATIONS = 3
N_BOND_TYPES = 4

TOXICOPHORE_SMARTS = [
    ("michael_acceptor", "[CX3]=[CX3][CX3]=O"),
    ("aromatic_amine", "[NX3;H2,H1;!$(NC=O)][c]"),
    ("epoxide", "C1OC1"),
    ("nitro_aromatic", "[c][NX3](=O)=O"),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("azo", "[#6][NX2]=[NX2][#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("alkyl_halide", "[CX4][F,Cl,Br,I]"),
]

EXTRA_ELEMENTS = ["Si", "B", "Sn", "Hg", "As", "Se", "Zn"]
PRETRAIN_ELEMENTS = ATOM_TYPES + EXTRA_ELEMENTS
CHARGES = [-2, -1, 0, 1, 2]
RING_SIZES = [3, 4, 5, 6, 7]
NODE_FEATURE_DIM_V2 = 21 + len(EXTRA_ELEMENTS) + (len(CHARGES) + 1) + (1 + len(RING_SIZES)) + 4 + 1 + 5
EDGE_FEATURE_DIM_V2 = 5 + 1 + 1 + 3
FEATURE_VERSION = 2


def node_dim(version: int) -> int:
    return {1: 21, 2: NODE_FEATURE_DIM_V2}[version]


def edge_dim(version: int) -> int:
    return {1: 5, 2: EDGE_FEATURE_DIM_V2}[version]


NODE_FEATURE_DIM = (len(ATOM_TYPES) + 1) + 1 + 1 + 1 + 1 + (N_HYBRIDIZATIONS + 1) + 1 + 1
EDGE_FEATURE_DIM = N_BOND_TYPES + 1
FLAT_FEATURE_DIM = N_FP_BITS + len(RDKIT_DESCRIPTOR_NAMES) + len(TOXICOPHORE_SMARTS)
DESC_SLICE = slice(N_FP_BITS, N_FP_BITS + len(RDKIT_DESCRIPTOR_NAMES))

_cache: dict = {}


def _mol(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable SMILES: {smiles}")
    return mol


def _fp_generator():
    if "fp" not in _cache:
        from rdkit.Chem import rdFingerprintGenerator

        _cache["fp"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_FP_BITS)
    return _cache["fp"]


def _toxicophore_patterns():
    if "tox" not in _cache:
        from rdkit import Chem

        _cache["tox"] = [Chem.MolFromSmarts(s) for _, s in TOXICOPHORE_SMARTS]
    return _cache["tox"]


def _feature_factory():
    if "fdef" not in _cache:
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures

        _cache["fdef"] = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        )
    return _cache["fdef"]


def featurize_fingerprint(smiles: str, counts: bool = True) -> np.ndarray:
    mol = _mol(smiles)
    gen = _fp_generator()
    arr = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
    return arr.astype(np.float32)


def morgan_on_bits(smiles: str) -> dict[int, int]:
    arr = featurize_fingerprint(smiles)
    return {int(i): int(arr[i]) for i in np.nonzero(arr)[0]}


def featurize_descriptors(smiles: str) -> np.ndarray:
    from rdkit.Chem import Descriptors

    mol = _mol(smiles)
    values = [getattr(Descriptors, name)(mol) for name in RDKIT_DESCRIPTOR_NAMES]
    return np.array(values, dtype=np.float32)


def featurize_toxicophore_alerts(smiles: str) -> np.ndarray:
    mol = _mol(smiles)
    out = np.zeros(len(TOXICOPHORE_SMARTS), dtype=np.float32)
    for i, pattern in enumerate(_toxicophore_patterns()):
        if pattern is not None and mol.HasSubstructMatch(pattern):
            out[i] = 1.0
    return out


def featurize_flat(smiles: str) -> np.ndarray:
    return np.concatenate([
        featurize_fingerprint(smiles),
        featurize_descriptors(smiles),
        featurize_toxicophore_alerts(smiles),
    ])


def _one_hot(value, choices) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else -1] = 1.0
    return vec


@dataclass
class MolGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


def featurize_graph(smiles: str, version: int = 1) -> MolGraph:
    if version not in (1, 2):
        raise ValueError(f"unknown feature version {version}")
    from rdkit import Chem

    mol = _mol(smiles)
    factory = _feature_factory()
    donors, acceptors = set(), set()
    for feat in factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == "Donor":
            donors.update(feat.GetAtomIds())
        elif feat.GetFamily() == "Acceptor":
            acceptors.update(feat.GetAtomIds())

    hybs = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

    nodes = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        f = _one_hot(atom.GetSymbol(), ATOM_TYPES)
        f.append(atom.GetAtomicNum() / 10.0)
        f.append(1.0 if idx in acceptors else 0.0)
        f.append(1.0 if idx in donors else 0.0)
        f.append(1.0 if atom.GetIsAromatic() else 0.0)
        f += _one_hot(atom.GetHybridization(), hybs)
        f.append(float(atom.GetTotalNumHs()))
        f.append(float(atom.GetDegree()))
        if version == 2:
            f += _node_extras(atom)
        nodes.append(f)
    node_features = np.array(nodes, dtype=np.float32)

    src, dst, efeats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _one_hot(bond.GetBondType(), bond_types)
        if version == 2:
            bf = bf + _edge_extras(bond)
        src += [i, j]
        dst += [j, i]
        efeats += [bf, bf]
    if not src:
        return MolGraph(node_features, np.zeros((2, 0), dtype=np.int64),
                        np.zeros((0, edge_dim(version)), dtype=np.float32))
    return MolGraph(node_features, np.array([src, dst], dtype=np.int64),
                    np.array(efeats, dtype=np.float32))


def _period(z: int) -> int:
    for period, last in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if z <= last:
            return period
    return 7


def _node_extras(atom) -> list[float]:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    z = atom.GetAtomicNum()
    f = _one_hot(atom.GetSymbol(), EXTRA_ELEMENTS)[:-1]
    f += _one_hot(atom.GetFormalCharge(), CHARGES)
    f.append(1.0 if atom.IsInRing() else 0.0)
    f += [1.0 if atom.IsInRingSize(n) else 0.0 for n in RING_SIZES]
    tag = atom.GetChiralTag()
    f += [1.0 if tag == Chem.ChiralType.CHI_UNSPECIFIED else 0.0,
          1.0 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0,
          1.0 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0,
          1.0 if tag not in (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW) else 0.0]
    f.append(float(atom.GetNumRadicalElectrons()))
    f += [pt.GetAtomicWeight(z) / 100.0, pt.GetRcovalent(z), pt.GetRvdw(z), pt.GetNOuterElecs(z) / 8.0, _period(z) / 7.0]
    return f


def _edge_extras(bond) -> list[float]:
    from rdkit import Chem

    st = bond.GetStereo()
    return [1.0 if bond.GetIsConjugated() else 0.0, 1.0 if bond.IsInRing() else 0.0,
            1.0 if st == Chem.BondStereo.STEREONONE else 0.0,
            1.0 if st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOCIS,
                          Chem.BondStereo.STEREOTRANS) else 0.0,
            1.0 if st == Chem.BondStereo.STEREOANY else 0.0]


def atom_element_classes(smiles: str) -> np.ndarray:
    mol = _mol(smiles)
    n = len(PRETRAIN_ELEMENTS)
    return np.array([PRETRAIN_ELEMENTS.index(a.GetSymbol()) if a.GetSymbol() in PRETRAIN_ELEMENTS else n
                     for a in mol.GetAtoms()], dtype=np.int64)


class FlatScaler:

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)

    @classmethod
    def fit(cls, flat_train: np.ndarray) -> "FlatScaler":
        d = flat_train[:, DESC_SLICE]
        return cls(d.mean(axis=0), d.std(axis=0))

    def transform(self, flat: np.ndarray) -> np.ndarray:
        out = np.array(flat, dtype=np.float32, copy=True)
        out[..., :N_FP_BITS] = np.log1p(out[..., :N_FP_BITS])
        out[..., DESC_SLICE] = (out[..., DESC_SLICE] - self.mean) / self.std
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FlatScaler":
        return cls(np.array(d["mean"]), np.array(d["std"]))
