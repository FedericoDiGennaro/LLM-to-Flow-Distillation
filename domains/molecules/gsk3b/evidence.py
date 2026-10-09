"""Cheap RDKit identity, descriptor, functional-group and geometry evidence."""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdchem import Mol


FUNCTIONAL_GROUP_SMARTS: tuple[tuple[str, str], ...] = (
    ("primary_amine", "[NX3;H2;!$(NC=O)]"),
    ("secondary_amine", "[NX3;H1;!$(NC=O)]"),
    ("tertiary_amine", "[NX3;H0;!$(NC=O)]"),
    ("amide", "[NX3,n][#6](=[OX1])"),
    ("carbonyl", "[#6]=[OX1]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("alcohol", "[OX2H]"),
    ("phenol", "[OX2H][c]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("aromatic_heterocycle", "[n,o,s;r]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("nitrile", "[CX2]#N"),
)


def mol_from_sdf(path: str) -> Mol | None:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next(iter(supplier), None) if supplier is not None else None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def mol_from_smiles(smiles: str, *, seed: int = 1) -> Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        from rdkit.Chem import AllChem

        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return mol
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return mol
    return mol


def qualify_molecule(mol: Mol | None) -> dict[str, Any]:
    if mol is None or mol.GetNumAtoms() < 1:
        return {
            "qualification_passed": False,
            "status": "unjudgeable_parse_failure",
            "reason": "RDKit could not parse a connected molecule",
        }
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        return {
            "qualification_passed": False,
            "status": "unjudgeable_sanitize_failure",
            "reason": str(error)[:300],
        }
    fragments = Chem.GetMolFrags(mol, asMols=False)
    if len(fragments) != 1:
        return {
            "qualification_passed": False,
            "status": "unjudgeable_disconnected",
            "reason": f"molecule has {len(fragments)} fragments",
        }
    if mol.GetNumConformers() < 1:
        return {
            "qualification_passed": False,
            "status": "unjudgeable_missing_conformer",
            "reason": "no 3D conformer",
        }
    return {
        "qualification_passed": True,
        "status": "judgeable",
        "atom_count": int(mol.GetNumAtoms()),
        "reason": None,
    }


def compute_evidence(mol: Mol) -> dict[str, Any]:
    qualification = qualify_molecule(mol)
    if qualification["qualification_passed"] is not True:
        return {
            "qualification": qualification,
            **_failed_sections(qualification["reason"]),
            "docking_used": False,
            "gnina_used": False,
        }
    return {
        "qualification": qualification,
        "identity": _identity(mol),
        "descriptors": _descriptors(mol),
        "functional_groups": _functional_groups(mol),
        "geometry": _geometry(mol),
        "docking_used": False,
        "gnina_used": False,
    }


def _identity(mol: Mol) -> dict[str, Any]:
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return {
        "status": "observed",
        "isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "atom_mapped_smiles": Chem.MolToSmiles(mapped, isomericSmiles=True),
        "empirical_formula": rdMolDescriptors.CalcMolFormula(mol),
        "elements": sorted({atom.GetSymbol() for atom in mol.GetAtoms()}),
        "atom_count": int(mol.GetNumAtoms()),
        "reason": None,
    }


def _descriptors(mol: Mol) -> dict[str, Any]:
    return {
        "status": "observed",
        "molecular_weight_da": float(Descriptors.MolWt(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "hydrogen_bond_donor_count": int(Lipinski.NumHDonors(mol)),
        "hydrogen_bond_acceptor_count": int(Lipinski.NumHAcceptors(mol)),
        "topological_polar_surface_area_angstrom2": float(Descriptors.TPSA(mol)),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "crippen_logp": float(Crippen.MolLogP(mol)),
    }


def _functional_groups(mol: Mol) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for name, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            groups.append(
                {"name": name, "atom_indices": [int(index) for index in match]}
            )
    return {"status": "observed", "groups": groups}


def _geometry(mol: Mol) -> dict[str, Any]:
    if mol.GetNumConformers() < 1:
        return {
            "status": "failed",
            "intramolecular_clash_count": None,
            "abnormal_bond_count": None,
            "aromatic_planarity_max_deviation_angstrom": None,
            "reason": "no 3D conformer",
        }
    conf = mol.GetConformer()
    bonded = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
    }
    one_three: set[tuple[int, int]] = set()
    for atom in mol.GetAtoms():
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for left in neighbors:
            for right in neighbors:
                if left < right:
                    one_three.add((left, right))
    clashes = 0
    table = Chem.GetPeriodicTable()
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            pair = (i, j)
            if pair in bonded or pair in one_three:
                continue
            distance = float(
                (conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length()
            )
            radii = table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) + table.GetRvdw(
                mol.GetAtomWithIdx(j).GetAtomicNum()
            )
            if distance < 0.7 * radii:
                clashes += 1
    abnormal = 0
    for begin, end in bonded:
        expected = table.GetRcovalent(
            mol.GetAtomWithIdx(begin).GetAtomicNum()
        ) + table.GetRcovalent(mol.GetAtomWithIdx(end).GetAtomicNum())
        length = float(
            (conf.GetAtomPosition(begin) - conf.GetAtomPosition(end)).Length()
        )
        if abs(length - expected) > 0.4:
            abnormal += 1
    return {
        "status": "observed",
        "intramolecular_clash_count": clashes,
        "abnormal_bond_count": abnormal,
        "aromatic_planarity_max_deviation_angstrom": _aromatic_planarity(mol, conf),
        "reason": None,
    }


def _aromatic_planarity(mol: Mol, conf: Any) -> float | None:
    deviations: list[float] = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) < 5 or not all(
            mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring
        ):
            continue
        points = [conf.GetAtomPosition(index) for index in ring]
        centroid = points[0]
        for point in points[1:]:
            centroid += point
        centroid /= len(points)
        axis_a = points[1] - points[0]
        axis_b = points[2] - points[0]
        normal = axis_a.CrossProduct(axis_b)
        if float(normal.Length()) < 1e-8:
            continue
        normal /= float(normal.Length())
        deviations.append(
            max(abs(float((point - centroid).DotProduct(normal))) for point in points)
        )
    if not deviations:
        return None
    return round(max(deviations), 5)


def _failed_sections(reason: str | None) -> dict[str, Any]:
    return {
        "identity": {"status": "failed", "reason": reason},
        "descriptors": {"status": "failed"},
        "functional_groups": {"status": "failed", "groups": []},
        "geometry": {"status": "failed", "reason": reason},
    }


__all__ = [
    "FUNCTIONAL_GROUP_SMARTS",
    "compute_evidence",
    "mol_from_sdf",
    "mol_from_smiles",
    "qualify_molecule",
]
