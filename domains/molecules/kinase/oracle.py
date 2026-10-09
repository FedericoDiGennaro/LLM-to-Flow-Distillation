"""Binary graph verifier for the Goal 5 kinase campaign objective.

A qualifying molecule is neutral and connected, with the amino-aza-aromatic
motif and an atom-disjoint saturated ring containing a free secondary amine.
Carbon linkage, conformer geometry, and saturated ring size are unrestricted.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable

def mol_from_smiles(smiles):
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)

def canonical_smiles(mol):
    from rdkit import Chem
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def score_smiles(smiles, model=None):
    mol = mol_from_smiles(smiles)
    return None if mol is None else score_mol(mol, model)

from domains.molecules.kinase.cards import FINAL_GOAL
MODEL_ID = 'kinase_link_relaxed_binary_prompt_exact_v1'
REWARD_MODEL_ID = MODEL_ID
REWARD_ENDPOINT = 'kinase_link_relaxed_final_goal_binary_prompt_exact'
HELD_OUT_EVALUATOR_IDS = ('kinase_final_goal_structural_check',)
PREFERENCE_RULE = 'reward_one_is_chosen_over_reward_zero'
EXTREME_LABEL_RULE = 'reward_one_positive_reward_zero_negative'
LABEL_RULE = EXTREME_LABEL_RULE

def load_oracle(path=None):
    return None

def score_mol(mol, model=None):
    return float(link_relaxed_prompt_exact_binary_reward(mol))

def induce_extreme_labels(item_ids, predictions, *, positive_count=2, negative_count=2):
    return {str(item): ('positive' if float(predictions[str(item)]['raw_reward']) == 1 else 'negative')
            if str(item) in predictions else 'abstain' for item in item_ids}


def _total_hydrogens(atom: Any) -> int:
    return int(atom.GetTotalNumHs(includeNeighbors=True))

def _heavy_neighbors(atom: Any) -> list[Any]:
    return [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1]

def _pyridine_like_nitrogen(atom: Any) -> bool:
    return bool(
        atom.GetAtomicNum() == 7
        and atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and _total_hydrogens(atom) == 0
    )

def _free_primary_amino_nitrogen(atom: Any) -> bool:
    return bool(
        atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and _total_hydrogens(atom) == 2
        and len(_heavy_neighbors(atom)) == 1
    )

def _amide_like_neighbor(atom: Any, ring: frozenset[int]) -> bool:
    """Whether a cyclic N is acyl/sulfonyl-bound rather than a free amine."""

    from rdkit.Chem.rdchem import BondType

    for neighbor in _heavy_neighbors(atom):
        if neighbor.GetIdx() not in ring:
            return True
        if neighbor.GetAtomicNum() not in (6,):
            return True
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if other.GetIdx() == atom.GetIdx():
                continue
            if bond.GetBondType() == BondType.DOUBLE and other.GetAtomicNum() in (
                7,
                8,
                16,
            ):
                return True
    return False

def _free_secondary_ring_amine(atom: Any, ring: frozenset[int]) -> bool:
    return bool(
        atom.GetIdx() in ring
        and atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and _total_hydrogens(atom) == 1
        and len(_heavy_neighbors(atom)) == 2
        and all(neighbor.GetIdx() in ring for neighbor in _heavy_neighbors(atom))
        and not _amide_like_neighbor(atom, ring)
    )

def _ordered_ring_bonds(mol: Any, ring: tuple[int, ...]) -> Iterable[Any]:
    for index, begin in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        bond = mol.GetBondBetweenAtoms(int(begin), int(end))
        if bond is None:
            return
        yield bond

def molecule_from_sdf(path: Path) -> Any | None:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    return next(iter(supplier), None) if supplier is not None else None

def _prompt_exact_six_membered_hinge_rings(
    mol: Any,
) -> list[frozenset[int]]:
    """Return six-membered aza-aromatic rings with the specified amino motif."""

    from rdkit.Chem.rdchem import BondType

    hinges: list[frozenset[int]] = []
    for ordered_ring in mol.GetRingInfo().AtomRings():
        if len(ordered_ring) != 6:
            continue
        ring = frozenset(int(index) for index in ordered_ring)
        if not all(
            mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring
        ):
            continue
        pyridine_nitrogens = {
            index
            for index in ring
            if _pyridine_like_nitrogen(mol.GetAtomWithIdx(index))
        }
        if not pyridine_nitrogens:
            continue
        matched = False
        for carbon_index in ring:
            carbon = mol.GetAtomWithIdx(carbon_index)
            if carbon.GetAtomicNum() != 6:
                continue
            has_free_primary_amino = any(
                neighbor.GetIdx() not in ring
                and _free_primary_amino_nitrogen(neighbor)
                and mol.GetBondBetweenAtoms(
                    carbon_index, neighbor.GetIdx()
                ).GetBondType()
                == BondType.SINGLE
                for neighbor in carbon.GetNeighbors()
            )
            if not has_free_primary_amino:
                continue
            if any(
                neighbor.GetIdx() in pyridine_nitrogens
                for neighbor in carbon.GetNeighbors()
                if neighbor.GetIdx() in ring
            ):
                matched = True
                break
        if matched:
            hinges.append(ring)
    return hinges

def _prompt_exact_saturated_secondary_amine_rings(
    mol: Any,
) -> list[frozenset[int]]:
    """Return saturated rings of any perceived size containing a free 2° amine."""

    from rdkit.Chem.rdchem import BondType

    tails: list[frozenset[int]] = []
    for ordered_ring in mol.GetRingInfo().AtomRings():
        ring = frozenset(int(index) for index in ordered_ring)
        bonds = list(_ordered_ring_bonds(mol, tuple(ordered_ring)))
        if not ring or len(bonds) != len(ordered_ring):
            continue
        if any(
            bond.GetIsAromatic() or bond.GetBondType() != BondType.SINGLE
            for bond in bonds
        ):
            continue
        if any(
            _free_secondary_ring_amine(mol.GetAtomWithIdx(index), ring)
            for index in ring
        ):
            tails.append(ring)
    return tails

def link_relaxed_prompt_exact_binary_reward(mol: Any | None) -> int:
    """Return one exactly when the registered half-relaxed prompt is met."""

    if mol is None:
        return 0
    try:
        from rdkit import Chem

        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() < 1 or len(Chem.GetMolFrags(mol)) != 1:
            return 0
        if Chem.GetFormalCharge(mol) != 0:
            return 0
        hinges = _prompt_exact_six_membered_hinge_rings(mol)
        tails = _prompt_exact_saturated_secondary_amine_rings(mol)
        return int(
            any(not (hinge & tail) for hinge in hinges for tail in tails)
        )
    except Exception:
        return 0

def reward_sdf(path: Path) -> int:
    try:
        molecule = molecule_from_sdf(path)
    except Exception:
        molecule = None
    return link_relaxed_prompt_exact_binary_reward(molecule)
