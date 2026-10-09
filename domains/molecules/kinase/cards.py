"""Strict LLM-visible cards for inexpensive generated-molecule evidence."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


CARD_CONTRACT = "molecule-candidate-card/1.0.0"
FINAL_GOAL = 'Generate neutral, connected small molecules for a kinase-focused discovery campaign that contain a six-membered aza-aromatic ring bearing a free primary amino group on a ring carbon adjacent to a pyridine-like ring nitrogen, and a distinct fully saturated ring containing a free secondary amine. The saturated amine-containing ring need not be carbon-linked to the aza-aromatic ring.'
MOLECULE_CARD_VISIBLE_FIELDS = frozenset(
    {
        "identity.status",
        "identity.isomeric_smiles",
        "identity.atom_mapped_smiles",
        "identity.empirical_formula",
        "identity.elements",
        "identity.atom_count",
        "identity.reason",
        "descriptors.status",
        "descriptors.molecular_weight_da",
        "descriptors.formal_charge",
        "descriptors.hydrogen_bond_donor_count",
        "descriptors.hydrogen_bond_acceptor_count",
        "descriptors.topological_polar_surface_area_angstrom2",
        "descriptors.rotatable_bond_count",
        "descriptors.aromatic_ring_count",
        "descriptors.ring_count",
        "descriptors.crippen_logp",
        "functional_groups.status",
        "functional_groups.groups",
        "geometry.status",
        "geometry.intramolecular_clash_count",
        "geometry.abnormal_bond_count",
        "geometry.aromatic_planarity_max_deviation_angstrom",
        "geometry.reason",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class IdentityCard(StrictModel):
    status: Literal["observed", "failed"]
    isomeric_smiles: str | None = Field(default=None, max_length=4000)
    atom_mapped_smiles: str | None = Field(default=None, max_length=8000)
    empirical_formula: str | None = Field(default=None, max_length=300)
    elements: list[str] = Field(default_factory=list, max_length=40)
    atom_count: int | None = Field(default=None, gt=0, le=500)
    reason: str | None = Field(default=None, max_length=1000)


class DescriptorCard(StrictModel):
    status: Literal["observed", "failed"]
    molecular_weight_da: float | None = Field(default=None, gt=0)
    formal_charge: int | None = None
    hydrogen_bond_donor_count: int | None = Field(default=None, ge=0)
    hydrogen_bond_acceptor_count: int | None = Field(default=None, ge=0)
    topological_polar_surface_area_angstrom2: float | None = Field(default=None, ge=0)
    rotatable_bond_count: int | None = Field(default=None, ge=0)
    aromatic_ring_count: int | None = Field(default=None, ge=0)
    ring_count: int | None = Field(default=None, ge=0)
    crippen_logp: float | None = None


class FunctionalGroupEntry(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    atom_indices: list[int] = Field(min_length=1, max_length=80)


class FunctionalGroupCard(StrictModel):
    status: Literal["observed", "failed"]
    groups: list[FunctionalGroupEntry] = Field(default_factory=list, max_length=80)


class GeometryCard(StrictModel):
    status: Literal["observed", "failed"]
    intramolecular_clash_count: int | None = Field(default=None, ge=0)
    abnormal_bond_count: int | None = Field(default=None, ge=0)
    aromatic_planarity_max_deviation_angstrom: float | None = Field(
        default=None, ge=0
    )
    reason: str | None = Field(default=None, max_length=1000)


class MoleculeCandidateCard(StrictModel):
    representation_contract: Literal["molecule-candidate-card/1.0.0"]
    item_id: str = Field(pattern=r"^item-\d{3}$")
    identity: IdentityCard
    descriptors: DescriptorCard
    functional_groups: FunctionalGroupCard
    geometry: GeometryCard
    evidence_limitations: str = Field(min_length=1, max_length=2000)


def evidence_tool_definition() -> dict[str, Any]:
    return {
        "card_contract": CARD_CONTRACT,
        "evidence_generation": (
            "The host precomputes RDKit identity, descriptors, graph-local "
            "functional groups, and cheap 3D geometry diagnostics before the "
            "LLM request. The LLM has no callable tools."
        ),
        "identity_limit": (
            "Isomeric SMILES and atom maps describe connectivity and stereo; they "
            "are not a pose, a binding mode, or a database identity."
        ),
        "descriptor_limit": (
            "Weight, polarity, H-bonding and logP are structural arithmetic, not "
            "kinase affinity, selectivity, permeability assays, or process evidence."
        ),
        "functional_group_limit": (
            "Named groups with atom indices locate chemistry on the graph. They "
            "are not a pharmacophore score, docking score, or target fingerprint."
        ),
        "geometry_limit": (
            "Clash, bond-length and planarity checks are local 3D diagnostics, "
            "not proof of a receptor pose or of experimental binding."
        ),
        "missing_value_policy": (
            "Null or failed evidence is unavailable, not automatically adverse."
        ),
        "hidden_evidence": (
            "No GNINA, CNNAffinity, pose RMSD, labelled kinase ligands, raw "
            "coordinate tables, generator provenance, internet, or experiments."
        ),
    }


def candidate_card(record: Mapping[str, Any], *, item_id: str) -> MoleculeCandidateCard:
    identity = record["identity"]
    descriptors = record["descriptors"]
    groups = record["functional_groups"]
    geometry = record["geometry"]
    return MoleculeCandidateCard(
        representation_contract=CARD_CONTRACT,
        item_id=item_id,
        identity=IdentityCard(
            status=identity["status"],
            isomeric_smiles=identity.get("isomeric_smiles"),
            atom_mapped_smiles=identity.get("atom_mapped_smiles"),
            empirical_formula=identity.get("empirical_formula"),
            elements=list(identity.get("elements") or []),
            atom_count=identity.get("atom_count"),
            reason=identity.get("reason"),
        ),
        descriptors=DescriptorCard(
            status=descriptors["status"],
            molecular_weight_da=_optional_float(descriptors.get("molecular_weight_da")),
            formal_charge=descriptors.get("formal_charge"),
            hydrogen_bond_donor_count=descriptors.get("hydrogen_bond_donor_count"),
            hydrogen_bond_acceptor_count=descriptors.get(
                "hydrogen_bond_acceptor_count"
            ),
            topological_polar_surface_area_angstrom2=_optional_float(
                descriptors.get("topological_polar_surface_area_angstrom2")
            ),
            rotatable_bond_count=descriptors.get("rotatable_bond_count"),
            aromatic_ring_count=descriptors.get("aromatic_ring_count"),
            ring_count=descriptors.get("ring_count"),
            crippen_logp=_optional_float(descriptors.get("crippen_logp")),
        ),
        functional_groups=FunctionalGroupCard(
            status=groups["status"],
            groups=[
                FunctionalGroupEntry(
                    name=str(entry["name"]),
                    atom_indices=[int(index) for index in entry["atom_indices"]],
                )
                for entry in groups.get("groups", [])
            ],
        ),
        geometry=GeometryCard(
            status=geometry["status"],
            intramolecular_clash_count=geometry.get("intramolecular_clash_count"),
            abnormal_bond_count=geometry.get("abnormal_bond_count"),
            aromatic_planarity_max_deviation_angstrom=_optional_float(
                geometry.get("aromatic_planarity_max_deviation_angstrom")
            ),
            reason=geometry.get("reason"),
        ),
        evidence_limitations=(
            "Host-computed RDKit identity, descriptors, graph-local functional "
            "groups and cheap 3D geometry only. These fields cannot establish "
            "kinase binding, pose, affinity or experimental activity; missing "
            "evidence remains unknown."
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 5)


__all__ = [
    "CARD_CONTRACT",
    "FINAL_GOAL",
    "MOLECULE_CARD_VISIBLE_FIELDS",
    "MoleculeCandidateCard",
    "candidate_card",
    "evidence_tool_definition",
]
