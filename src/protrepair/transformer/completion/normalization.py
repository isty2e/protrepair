"""Residue normalization helpers for completion target derivation."""

from protrepair.chemistry import ComponentLibrary
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

ILE_ATOM_ALIASES: dict[str, str] = {"CD": "CD1"}


def normalize_residue_inputs(
    residue: CompletionResiduePayload,
    component_library: ComponentLibrary,
) -> tuple[CompletionResiduePayload, str | None]:
    """Normalize supported residue aliases before completion planning."""

    normalized_component_id = component_library.normalize_component_id(
        residue.component_id
    )
    normalized_residue = residue
    normalization_details: str | None = None
    if normalized_component_id != residue.component_id:
        normalized_residue = residue.with_component_id(normalized_component_id)
        normalization_details = (
            f"normalized component {residue.component_id} -> {normalized_component_id}"
        )

    if (
        normalized_component_id == "ILE"
        and normalized_residue.has_atom("CD")
        and not normalized_residue.has_atom("CD1")
    ):
        normalized_residue = normalized_residue.renamed_atoms(ILE_ATOM_ALIASES)

    return normalized_residue, normalization_details


def normalized_reference_residue_by_id(
    reference_structure: ProteinStructure | None,
    *,
    chain_id: str,
    component_library: ComponentLibrary,
) -> dict[ResidueId, CompletionResiduePayload]:
    """Return normalized reference residues for one chain keyed by residue id."""

    if reference_structure is None or not reference_structure.has_chain(chain_id):
        return {}

    normalized_residue_by_id: dict[ResidueId, CompletionResiduePayload] = {}
    for residue_site in reference_structure.constitution.chain(chain_id).residues:
        residue_index = reference_structure.constitution.residue_index(
            residue_site.residue_id
        )
        normalized_residue, _ = normalize_residue_inputs(
            CompletionResiduePayload(
                residue_site=residue_site,
                residue_geometry=reference_structure.geometry.residue_geometry(
                    constitution=reference_structure.constitution,
                    residue_index=residue_index,
                ),
                formal_charge_by_atom_name=(
                    reference_structure.topology.residue_formal_charge_by_atom_name(
                        constitution=reference_structure.constitution,
                        residue_index=residue_index,
                    )
                ),
            ),
            component_library,
        )
        normalized_residue_by_id[normalized_residue.residue_id] = normalized_residue

    return normalized_residue_by_id
