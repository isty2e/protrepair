"""Retained source chemistry when a real component has an incomplete scaffold."""

from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)


@pytest.mark.parametrize("public_workflow", [False, True])
def test_3enj_retains_source_cysteine_hydrogens(public_workflow: bool) -> None:
    source = read_structure(
        Path(__file__).resolve().parents[1] / "fixtures" / "corpus" / "pdb3enj.ent"
    )
    residue_id = ResidueId("A", 500)
    source_index = source.constitution.residue_index(residue_id)
    original = source.constitution.residue_site_at(source_index)
    assert original.atom_site_names() == (
        "N",
        "CA",
        "CB",
        "SG",
        "H1",
        "H2",
        "H3",
        "HB2",
        "HB3",
    )
    if public_workflow:
        result = process_structure(
            source,
            ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
            requested_goals=(
                requested_process_goal(
                    scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
                ),
            ),
        )
    else:
        result = add_retained_non_polymer_hydrogens(
            source, target_residue_ids=frozenset({residue_id})
        )

    repaired = result.structure
    repaired_index = repaired.constitution.residue_index(residue_id)
    assert repaired.constitution.residue_site_at(repaired_index) == original
    assert repaired.residue_geometry(repaired_index) == source.residue_geometry(
        source_index
    )
    assert repaired.residue_formal_charge_by_atom_name(
        repaired_index
    ) == source.residue_formal_charge_by_atom_name(source_index)
    assert not any(repair.residue_id == residue_id for repair in result.repairs)
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        and issue.residue_id == residue_id
        and issue.atom_names == ("H1", "H2", "H3", "HB2", "HB3")
        for issue in result.issues
    )

    observation = StructureChemistryObservation(
        repaired, component_library=build_default_component_library()
    )
    _, chemistry = observation.whole_structure_facts()
    fact = next(
        fact
        for fact in chemistry.retained_non_polymer_facts
        if fact.residue_id == residue_id
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.ABSENT
    )

    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        reread = read_structure_string(
            write_structure_string(repaired, file_format=file_format),
            file_format=file_format,
        )
        reread_residue = reread.constitution.residue_or_ligand(residue_id)
        assert reread_residue is not None
        assert reread_residue.atom_site_names() == original.atom_site_names()
