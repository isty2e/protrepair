"""Public workflow coverage for calculation-only peptide support."""

from pathlib import Path

import pytest
from tests.support.request_builders import transform_requests

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.continuous.binding_policy import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationConfig
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec

pytestmark = pytest.mark.workflow


@pytest.mark.representative_regression
def test_heavy_only_source_refines_without_preparing_omitted_cap_partners() -> None:
    source = read_structure_string(
        "".join(
            line
            for line in Path("tests/fixtures/pdb/1aho.pdb")
            .read_text()
            .splitlines(keepends=True)
            if not (
                line.startswith(("ATOM", "HETATM"))
                and line[76:78].strip() in {"H", "D", "T"}
            )
        ),
        file_format=FileFormat.PDB,
    )
    selected = ResidueId("A", 10)
    result = process_structure(
        source,
        transform_requests=transform_requests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((selected,)),
                binding=RecommendedContinuousRelaxationBinding(),
                config=ContinuousRelaxationConfig(context_radius_angstrom=0),
            )
        ),
    )
    assert any(
        event.kind is RepairEventKind.LOCAL_REFINEMENT_APPLIED
        for event in result.repairs
    )
    assert any(
        issue.kind is ValidationIssueKind.COMPUTATIONAL_PEPTIDE_CAP
        for issue in result.issues
    )
    for number in (8, 12):
        residue = result.structure.constitution.residue_or_ligand(
            ResidueId("A", number)
        )
        assert residue is not None
        assert not any(atom.is_hydrogen() for atom in residue.atom_sites)
    for residue in source.constitution.iter_residues():
        if residue.residue_id == selected:
            continue
        for atom in residue.atom_sites:
            ref = AtomRef(residue.residue_id, atom.name)
            before = source.constitution.atom_index(ref)
            after = result.structure.constitution.atom_index(ref)
            assert result.structure.geometry.position(
                after
            ) == source.geometry.position(before)
    assert result.structure.provenance.ingress is source.provenance.ingress
    assert result.structure.provenance.lineage == source.provenance.lineage
    assert not any(
        choice.residue_id in {ResidueId("A", 8), ResidueId("A", 12)}
        for choice in result.structure.provenance.microstate_choices
    )
    assert result.structure.chain_ids() == source.chain_ids()
    serialized = read_structure_string(
        write_structure_string(result.structure, file_format=FileFormat.MMCIF),
        file_format=FileFormat.MMCIF,
    )
    assert (
        serialized.constitution.atom_slots == result.structure.constitution.atom_slots
    )
