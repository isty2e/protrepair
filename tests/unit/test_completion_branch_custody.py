"""Polymer completion preserves ligand edits already accepted on the branch."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions import heavy_completion, hydrogen_completion
from protrepair.workflow.actions.context import TransformerExecutionContext
from protrepair.workflow.contracts import OrphanFragmentPolicy


@pytest.mark.parametrize("hydrogens", (False, True))
def test_polymer_completion_does_not_restore_original_ligand(hydrogens, monkeypatch):
    polymer_id = ResidueId("A", 1)
    ligand_id = ResidueId("L", 1)
    current = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=polymer_id,
                        atoms=(atom_payload("CA", "C", Vec3(0, 0, 0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ligand_id,
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(5, 0, 0)),
                    atom_payload("H1", "H", Vec3(6, 0, 0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    original = current.without_atom_refs((AtomRef(ligand_id, "H1"),))
    module = hydrogen_completion if hydrogens else heavy_completion
    name = "add_hydrogens" if hydrogens else "repair_heavy_atoms_core"
    monkeypatch.setattr(
        module,
        name,
        lambda structure, **kwargs: TransformationResult(structure, (), ()),
    )
    action = (
        hydrogen_completion.HydrogenCompletionTransformer
        if hydrogens
        else heavy_completion.HeavyAtomCompletionTransformer
    )(scope=ResidueSetScope((polymer_id,)))
    result = action.execute(
        TransformationResult(current, (), ()),
        context=TransformerExecutionContext(
            component_library=build_default_component_library(),
            original_structure=original,
            orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
        ),
    )
    assert result.structure.constitution == current.constitution
    assert result.structure.geometry == current.geometry
    assert result.structure.topology == current.topology
