"""Native glycan graph admission on an unchanged, full local FF context."""

from math import isfinite
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.api import process_structure
from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.io import read_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement import repair_stage
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)


@pytest.fixture(scope="module")
def glycosylated_local_context() -> tuple[ProteinStructure, tuple[AtomRef, ...]]:
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb2z62.ent")
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(),
                value=HydrogenCoverageState.COMPLETE,
            ),
        ),
    )
    assert result.requested_goals_fully_satisfied()
    structure = result.structure
    atoms = tuple(
        structure.constitution.atom_ref_at(index)
        for index, _ in structure.geometry.iter_positions()
        if structure.constitution.atom_ref_at(index).residue_id
        in {ResidueId("A", seq) for seq in range(191, 197)}
    )
    directive = RepairLocalRefinementDirective.from_atom_refs(
        atoms,
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    staged, _ = repair_stage._apply_local_refinement_prerequisites(
        TransformationResult(structure, (), ()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert not staged.issues
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        current = staged.structure.constitution.atom_index(ref)
        assert staged.structure.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(old)
        assert staged.structure.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(old)
        if source.topology.formal_charge(old) is not None:
            assert staged.structure.topology.formal_charge(
                current
            ) == source.topology.formal_charge(old)
    return staged.structure, atoms


@pytest.mark.parametrize(
    "family", (ContinuousRelaxationForceField.UFF, ContinuousRelaxationForceField.MMFF)
)
def test_glycans_bind_native_force_fields_without_optimization_or_context_loss(
    glycosylated_local_context: tuple[ProteinStructure, tuple[AtomRef, ...]],
    family: ContinuousRelaxationForceField,
) -> None:
    structure, atoms = glycosylated_local_context
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(atoms),
        component_library=build_default_component_library(),
        force_field=family,
    )
    assert set(problem.region.movable_atom_indices) == {
        structure.constitution.atom_index(ref) for ref in atoms
    }
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    glycans = set()
    for residue in structure.constitution.ligands:
        if residue.component_id != "NAG":
            continue
        indices = structure.constitution.atom_indices_for_residue(residue.residue_id)
        if not any(index in mapping for index in indices):
            continue
        assert all(index in mapping for index in indices)
        assert all(
            molecule.GetAtomWithIdx(mapping[index]).GetFormalCharge() == 0
            and molecule.GetAtomWithIdx(mapping[index]).GetNumRadicalElectrons() == 0
            for index in indices
        )
        glycans.add(residue.residue_id)
    assert glycans == {ResidueId("D", 1), ResidueId("D", 2)}
    assert {
        index
        for residue_id in glycans
        for index in structure.constitution.atom_indices_for_residue(residue_id)
    } <= set(problem.region.fixed_context_atom_indices)
    field = build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(field.CalcEnergy()) and all(map(isfinite, field.CalcGrad()))
