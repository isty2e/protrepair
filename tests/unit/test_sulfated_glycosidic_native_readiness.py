"""Bound sulfated glycans admit native fields in the original protein context."""

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
from protrepair.io import read_structure, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.artifacts import RegionTransformationResult, StructureDelta
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.peptide_boundary import PeptideBoundaryPlan
from protrepair.transformer.continuous.peptide_caps import PeptideCapTransformer
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.readiness import (
    atom_scope_facts_continuous_relaxation_error,
    derive_atom_scope_continuous_relaxation_facts,
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
def sulfated_local_context() -> tuple[ProteinStructure, tuple[AtomRef, ...]]:
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb4oee.ent")
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert not result.issues
    atoms = tuple(
        result.structure.constitution.atom_ref_at(index)
        for index, _ in result.structure.geometry.iter_positions()
        if result.structure.constitution.atom_ref_at(index).residue_id
        in {ResidueId("A", seq) for seq in range(130, 136)}
    )
    directive = RepairLocalRefinementDirective.from_atom_refs(
        atoms,
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    staged, _ = repair_stage._apply_local_refinement_prerequisites(
        TransformationResult(result.structure, (), ()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert not staged.issues
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        new = staged.structure.constitution.atom_index(ref)
        assert staged.structure.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(old)
        assert staged.structure.geometry.atom_geometry(
            new
        ) == source.geometry.atom_geometry(old)
        if source.topology.formal_charge(old) is not None:
            assert staged.structure.topology.formal_charge(
                new
            ) == source.topology.formal_charge(old)
    return staged.structure, atoms


@pytest.mark.parametrize(
    "family", (ContinuousRelaxationForceField.UFF, ContinuousRelaxationForceField.MMFF)
)
def test_sulfated_glycans_bind_closed_shell_fields_without_context_loss(
    sulfated_local_context: tuple[ProteinStructure, tuple[AtomRef, ...]],
    family: ContinuousRelaxationForceField,
) -> None:
    structure, atoms = sulfated_local_context
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(atoms),
        component_library=build_default_component_library(),
        force_field=family,
    )
    facts = derive_atom_scope_continuous_relaxation_facts(
        problem.region.snapshot,
        problem.region.atom_input.observed_atom_scope(problem.region.snapshot),
        component_library=build_default_component_library(),
        context_radius_angstrom=6.0,
    )
    assert atom_scope_facts_continuous_relaxation_error(facts) is None
    plan = PeptideBoundaryPlan.from_region(
        problem.region, problem.bonds, build_default_component_library()
    )
    capped = PeptideCapTransformer().transform(problem, plan)
    assert (
        capped.problem.region.movable_atom_indices
        == problem.region.movable_atom_indices
    )
    assert set(problem.region.included_atom_indices()) <= set(
        capped.problem.region.included_atom_indices()
    )
    molecule, mapping = build_rdkit_molecule(capped.problem)
    Chem.SanitizeMol(molecule)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    glycans = set()
    for residue in structure.constitution.ligands:
        if residue.component_id not in {"GNX", "IDY"}:
            continue
        indices = structure.constitution.atom_indices_for_residue(residue.residue_id)
        assert all(index in mapping for index in indices)
        assert all(
            molecule.GetAtomWithIdx(mapping[index]).GetFormalCharge() == 0
            for index in indices
        )
        glycans.add(residue.residue_id)
    assert glycans == {ResidueId("B", 1), ResidueId("B", 2)}
    field = build_rdkit_force_field(
        molecule,
        problem=capped.problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(field.CalcEnergy()) and all(map(isfinite, field.CalcGrad()))
    calculation = capped.problem.region.snapshot.structure
    projected = capped.project_result(
        RegionTransformationResult(
            calculation,
            StructureDelta(
                before_constitution=calculation.constitution,
                after_constitution=calculation.constitution,
            ),
            (),
            "test",
        )
    ).refined_structure
    assert projected == structure


def test_whole_prepared_sulfated_protein_is_readable_without_conect(
    sulfated_local_context: tuple[ProteinStructure, tuple[AtomRef, ...]],
) -> None:
    structure, _ = sulfated_local_context
    pdb = (
        "\n".join(
            line
            for line in write_structure_string(structure, FileFormat.PDB).splitlines()
            if not line.startswith("CONECT")
        )
        + "\n"
    )
    molecule = Chem.MolFromPDBBlock(
        pdb, sanitize=True, removeHs=False, proximityBonding=True
    )
    assert molecule is not None
    assert molecule.GetNumAtoms() == len(structure.constitution.atom_slots)
