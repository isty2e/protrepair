"""Actual covalent QS7 preparation admits native fields in unchanged context."""

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
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import BondRelationshipType
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.binding import ManualContinuousRelaxationBinding
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
def michael_local_context() -> tuple[ProteinStructure, tuple[AtomRef, ...]]:
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb6v9c.ent")
    prepared = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    ).structure
    # This is the fixed original eight-residue cohort window, not a reduced
    # ligand-only substitute. Context radius remains the production 6 A.
    atoms = tuple(
        prepared.constitution.atom_ref_at(i)
        for i, _ in prepared.geometry.iter_positions()
        if prepared.constitution.atom_ref_at(i).residue_id
        in {ResidueId("A", n) for n in range(482, 490)}
    )
    directive = RepairLocalRefinementDirective.from_atom_refs(
        atoms,
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    staged, _ = repair_stage._apply_local_refinement_prerequisites(
        TransformationResult(prepared, (), ()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert not staged.issues
    for old, _ in source.geometry.iter_positions():
        new = staged.structure.constitution.atom_index(
            source.constitution.atom_ref_at(old)
        )
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
def test_actual_qs7_cysteine_adduct_binds_closed_shell_fields(
    michael_local_context, family
):
    structure, atoms = michael_local_context
    _assert_native_fields(structure, atoms, family)


def _assert_native_fields(structure, atoms, family):
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
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    for index, native_index in mapping.items():
        element = capped.problem.region.snapshot.structure.constitution.atom_site_at(
            index
        ).element
        if element in {"D", "T"}:
            assert molecule.GetAtomWithIdx(native_index).GetIsotope() == (
                2 if element == "D" else 3
            )
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    for residue_id in (ResidueId("A", 552), ResidueId("A", 801)):
        assert set(structure.constitution.atom_indices_for_residue(residue_id)) <= set(
            mapping
        )
    carbon = structure.constitution.atom_index(AtomRef(ResidueId("A", 801), "CBG"))
    sulfur = structure.constitution.atom_index(AtomRef(ResidueId("A", 552), "SG"))
    bond = molecule.GetBondBetweenAtoms(mapping[carbon], mapping[sulfur])
    assert bond is not None and bond.GetBondType() == Chem.BondType.SINGLE
    assert molecule.GetAtomWithIdx(mapping[sulfur]).GetFormalCharge() == 0
    Chem.AssignStereochemistryFrom3D(molecule)
    for name, expected in (("CAY", "S"), ("CBC", "R")):
        index = structure.constitution.atom_index(AtomRef(ResidueId("A", 801), name))
        assert molecule.GetAtomWithIdx(mapping[index]).GetProp("_CIPCode") == expected
    for name in ("CBF", "CBG"):
        index = structure.constitution.atom_index(AtomRef(ResidueId("A", 801), name))
        assert not molecule.GetAtomWithIdx(mapping[index]).HasProp("_CIPCode")
    field = build_rdkit_force_field(
        molecule,
        problem=capped.problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )
    assert isfinite(field.CalcEnergy()) and all(map(isfinite, field.CalcGrad()))


@pytest.mark.parametrize(
    "file_format,omit_conect,reverse_atoms,isotope",
    (
        (FileFormat.PDB, False, False, "H"),
        (FileFormat.PDB, True, False, "H"),
        (FileFormat.PDB, False, True, "H"),
        (FileFormat.MMCIF, False, False, "H"),
        (FileFormat.PDB, False, True, "D"),
        (FileFormat.PDB, True, False, "T"),
    ),
)
def test_local_roundtrip_preparation_resolves_product_before_polymer_hydrogens(
    michael_local_context, file_format, omit_conect, reverse_atoms, isotope
):
    structure, atoms = michael_local_context
    text = write_structure_string(structure, file_format)
    if file_format is FileFormat.PDB:
        lines = text.splitlines()
        if isotope != "H":
            matching = [
                offset
                for offset, line in enumerate(lines)
                if line.startswith("ATOM  ")
                and line[12:16].strip() == "H"
                and line[21:26] == "A 482"
            ]
            assert len(matching) == 1
            offset = matching[0]
            lines[offset] = lines[offset][:76] + f"{isotope:>2}" + lines[offset][78:]
        if omit_conect:
            lines = [line for line in lines if not line.startswith("CONECT")]
        if reverse_atoms:
            start = 0
            while start < len(lines):
                if not lines[start].startswith(("ATOM  ", "HETATM")):
                    start += 1
                    continue
                end = start + 1
                while (
                    end < len(lines)
                    and lines[end].startswith(("ATOM  ", "HETATM"))
                    and lines[end][17:27] == lines[start][17:27]
                ):
                    end += 1
                lines[start:end] = reversed(lines[start:end])
                start = end
        text = "\n".join(lines) + "\n"
    recovered = read_structure_string(text, file_format)
    directive = RepairLocalRefinementDirective.from_atom_refs(
        atoms,
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    staged, observation = repair_stage._apply_local_refinement_prerequisites(
        TransformationResult(recovered, (), ()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert not staged.issues
    assert observation.context_structure is staged.structure
    assert staged.structure.constitution == recovered.constitution
    assert staged.structure.geometry == recovered.geometry
    assert (
        staged.structure.topology.atom_topologies == recovered.topology.atom_topologies
    )
    assert staged.structure.provenance.ingress is recovered.provenance.ingress
    index = staged.structure.constitution.atom_index(AtomRef(ResidueId("A", 482), "H"))
    assert staged.structure.constitution.atom_site_at(index).element == isotope
    for old in recovered.topology.bonds:
        new = staged.structure.topology.bond_between(*old.endpoint_pair())
        assert new is not None
        assert new.provenance is old.provenance
        assert new.source_metadata == old.source_metadata
    for family in (
        ContinuousRelaxationForceField.UFF,
        ContinuousRelaxationForceField.MMFF,
    ):
        _assert_native_fields(staged.structure, atoms, family)
    repeated, repeated_observation = repair_stage._apply_local_refinement_prerequisites(
        staged,
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert repeated is staged
    assert repeated_observation.context_structure is staged.structure


@pytest.mark.parametrize("connection", ("multiple-peptide", "unknown-sidechain"))
def test_local_preparation_preserves_incompatible_source_connections(
    michael_local_context, connection
):
    structure, atoms = michael_local_context
    lines = write_structure_string(structure, FileFormat.PDB).splitlines()
    serials = {
        AtomRef(ResidueId(line[21], int(line[22:26])), line[12:16].strip()): int(
            line[6:11]
        )
        for line in lines
        if line.startswith(("ATOM  ", "HETATM"))
    }
    first, second = (
        (AtomRef(ResidueId("A", 481), "C"), AtomRef(ResidueId("A", 482), "N"))
        if connection == "multiple-peptide"
        else (AtomRef(ResidueId("A", 482), "N"), AtomRef(ResidueId("A", 484), "CB"))
    )
    # The peptide already occurs once in the serialized connectivity. A second
    # occurrence supplies an explicit double order; the non-peptide pair is new.
    lines.insert(-1, f"CONECT{serials[first]:5d}{serials[second]:5d}")
    recovered = read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)
    pair = tuple(recovered.constitution.atom_index(ref) for ref in (first, second))
    original = recovered.topology.bond_between(*pair)
    assert original is not None
    if connection == "multiple-peptide":
        assert original.order == 2
        assert original.source_metadata is not None
        assert original.source_metadata.reported_order == 2
    else:
        assert original.relationship_type is BondRelationshipType.UNKNOWN
        assert original.order is None
    directive = RepairLocalRefinementDirective.from_atom_refs(
        atoms,
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.MMFF),
    )
    staged, _ = repair_stage._apply_local_refinement_prerequisites(
        TransformationResult(recovered, (), ()),
        local_refinement=directive,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=True,
        retained_non_polymer_chemistry_evidence=(),
    )
    assert staged.structure.topology.bond_between(*pair) == original
    assert staged.structure.constitution == recovered.constitution
    assert staged.structure.geometry == recovered.geometry
    assert (
        staged.structure.topology.atom_topologies == recovered.topology.atom_topologies
    )
    assert staged.structure.provenance.ingress is recovered.provenance.ingress
    problem = build_continuous_relaxation_problem(
        staged.structure,
        LocalScopeSpec.from_atoms(atoms),
        component_library=build_default_component_library(),
    )
    facts = derive_atom_scope_continuous_relaxation_facts(
        problem.region.snapshot,
        problem.region.atom_input.observed_atom_scope(problem.region.snapshot),
        component_library=build_default_component_library(),
        context_radius_angstrom=6.0,
    )
    assert atom_scope_facts_continuous_relaxation_error(facts) is not None


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_whole_qs7_product_roundtrip_keeps_contextual_chemistry(
    michael_local_context, file_format
):
    structure, atoms = michael_local_context
    recovered = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    library = build_default_component_library()
    resolved = resolve_retained_non_polymer_topology(
        recovered, library, target_residue_ids=frozenset({ResidueId("A", 801)})
    )
    assert not resolved.issues
    recovered = resolved.structure
    ligand = recovered.constitution.residue_or_ligand(ResidueId("A", 801))
    assert ligand is not None
    assert CysteineMichaelAdductBinding(
        recovered, ligand, library.require("QS7"), library
    ).cysteine_is_realized()
    chemistry = StructureChemistryObservation(recovered).whole_structure_facts()[1]
    ligand_fact = next(
        f
        for f in chemistry.retained_non_polymer_facts
        if f.residue_id == ligand.residue_id
    )
    assert (
        ligand_fact.heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert (
        ligand_fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    # Connectivity-only PDB H bonds need preparation before native admission.
    if file_format is FileFormat.PDB:
        directive = RepairLocalRefinementDirective.from_atom_refs(
            atoms,
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationForceField.MMFF
            ),
        )
        staged, _ = repair_stage._apply_local_refinement_prerequisites(
            TransformationResult(recovered, (), ()),
            local_refinement=directive,
            component_library=library,
            allow_retained_non_polymer_rdkit_fallback=True,
            retained_non_polymer_chemistry_evidence=(),
        )
        assert not staged.issues
        recovered = staged.structure
    problem = build_continuous_relaxation_problem(
        recovered,
        LocalScopeSpec.from_atoms(atoms),
        component_library=build_default_component_library(),
    )
    facts = derive_atom_scope_continuous_relaxation_facts(
        problem.region.snapshot,
        problem.region.atom_input.observed_atom_scope(problem.region.snapshot),
        component_library=build_default_component_library(),
        context_radius_angstrom=6.0,
    )
    assert atom_scope_facts_continuous_relaxation_error(facts) is None


def test_whole_prepared_qs7_protein_is_readable_without_conect(michael_local_context):
    structure, _ = michael_local_context
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
    assert molecule is not None and molecule.GetNumAtoms() == len(
        structure.constitution.atom_slots
    )
