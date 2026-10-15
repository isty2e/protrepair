"""Explicit retained chemistry distinguishes assigned neutrality from unknown charge."""

from dataclasses import replace

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import (
    ingress_options,
    whole_structure_requested_goals,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.contracts import LigandPolicy

LIGAND_ID = ResidueId("L", 1)


def _methanol(
    *, carbon_charge: int | None = None, oxygen_charge: int | None = None
) -> ProteinStructure:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0), formal_charge=carbon_charge),
                    atom_payload(
                        "O1", "O", Vec3(1.43, 0, 0), formal_charge=oxygen_charge
                    ),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return read_structure_string(
        write_structure_string(structure, file_format=FileFormat.PDB),
        file_format=FileFormat.PDB,
    )


def _with_charge(
    structure: ProteinStructure, name: str, charge: int | None
) -> ProteinStructure:
    index = structure.constitution.atom_index(AtomRef(LIGAND_ID, name))
    atoms = list(structure.topology.atom_topologies)
    atoms[index.value] = AtomTopology(formal_charge=charge)
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(atoms),
            bonds=structure.topology.bonds,
        ),
        provenance=structure.provenance,
    )


@pytest.mark.parametrize(
    ("smiles", "oxygen_charge", "hydrogen_count"),
    (("CO", 0, 4), ("C[O-]", -1, 3), ("C[OH2+]", 1, 5)),
)
@pytest.mark.parametrize("source_carbon_charge", (None, 0))
def test_explicit_smiles_publishes_all_selected_heavy_charges(
    smiles: str,
    oxygen_charge: int,
    hydrogen_count: int,
    source_carbon_charge: int | None,
) -> None:
    source = _methanol(carbon_charge=source_carbon_charge)
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, smiles, ("C1", "O1")
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )

    assert not result.issues
    observation = result.structure.provenance.ingress.observation
    assert observation is not None
    for name, charge, reported in (
        ("C1", 0, source_carbon_charge),
        ("O1", oxygen_charge, None),
    ):
        ref = AtomRef(LIGAND_ID, name)
        old = source.constitution.atom_index(ref)
        new = result.structure.constitution.atom_index(ref)
        assert result.structure.topology.formal_charge(new) == charge
        assert observation.formal_charge(ref) == reported
        assert result.structure.constitution.atom_site_at(new) == (
            source.constitution.atom_site_at(old)
        )
        assert result.structure.geometry.atom_geometry(new) == (
            source.geometry.atom_geometry(old)
        )
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == hydrogen_count
    resolution = resolve_retained_non_polymer_chemistry(
        result.structure,
        residue,
        component_library=ComponentLibrary({}),
        evidence=evidence,
    )
    assert resolution.has_formal_charge_topology(result.structure, residue)

    problem = build_continuous_relaxation_problem(
        result.structure,
        LocalScopeSpec.from_residues((LIGAND_ID,)),
        component_library=ComponentLibrary(
            {"UNK": ResidueTemplate(ChemicalComponentDefinition("UNK", ("C1", "O1")))}
        ),
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    assert not any(atom.GetNumImplicitHs() for atom in molecule.GetAtoms())
    for name, charge in (("C1", 0), ("O1", oxygen_charge)):
        index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, name))
        assert molecule.GetAtomWithIdx(mapping[index]).GetFormalCharge() == charge


@pytest.mark.parametrize(
    ("smiles", "source_oxygen_charge"),
    (("CO", -1), ("CO", 1), ("C[O-]", 0), ("C[OH2+]", 0)),
)
def test_conflicting_known_charge_refuses_explicit_chemistry_atomically(
    smiles: str, source_oxygen_charge: int
) -> None:
    source = _methanol(oxygen_charge=source_oxygen_charge)
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, smiles, ("C1", "O1")
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )

    assert result.structure is source
    assert not result.repairs
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )


def test_unknown_charge_does_not_realize_explicit_neutral_selection() -> None:
    source = _methanol()
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "CO", ("C1", "O1")
    ).to_evidence()
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=ComponentLibrary({}),
        evidence=evidence,
    )

    assert not resolution.has_formal_charge_topology(
        source, source.constitution.ligands[0]
    )


def test_deleted_neutral_charge_invalidates_explicit_chemistry_readiness() -> None:
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "CO", ("C1", "O1")
    ).to_evidence()
    prepared = add_retained_non_polymer_hydrogens(
        _methanol(), ComponentLibrary({}), chemistry_evidence=(evidence,)
    ).structure
    prepared = _with_charge(_with_charge(prepared, "C1", 0), "O1", 0)
    stale = _with_charge(prepared, "O1", None)
    for structure, needs_resolution in ((prepared, False), (stale, True)):
        fact = (
            StructureChemistryObservation(
                structure,
                component_library=ComponentLibrary({}),
                retained_non_polymer_chemistry_evidence=(evidence,),
            )
            .whole_structure_facts()[1]
            .retained_non_polymer_facts[0]
        )
        assert fact.requires_topology_resolution() is needs_resolution


def _complete_source_methanol(symbol: str) -> ProteinStructure:
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "CO", ("C1", "O1")
    ).to_evidence()
    prepared = add_retained_non_polymer_hydrogens(
        _methanol(), ComponentLibrary({}), chemistry_evidence=(evidence,)
    ).structure
    atoms = []
    for atom in prepared.constitution.ligands[0].atom_sites:
        index = prepared.constitution.atom_index(AtomRef(LIGAND_ID, atom.name))
        geometry = prepared.geometry.atom_geometry(index)
        atoms.append(
            atom_payload(
                atom.name,
                symbol if atom.is_hydrogen() else atom.element,
                geometry.position,
                occupancy=0.75,
                b_factor=8.5,
            )
        )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=tuple(atoms),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=prepared.topology.bonds,
        ),
        provenance=source.provenance,
    )
    return read_structure_string(
        write_structure_string(source, file_format=FileFormat.PDB),
        file_format=FileFormat.PDB,
    )


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
@pytest.mark.parametrize("topology_only", (False, True))
def test_complete_source_hydrogens_receive_only_missing_selected_charges(
    symbol: str, topology_only: bool
) -> None:
    source = _complete_source_methanol(symbol)
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "CO", ("C1", "O1")
    ).to_evidence()
    transform = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transform(source, ComponentLibrary({}), chemistry_evidence=(evidence,))

    assert not result.issues
    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry
    observation = result.structure.provenance.ingress.observation
    assert observation is source.provenance.ingress.observation
    assert observation is not None
    for atom in source.constitution.ligands[0].atom_sites:
        ref = AtomRef(LIGAND_ID, atom.name)
        index = source.constitution.atom_index(ref)
        assert result.structure.topology.formal_charge(index) == (
            None if atom.is_hydrogen() else 0
        )
        assert observation.formal_charge(ref) is None
    for before in source.topology.bonds:
        after = result.structure.topology.bond_between(*before.endpoint_pair())
        assert after is not None
        assert after.provenance is before.provenance
        assert after.source_metadata == before.source_metadata
    repeated = transform(
        result.structure, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert repeated.structure is result.structure
    assert not repeated.issues and not repeated.repairs


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
def test_incompatible_source_hydrogen_blocks_charge_and_atom_publication(
    symbol: str,
) -> None:
    source = _complete_source_methanol(symbol)
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "C[O-]", ("C1", "O1")
    ).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary({}), chemistry_evidence=(evidence,)
    )
    assert result.structure is source
    assert result.issues and not result.repairs


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("smiles", ("CO", "C[O-]", "C[OH2+]"))
def test_output_roundtrip_preserves_selected_heavy_charges(
    file_format: FileFormat, smiles: str
) -> None:
    override = RetainedNonPolymerChemistryOverride(LIGAND_ID, smiles, ("C1", "O1"))
    prepared = add_retained_non_polymer_hydrogens(
        _methanol(), ComponentLibrary({}), chemistry_evidence=(override.to_evidence(),)
    ).structure
    restored = read_structure_string(
        write_structure_string(prepared, file_format=file_format),
        file_format=file_format,
    )
    for name in ("C1", "O1"):
        ref = AtomRef(LIGAND_ID, name)
        before = prepared.constitution.atom_index(ref)
        after = restored.constitution.atom_index(ref)
        assert restored.topology.formal_charge(
            after
        ) == prepared.topology.formal_charge(before)
    resolved = resolve_retained_non_polymer_topology(
        restored, ComponentLibrary({}), chemistry_evidence=(override.to_evidence(),)
    )
    assert not resolved.issues
    resolution = resolve_retained_non_polymer_chemistry(
        resolved.structure,
        resolved.structure.constitution.ligands[0],
        component_library=ComponentLibrary({}),
        evidence=override.to_evidence(),
    )
    assert resolution.has_formal_charge_topology(
        resolved.structure, resolved.structure.constitution.ligands[0]
    )


@pytest.mark.parametrize("reverse_smiles", (False, True))
def test_public_workflow_realizes_selected_neutrality_independent_of_smiles_order(
    reverse_smiles: bool,
) -> None:
    override = RetainedNonPolymerChemistryOverride(
        LIGAND_ID,
        "OC" if reverse_smiles else "CO",
        ("O1", "C1") if reverse_smiles else ("C1", "O1"),
    )
    result = process_structure(
        _methanol(),
        ingress=ingress_options(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=(override,),
        ),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.has_errors()
    assert result.requested_goals_fully_satisfied()
    for name in ("C1", "O1"):
        index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, name))
        assert result.structure.topology.formal_charge(index) == 0


def test_neutral_template_accepts_unknown_but_not_conflicting_charge() -> None:
    source = _methanol()
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID, "CO", ("C1", "O1")
    ).to_evidence()
    resolution = replace(
        resolve_retained_non_polymer_chemistry(
            source,
            source.constitution.ligands[0],
            component_library=ComponentLibrary({}),
            evidence=evidence,
        ),
        source=RetainedNonPolymerChemistryEvidenceSource.TEMPLATE,
    )
    assert resolution.has_formal_charge_topology(source, source.constitution.ligands[0])
    conflicting = _with_charge(source, "O1", -1)
    assert not resolution.has_formal_charge_topology(
        conflicting, conflicting.constitution.ligands[0]
    )


def test_fallback_does_not_assign_neutral_charge_without_explicit_chemistry() -> None:
    result = add_retained_non_polymer_hydrogens(_methanol(), ComponentLibrary({}))
    assert result.issues
    assert all(
        issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
        for issue in result.issues
    )
    for name in ("C1", "O1"):
        index = result.structure.constitution.atom_index(AtomRef(LIGAND_ID, name))
        assert result.structure.topology.formal_charge(index) is None


def test_charge_conflict_does_not_prevent_neighbor_preparation() -> None:
    neighbor_id = ResidueId("L", 2)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 0, 0)),
                    atom_payload("O1", "O", Vec3(1.43, 0, 0), formal_charge=1),
                ),
                is_hetero=True,
            ),
            residue_payload(
                component_id="UNK",
                residue_id=neighbor_id,
                atoms=(
                    atom_payload("C1", "C", Vec3(0, 10, 0)),
                    atom_payload("O1", "O", Vec3(1.43, 10, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        ComponentLibrary({}),
        chemistry_evidence=tuple(
            RetainedNonPolymerChemistryOverride(
                residue_id, "CO", ("C1", "O1")
            ).to_evidence()
            for residue_id in (LIGAND_ID, neighbor_id)
        ),
    )
    assert result.issues
    assert all(issue.residue_id == LIGAND_ID for issue in result.issues)
    assert result.structure.constitution.ligands[0] == source.constitution.ligands[0]
    for name in ("C1", "O1"):
        ref = AtomRef(LIGAND_ID, name)
        old = source.constitution.atom_index(ref)
        new = result.structure.constitution.atom_index(ref)
        assert result.structure.topology.formal_charge(new) == (
            source.topology.formal_charge(old)
        )
        assert result.structure.geometry.atom_geometry(new) == (
            source.geometry.atom_geometry(old)
        )
        neighbor = result.structure.constitution.atom_index(AtomRef(neighbor_id, name))
        assert result.structure.topology.formal_charge(neighbor) == 0
    assert len(result.structure.constitution.ligands[1].atom_sites) == 6
