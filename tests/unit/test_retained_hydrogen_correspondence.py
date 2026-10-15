"""Retained H correspondence must respect current chemistry and source identity."""

from dataclasses import replace
from math import isfinite
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import (
    ingress_options,
    whole_structure_requested_goals,
)

from protrepair.api import process_structure
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.inference import retained_non_polymer_fallback as fallback
from protrepair.chemistry.restraint.defaults import build_default_restraint_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import AtomSetScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryResolution,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.retained_non_polymer_hydrogen import repair
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.rdkit import (
    build_rdkit_force_field,
    build_rdkit_molecule,
)
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.contracts import LigandPolicy

LIGAND_ID = ResidueId("L", 1)


def _alkane(carbon_count: int) -> ProteinStructure:
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=tuple(
                    atom_payload(
                        f"C{index:02d}",
                        "C",
                        Vec3(1.26 * index, 0.88 * (index % 2), 0.0),
                    )
                    for index in range(carbon_count)
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _prepared_alkane(carbon_count: int = 38) -> ProteinStructure:
    result = add_retained_non_polymer_hydrogens(
        _alkane(carbon_count), ComponentLibrary({})
    )
    assert len(result.repairs) == 1
    return result.structure


def _with_bonds(
    structure: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=structure.provenance,
    )


def _hydrogen_anchors(structure: ProteinStructure) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for bond in structure.topology.bonds:
        first = structure.constitution.atom_site_at(bond.atom_index_1)
        second = structure.constitution.atom_site_at(bond.atom_index_2)
        if first.is_hydrogen() != second.is_hydrogen():
            hydrogen, heavy = (
                (first, second) if first.is_hydrogen() else (second, first)
            )
            anchors[hydrogen.name] = heavy.name
    return anchors


def _resolution(structure: ProteinStructure) -> RetainedNonPolymerChemistryResolution:
    return resolve_retained_non_polymer_chemistry(
        structure,
        structure.constitution.ligands[0],
        component_library=ComponentLibrary({}),
    )


@pytest.mark.parametrize("carbon_count", (1, 2, 31, 32, 38))
def test_generated_hydrogens_remain_observable_on_both_sides_of_pair_bound(
    carbon_count: int,
) -> None:
    prepared = _prepared_alkane(carbon_count)
    before = (
        prepared.constitution,
        prepared.geometry,
        prepared.topology,
        prepared.provenance,
    )

    resolution = _resolution(prepared)

    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    assert len(resolution.expected_hydrogen_atom_names) == 2 * carbon_count + 2
    assert resolution.has_hydrogen_topology(prepared, prepared.constitution.ligands[0])
    assert resolution.hydrogen_name_projection_candidate_count <= (
        resolution.hydrogen_name_projection_candidate_limit
    )
    assert (
        prepared.constitution,
        prepared.geometry,
        prepared.topology,
        prepared.provenance,
    ) == before


@pytest.mark.parametrize("carbon_count", (2, 38))
def test_complete_hydrogen_regeneration_preserves_exact_current_atoms(
    carbon_count: int,
) -> None:
    prepared = _prepared_alkane(carbon_count)

    regenerated = add_retained_non_polymer_hydrogens(prepared, ComponentLibrary({}))
    topology_only = resolve_retained_non_polymer_topology(
        prepared, ComponentLibrary({})
    )

    for result in (regenerated, topology_only):
        assert result.structure.constitution == prepared.constitution
        assert result.structure.geometry == prepared.geometry
        assert result.structure.topology == prepared.topology
        assert result.structure.provenance.ingress is prepared.provenance.ingress
        assert not result.repairs
        assert not any("resource limit" in issue.message for issue in result.issues)


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_large_reconciliation_preserves_source_isotope_and_scalar_payload(
    isotope: str,
) -> None:
    prepared = _prepared_alkane()
    residue = prepared.constitution.ligands[0]
    index = prepared.constitution.residue_index(LIGAND_ID)
    geometry = prepared.residue_geometry(index)
    atom = residue.atom_site("H001")
    source = prepared.with_updated_residue_facets_batch(
        (
            (
                replace(
                    residue,
                    atom_sites=tuple(
                        replace(site, element=isotope) if site == atom else site
                        for site in residue.atom_sites
                    ),
                ),
                geometry.with_atom_geometry(
                    "H001",
                    replace(
                        geometry.atom_geometry("H001"), occupancy=0.75, b_factor=12.5
                    ),
                ),
                (("H001", 0),),
            ),
        )
    )

    result = add_retained_non_polymer_hydrogens(source, ComponentLibrary({}))

    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert _hydrogen_anchors(result.structure) == _hydrogen_anchors(source)
    assert _resolution(result.structure).source.is_resolved()


@pytest.mark.parametrize("carbon_count", (2, 38))
def test_current_attachment_survives_hydrogen_coordinate_exchange(carbon_count: int):
    prepared = _prepared_alkane(carbon_count)
    residue = prepared.constitution.ligands[0]
    anchors = _hydrogen_anchors(prepared)
    first = next(iter(anchors))
    second = next(name for name in anchors if anchors[name] != anchors[first])
    index = prepared.constitution.residue_index(LIGAND_ID)
    geometry = prepared.residue_geometry(index)
    moved = prepared.with_updated_residue_facets_batch(
        (
            (
                residue,
                geometry.with_atom_geometry(
                    first, geometry.atom_geometry(second)
                ).with_atom_geometry(second, geometry.atom_geometry(first)),
                prepared.residue_formal_charge_by_atom_name(index),
            ),
        )
    )

    resolution = _resolution(moved)
    observed = {
        (bond.atom_name_2 if bond.atom_name_1 in anchors else bond.atom_name_1)
        for bond in resolution.hydrogen_bond_definitions
    }

    assert resolution.source.is_resolved()
    assert observed == set(anchors.values())
    assert resolution.has_hydrogen_topology(moved, residue)
    inferred = {
        bond.atom_name_1: bond.atom_name_2
        for bond in resolution.hydrogen_bond_definitions
        if bond.atom_name_1 in anchors
    } | {
        bond.atom_name_2: bond.atom_name_1
        for bond in resolution.hydrogen_bond_definitions
        if bond.atom_name_2 in anchors
    }
    assert inferred == anchors


def test_graphless_large_hydrogen_input_retains_resource_refusal_atomically():
    prepared = _prepared_alkane()
    graphless = _with_bonds(prepared, ())

    resolution = _resolution(graphless)
    result = add_retained_non_polymer_hydrogens(graphless, ComponentLibrary({}))

    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "resource limit" in resolution.failure_reason
    assert result.structure is graphless
    assert not result.repairs
    assert any("resource limit" in issue.message for issue in result.issues)


def test_reobservation_separates_chemistry_from_attachment_multiplicity_conflict():
    prepared = _prepared_alkane(2)
    anchors = _hydrogen_anchors(prepared)
    hydrogen = next(name for name, anchor in anchors.items() if anchor == "C00")
    hydrogen_index = prepared.constitution.atom_index(AtomRef(LIGAND_ID, hydrogen))
    other_index = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "C01"))
    changed = _with_bonds(
        prepared,
        tuple(
            bond
            for bond in prepared.topology.bonds
            if not bond.involves(hydrogen_index)
        )
        + (
            TopologyBond(
                hydrogen_index, other_index, provenance=BondProvenance.REPAIR_INFERRED
            ),
        ),
    )

    resolution = _resolution(changed)

    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
    assert not resolution.has_hydrogen_topology(
        changed, changed.constitution.ligands[0]
    )
    assert prepared.topology != changed.topology


@pytest.mark.corpus
def test_actual_1kgb_long_ligands_remain_chemically_observable_after_public_h():
    source = read_structure(Path("tests/fixtures/corpus/pdb1kgb.ent"))
    result = process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(
            HydrogenCoverageState.COMPLETE, include_default_heavy_completion=False
        ),
    )
    library = build_default_component_library()
    for number, expected_h in ((602, 78), (610, 76), (607, 72)):
        residue_id = ResidueId("A", number)
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None
        resolution = resolve_retained_non_polymer_chemistry(
            result.structure, residue, component_library=library
        )
        assert len(resolution.expected_hydrogen_atom_names) == expected_h
        assert (
            resolution.source
            is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
        )
        assert resolution.has_hydrogen_topology(result.structure, residue)
        assert resolution.hydrogen_name_projection_candidate_count <= 4096
    assert result.structure.provenance.ingress is source.provenance.ingress


@pytest.mark.parametrize("carbon_count", (1, 2, 3))
@pytest.mark.parametrize("ties", (False, True))
def test_unanchored_projection_matches_exhaustive_distance_name_oracle(
    carbon_count: int, ties: bool
) -> None:
    source = _alkane(carbon_count)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND_ID))
    generated = fallback.infer_retained_non_polymer_rdkit_fallback(residue, geometry)
    molecule = generated.hydrogenated_molecule
    conformer = molecule.GetConformer()
    positions = dict(
        zip(
            generated.projected_hydrogen_atom_names(),
            (
                Vec3(*conformer.GetAtomPosition(atom.GetIdx()))
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() == 1
            ),
            strict=True,
        )
    )
    target_names = tuple(f"Q{index:03d}" for index in range(len(positions)))
    target_positions = {
        name: Vec3(0, 0, 0) if ties else position
        for name, position in zip(
            target_names, reversed(tuple(positions.values())), strict=True
        )
    }
    pairs = sorted(
        (position.distance_to(target_positions[target]), name, target)
        for name, position in positions.items()
        for target in target_names
    )
    expected: dict[str, str] = {}
    for _, name, target in pairs:
        if name not in expected and target not in expected.values():
            expected[name] = target

    for names in (target_names, tuple(reversed(target_names))):
        result = fallback.infer_retained_non_polymer_rdkit_fallback(
            residue,
            geometry,
            hydrogen_atom_names=names,
            hydrogen_position_by_name=target_positions,
        )
        assert dict(result.hydrogen_name_projection) == expected
        assert result.hydrogen_name_projection_candidate_count == len(names) ** 2


def test_partial_attachments_reserve_known_parent_before_unanchored_names() -> None:
    source = _prepared_alkane(2)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND_ID))
    anchors = _hydrogen_anchors(source)
    known = next(name for name, parent in anchors.items() if parent == "C00")
    position_by_name = {name: geometry.position(name) for name in anchors}
    unanchored = tuple(name for name in anchors if name != known)
    carbon_zero_positions = tuple(
        geometry.position(name) for name, parent in anchors.items() if parent == "C00"
    )
    for name, position in zip(unanchored[:3], carbon_zero_positions, strict=True):
        position_by_name[name] = position
    position_by_name[known] = Vec3(2000, 0, 0)

    result = fallback.infer_retained_non_polymer_rdkit_fallback(
        residue,
        geometry,
        hydrogen_position_by_name=position_by_name,
        hydrogen_anchor_by_name={known: "C00"},
    )

    assert len(result.hydrogen_name_projection) == len(anchors)
    assert result.hydrogen_name_projection_candidate_count == 3 + 5 * 6
    assert any(
        {bond.atom_name_1, bond.atom_name_2} == {known, "C00"}
        for bond in result.hydrogen_bond_definitions
    )


@pytest.mark.parametrize("limit, resolved", ((161, False), (162, True)))
def test_attached_projection_counts_only_admissible_pairs_at_exact_bound(
    monkeypatch: pytest.MonkeyPatch, limit: int, resolved: bool
) -> None:
    source = _prepared_alkane()
    monkeypatch.setattr(fallback, "_HYDROGEN_NAME_PROJECTION_CANDIDATE_LIMIT", limit)

    resolution = _resolution(source)

    assert resolution.source.is_resolved() is resolved
    if resolved:
        assert resolution.hydrogen_name_projection_candidate_count == 162
        assert resolution.has_hydrogen_topology(source, source.constitution.ligands[0])
    else:
        assert "resource limit" in resolution.failure_reason


@pytest.mark.parametrize("limit, accepted", ((161, False), (162, True)))
def test_attached_source_reconciliation_is_atomic_at_exact_bound(
    monkeypatch: pytest.MonkeyPatch, limit: int, accepted: bool
) -> None:
    source = _prepared_alkane()
    monkeypatch.setattr(
        repair, "_SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT", limit
    )

    result = add_retained_non_polymer_hydrogens(source, ComponentLibrary({}))

    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert not result.repairs
    assert (
        any("resource limit" in issue.message for issue in result.issues)
        is not accepted
    )


@pytest.mark.parametrize(
    "relationship, provenance, order, aromatic, reusable",
    (
        (BondRelationshipType.COVALENT, BondProvenance.REPAIR_INFERRED, 1, False, True),
        (
            BondRelationshipType.UNKNOWN,
            BondProvenance.SOURCE_EXPLICIT,
            None,
            False,
            True,
        ),
        (
            BondRelationshipType.COVALENT,
            BondProvenance.SOURCE_EXPLICIT,
            None,
            False,
            True,
        ),
        (
            BondRelationshipType.UNKNOWN,
            BondProvenance.REPAIR_INFERRED,
            None,
            False,
            False,
        ),
        (
            BondRelationshipType.COVALENT,
            BondProvenance.REPAIR_INFERRED,
            2,
            False,
            False,
        ),
        (BondRelationshipType.COVALENT, BondProvenance.REPAIR_INFERRED, 1, True, False),
        (
            BondRelationshipType.DISULFIDE,
            BondProvenance.SOURCE_EXPLICIT,
            1,
            False,
            False,
        ),
        (
            BondRelationshipType.HYDROGEN_BOND,
            BondProvenance.SOURCE_EXPLICIT,
            1,
            False,
            False,
        ),
        (
            BondRelationshipType.METAL_COORDINATION,
            BondProvenance.SOURCE_EXPLICIT,
            1,
            False,
            False,
        ),
    ),
)
def test_topology_parent_query_distinguishes_connectivity_from_chemistry(
    relationship: BondRelationshipType,
    provenance: BondProvenance,
    order: int | None,
    aromatic: bool,
    reusable: bool,
) -> None:
    prepared = _prepared_alkane(2)
    hydrogen_index = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "H001"))
    carbon_index = prepared.constitution.atom_index(AtomRef(LIGAND_ID, "C00"))
    source = _with_bonds(
        prepared,
        (
            TopologyBond(
                hydrogen_index,
                carbon_index,
                order=order,
                aromatic=aromatic,
                relationship_type=relationship,
                provenance=provenance,
            ),
        ),
    )

    anchors = source.topology.residue_hydrogen_anchor_by_name(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(LIGAND_ID),
    )

    assert anchors == ({"H001": "C00"} if reusable else {})
    assert _resolution(source).source.is_resolved()


@pytest.mark.parametrize("other_name", ("C01", "H002"))
def test_ambiguous_or_hydrogen_parent_is_not_reused(other_name: str) -> None:
    prepared = _prepared_alkane(2)
    anchors = _hydrogen_anchors(prepared)
    hydrogen = next(name for name, parent in anchors.items() if parent == "C00")
    changed = _with_bonds(
        prepared,
        prepared.topology.bonds
        + (
            TopologyBond(
                prepared.constitution.atom_index(AtomRef(LIGAND_ID, hydrogen)),
                prepared.constitution.atom_index(AtomRef(LIGAND_ID, other_name)),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )

    projected = changed.topology.residue_hydrogen_anchor_by_name(
        constitution=changed.constitution,
        residue_index=changed.constitution.residue_index(LIGAND_ID),
    )

    assert hydrogen not in projected
    assert _resolution(changed).source.is_resolved()
    assert not _resolution(changed).has_hydrogen_topology(
        changed, changed.constitution.ligands[0]
    )


def test_charge_change_rederives_chemistry_instead_of_reusing_h_inventory() -> None:
    source = _prepared_alkane(1)
    residue = source.constitution.ligands[0]
    index = source.constitution.residue_index(LIGAND_ID)
    charged = source.with_updated_residue_facets_batch(
        ((residue, source.residue_geometry(index), (("C00", 1),)),)
    )

    before = _resolution(source)
    after = _resolution(charged)

    assert len(before.expected_hydrogen_atom_names) == 4
    assert after.source.is_resolved()
    assert len(after.expected_hydrogen_atom_names) == 3
    assert not after.has_hydrogen_topology(charged, residue)
    assert _resolution(source) == before


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_reingested_large_source_isotope_payload_and_metadata_are_preserved(
    isotope: str,
) -> None:
    prepared = _prepared_alkane()
    residue = prepared.constitution.ligands[0]
    index = prepared.constitution.residue_index(LIGAND_ID)
    prepared = prepared.with_updated_residue_facets_batch(
        (
            (
                replace(
                    residue,
                    atom_sites=tuple(
                        replace(atom, element=isotope) if atom.name == "H001" else atom
                        for atom in residue.atom_sites
                    ),
                ),
                prepared.residue_geometry(index).with_atom_geometry(
                    "H001",
                    replace(
                        prepared.residue_geometry(index).atom_geometry("H001"),
                        occupancy=0.75,
                        b_factor=12.5,
                    ),
                ),
                prepared.residue_formal_charge_by_atom_name(index),
            ),
        )
    )
    source = read_structure_string(
        write_structure_string(prepared, FileFormat.MMCIF), FileFormat.MMCIF
    )
    source_bond = next(
        bond
        for bond in source.topology.bonds
        if bond.involves(source.constitution.atom_index(AtomRef(LIGAND_ID, "H001")))
    )
    assert source_bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert source_bond.source_metadata is not None

    result = add_retained_non_polymer_hydrogens(source, ComponentLibrary({}))

    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert result.structure.provenance.ingress is source.provenance.ingress
    assert _resolution(result.structure).source.is_resolved()
    assert not any("resource limit" in issue.message for issue in result.issues)


def test_strict_policy_and_explicit_override_do_not_enter_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _prepared_alkane()
    residue = source.constitution.ligands[0]
    evidence = RetainedNonPolymerChemistryOverride(
        LIGAND_ID,
        "C" * 38,
        tuple(atom.name for atom in residue.atom_sites if not atom.is_hydrogen()),
    ).to_evidence()

    def forbidden(*args, **kwargs):
        raise AssertionError("strict policy or explicit evidence entered fallback")

    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry."
        "infer_retained_non_polymer_rdkit_fallback",
        forbidden,
    )
    monkeypatch.setattr(repair, "infer_retained_non_polymer_rdkit_fallback", forbidden)

    strict = resolve_retained_non_polymer_chemistry(
        source,
        residue,
        component_library=ComponentLibrary({}),
        allow_rdkit_fallback=False,
    )
    overridden = resolve_retained_non_polymer_chemistry(
        source,
        residue,
        component_library=ComponentLibrary({}),
        evidence=evidence,
        allow_rdkit_fallback=False,
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        ComponentLibrary({}),
        chemistry_evidence=(evidence,),
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert strict.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert (
        overridden.source is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    assert overridden.has_hydrogen_topology(source, residue)
    assert result.structure.geometry == source.geometry
    assert not any("resource limit" in issue.message for issue in result.issues)


@pytest.mark.parametrize(
    "relationship, reusable",
    (
        (BondRelationshipType.COVALENT, False),
        (BondRelationshipType.HYDROGEN_BOND, True),
        (BondRelationshipType.METAL_COORDINATION, True),
    ),
)
def test_external_contact_is_not_confused_with_local_covalent_parent(
    relationship: BondRelationshipType, reusable: bool
) -> None:
    external_id = ResidueId("E", 1)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=(
                    atom_payload("C", "C", Vec3(0, 0, 0)),
                    atom_payload("H", "H", Vec3(1, 0, 0)),
                ),
                is_hetero=True,
            ),
            residue_payload(
                component_id="HOH",
                residue_id=external_id,
                atoms=(atom_payload("O", "O", Vec3(2, 0, 0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    hydrogen = source.constitution.atom_index(AtomRef(LIGAND_ID, "H"))
    source = _with_bonds(
        source,
        (
            TopologyBond(
                hydrogen, source.constitution.atom_index(AtomRef(LIGAND_ID, "C"))
            ),
            TopologyBond(
                hydrogen,
                source.constitution.atom_index(AtomRef(external_id, "O")),
                relationship_type=relationship,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )

    anchors = source.topology.residue_hydrogen_anchor_by_name(
        constitution=source.constitution,
        residue_index=source.constitution.residue_index(LIGAND_ID),
    )

    assert anchors == ({"H": "C"} if reusable else {})


def test_heavy_geometry_change_rederives_carbonyl_chemistry() -> None:
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=(
                    atom_payload("C", "C", Vec3(0, 0, 0)),
                    atom_payload("O", "O", Vec3(1.4, 0, 0)),
                    atom_payload("C2", "C", Vec3(-1.4, 0, 0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = add_retained_non_polymer_hydrogens(source, ComponentLibrary({})).structure
    residue = source.constitution.ligands[0]
    index = source.constitution.residue_index(LIGAND_ID)
    geometry = source.residue_geometry(index)
    changed = source.with_updated_residue_facets_batch(
        (
            (
                residue,
                geometry.with_atom_geometry(
                    "O", replace(geometry.atom_geometry("O"), position=Vec3(1.22, 0, 0))
                ),
                source.residue_formal_charge_by_atom_name(index),
            ),
        )
    )

    before, after = _resolution(source), _resolution(changed)

    assert len(before.expected_hydrogen_atom_names) == 6
    assert after.source.is_resolved()
    assert len(after.expected_hydrogen_atom_names) == 4
    assert any(bond.order == 2 for bond in after.heavy_bond_definitions)
    assert not after.has_hydrogen_topology(changed, residue)
    assert _resolution(source) == before


def test_large_retained_context_binds_native_ff_without_solve() -> None:
    polymer_id = ResidueId("A", 1)
    source = build_bonded_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=polymer_id,
                        atoms=(
                            atom_payload("N", "N", Vec3(0, 4, 0), formal_charge=1),
                            atom_payload("CA", "C", Vec3(1.4, 4, 0)),
                            atom_payload("C", "C", Vec3(2.8, 4, 0)),
                            atom_payload("O", "O", Vec3(3.8, 4, 0)),
                            atom_payload(
                                "OXT", "O", Vec3(2.8, 2.75, 0), formal_charge=-1
                            ),
                            atom_payload("CB", "C", Vec3(1.4, 2.6, 0)),
                            atom_payload("OG", "O", Vec3(1.4, 1.4, 0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=LIGAND_ID,
                atoms=tuple(
                    atom_payload(
                        f"C{index:02d}", "C", Vec3(1.26 * index, 0.88 * (index % 2), 0)
                    )
                    for index in range(38)
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    library = build_default_component_library()
    prepared = add_retained_non_polymer_hydrogens(
        materialize_hydrogens_core(source).structure, library
    ).structure
    scope = AtomSetScope((AtomRef(polymer_id, "OG"),))
    snapshot = ProteinStructureSnapshot.from_structure(prepared)

    facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        scope,
        component_library=library,
        context_radius_angstrom=3.0,
    )
    problem = build_continuous_relaxation_problem(
        prepared,
        LocalScopeSpec.from_atoms(scope.atom_refs),
        component_library=library,
        context_radius_angstrom=3.0,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    force_field = build_rdkit_force_field(
        molecule,
        problem=problem,
        restraint_library=build_default_restraint_library(),
        rdkit_atom_index_by_structure_atom_index=mapping,
    )

    require_atom_scope_continuous_relaxation_execution(facts)
    assert len(_resolution(prepared).expected_hydrogen_atom_names) == 78
    ligand_indices = set(
        prepared.constitution.atom_indices_for_residue_index(
            prepared.constitution.residue_index(LIGAND_ID)
        )
    )
    assert ligand_indices <= set(problem.region.fixed_context_atom_indices)
    assert not ligand_indices.intersection(problem.region.movable_atom_indices)
    assert isfinite(force_field.CalcEnergy())


def test_partial_large_source_h_inventory_survives_completion() -> None:
    prepared = _prepared_alkane()
    residue = prepared.constitution.ligands[0]
    index = prepared.constitution.residue_index(LIGAND_ID)
    removed_names = tuple(_hydrogen_anchors(prepared))[-8:]
    source = prepared.with_updated_residue_facets_batch(
        (
            (
                residue.without_atom_sites(removed_names),
                prepared.residue_geometry(index).without_atoms(removed_names),
                tuple(
                    (name, charge)
                    for name, charge in prepared.residue_formal_charge_by_atom_name(
                        index
                    )
                    if name not in removed_names
                ),
            ),
        )
    )
    original_geometry = source.residue_geometry(index)
    original_names = source.constitution.ligands[0].atom_site_names()
    original_anchors = _hydrogen_anchors(source)
    assert len(original_anchors) == 70
    assert len(original_anchors) * 78 > 4096

    result = add_retained_non_polymer_hydrogens(source, ComponentLibrary({}))

    assert len(_hydrogen_anchors(result.structure)) == 78
    final_residue = result.structure.constitution.ligands[0]
    final_geometry = result.structure.residue_geometry(index)
    for name in original_names:
        assert final_residue.atom_site(name) == source.constitution.ligands[
            0
        ].atom_site(name)
        assert final_geometry.atom_geometry(name) == original_geometry.atom_geometry(
            name
        )
    assert all(
        _hydrogen_anchors(result.structure)[name] == parent
        for name, parent in original_anchors.items()
    )
    assert _resolution(result.structure).has_hydrogen_topology(
        result.structure, final_residue
    )
