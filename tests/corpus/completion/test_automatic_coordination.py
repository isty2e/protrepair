"""Contextual preparation must not invent the source metal's oxidation state."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import ingress_options

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import MicrostateSelectionBasis
from protrepair.chemistry.retained_non_polymer.coordination import (
    coordinated_component_preparation,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    resolve_local_bond_planning_support,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.contracts import LigandPolicy, requested_process_goal


@pytest.fixture(scope="module")
def source_2d6c():
    return read_structure(
        Path(__file__).resolve().parents[2] / "fixtures/corpus/pdb2d6c.ent"
    )


@pytest.fixture(scope="module")
def source_1f4t():
    return read_structure(
        Path(__file__).resolve().parents[2] / "fixtures/corpus/pdb1f4t.ent"
    )


def _with_source_topology(structure, topology):
    observation = StructureObservation.from_source_facets(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=replace(
            structure.provenance,
            ingress=replace(structure.provenance.ingress, observation=observation),
        ),
    )


def _charged_source(structure, ref, charge):
    charges = list(structure.topology.atom_topologies)
    charges[structure.constitution.atom_index(ref).value] = AtomTopology(charge)
    return _with_source_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(charges),
            bonds=structure.topology.bonds,
        ),
    )


def _hydrogen_source(structure, parent):
    index = structure.constitution.residue_index(parent.residue_id)
    residue = structure.constitution.residue_site_at(index)
    geometry = structure.residue_geometry(index)
    hydrogen = AtomSite("SOURCEH", "D")
    updated = structure.with_updated_residue_facets(
        residue.with_atom_site(hydrogen),
        residue_geometry=geometry.with_atom_geometry(
            hydrogen.name,
            AtomGeometry(geometry.position(parent.atom_name).with_offset(1.0, 0, 0)),
        ),
        formal_charge_by_atom_name=structure.residue_formal_charge_by_atom_name(index),
    )
    bond = TopologyBond(
        updated.constitution.atom_index(parent),
        updated.constitution.atom_index(AtomRef(parent.residue_id, hydrogen.name)),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="independent-source-h",
            reported_order=1,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    return _with_source_topology(
        updated,
        StructureTopology(
            constitution=updated.constitution,
            atom_topologies=updated.topology.atom_topologies,
            bonds=(*updated.topology.bonds, bond),
        ),
    )


def test_1f4t_both_pim_scaffolds_publish_h_and_preserve_original_site(source_1f4t):
    source = source_1f4t
    targets = frozenset(ResidueId(chain, 411) for chain in ("A", "B"))
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=targets,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    prepared = result.structure
    assert (
        len(prepared.constitution.atom_slots)
        == len(source.constitution.atom_slots) + 16
    )
    assert prepared.provenance.ingress is source.provenance.ingress
    assert not any(
        issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
        for issue in result.issues
    )
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(index)
        charge = source.topology.formal_charge(index)
        if charge is not None or ref.residue_id not in targets:
            assert prepared.topology.formal_charge(current) == charge
    for bond in source.topology.bonds:
        indices = tuple(
            prepared.constitution.atom_index(source.constitution.atom_ref_at(index))
            for index in bond.endpoint_pair()
        )
        current = prepared.topology.bond_between(*indices)
        assert current is not None
        assert current.source_metadata == bond.source_metadata
        assert current.provenance is bond.provenance
        if bond.relationship_type is not BondRelationshipType.UNKNOWN:
            assert current.relationship_type is bond.relationship_type
        if bond.order is not None:
            assert current.order == bond.order

    library = build_default_component_library()
    for chain in ("A", "B"):
        residue_id = ResidueId(chain, 411)
        index = prepared.constitution.residue_index(residue_id)
        residue = prepared.constitution.residue_site_at(index)
        assert len(residue.atom_sites) == 19
        anchors = prepared.topology.residue_hydrogen_anchor_by_name(
            constitution=prepared.constitution, residue_index=index
        )
        assert len(anchors) == 8
        assert "N3" not in anchors.values()
        assert list(anchors.values()).count("N1") == 1
        resolution = resolve_retained_non_polymer_chemistry(
            prepared, residue, component_library=library, allow_rdkit_fallback=False
        )
        assert (
            resolution.source
            is RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
        )
        assert resolution.has_hydrogen_topology(prepared, residue)
        assert resolution.has_formal_charge_topology(prepared, residue)
        assert resolution.has_coordination_topology(prepared)
        metal = prepared.constitution.atom_index(AtomRef(ResidueId(chain, 410), "FE"))
        assert prepared.topology.formal_charge(metal) is None

    repeated = add_retained_non_polymer_hydrogens(prepared, target_residue_ids=targets)
    assert repeated.structure.constitution == prepared.constitution
    assert repeated.structure.geometry == prepared.geometry
    assert repeated.structure.topology == prepared.topology


@pytest.mark.parametrize("source_fixture", ("source_1f4t", "source_2d6c"))
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize("prepared", (False, True))
def test_deposited_known_coordinated_ligands_keep_passive_bond_support(
    source_fixture, strict, prepared, request, monkeypatch
):
    source = request.getfixturevalue(source_fixture)
    components = {"PIM"} if source_fixture == "source_1f4t" else {"HME", "IMD"}
    targets = frozenset(
        residue.residue_id
        for residue in source.constitution.ligands
        if residue.component_id in components
    )
    if prepared:
        source = add_retained_non_polymer_hydrogens(
            source, target_residue_ids=targets
        ).structure

    def fail_fallback(*args, **kwargs):
        raise AssertionError("known deposited chemistry used coordinate fallback")

    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry."
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    snapshot = ProteinStructureSnapshot.from_structure(source)
    library = build_default_component_library()
    for residue_id in targets:
        index = source.constitution.residue_index(residue_id)
        residue = source.constitution.residue_site_at(index)
        chemistry = resolve_retained_non_polymer_chemistry(
            source, residue, component_library=library, allow_rdkit_fallback=not strict
        )
        support = resolve_local_bond_planning_support(
            snapshot,
            index,
            residue,
            movable_atom_indices=(),
            component_library=library,
            allow_retained_non_polymer_rdkit_fallback=not strict,
        )
        assert (
            chemistry.source
            is RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
        )
        assert (
            support.mode is LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
        )
        assert set(support.fallback_bond_definitions) == set(
            chemistry.heavy_bond_definitions
        )
        if residue.has_atom_site("FE"):
            assert (
                source.topology.formal_charge(
                    source.constitution.atom_index(AtomRef(residue_id, "FE"))
                )
                is None
            )


def test_1f4t_pim_goals_are_reobserved_without_claiming_full_metal_site(source_1f4t):
    result = process_structure(
        source_1f4t,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert result.requested_goals_fully_satisfied()
    assert result.structure.provenance.ingress is source_1f4t.provenance.ingress
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(result.structure), build_default_component_library()
    )
    for index, _ in source_1f4t.geometry.iter_positions():
        ref = source_1f4t.constitution.atom_ref_at(index)
        current = result.structure.constitution.atom_index(ref)
        assert result.structure.constitution.atom_site_at(
            current
        ) == source_1f4t.constitution.atom_site_at(index)
        assert result.structure.geometry.atom_geometry(
            current
        ) == source_1f4t.geometry.atom_geometry(index)
    for chain in ("A", "B"):
        index = result.structure.constitution.residue_index(ResidueId(chain, 411))
        residue = result.structure.constitution.residue_site_at(index)
        assert len(residue.atom_sites) == 19
        metal = result.structure.constitution.atom_index(
            AtomRef(ResidueId(chain, 410), "FE")
        )
        assert result.structure.topology.formal_charge(metal) is None
        cysteine = ResidueId(chain, 317)
        target = next(
            entry
            for entry in preparation.targets_for(cysteine)
            if entry.site.kind is PolymerChemicalSite.SIDECHAIN
        )
        assert target.is_realized()
        assert target.resolution.basis is MicrostateSelectionBasis.DEFAULT
        assert target.resolution.graph is not None
        # Automatic ligand preparation does not select a polymer thiolate state.
        assert target.resolution.graph.atom("SG").charge == 0
        assert target.resolution.graph.atom("SG").hydrogens == 1
        cysteine_site = result.structure.constitution.residue_or_ligand(cysteine)
        assert cysteine_site is not None and cysteine_site.has_atom_site("HG")
        sulfur = result.structure.constitution.atom_index(AtomRef(cysteine, "SG"))
        bond = result.structure.topology.bond_between(sulfur, metal)
        original = source_1f4t.topology.bond_between(
            source_1f4t.constitution.atom_index(AtomRef(cysteine, "SG")),
            source_1f4t.constitution.atom_index(AtomRef(ResidueId(chain, 410), "FE")),
        )
        assert bond is not None and original is not None
        assert bond.relationship_type is BondRelationshipType.METAL_COORDINATION
        assert bond.source_metadata == original.source_metadata
        assert bond.order is None


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_1f4t_pim_preparation_roundtrip_preserves_typed_axial_connections(
    source_1f4t, file_format
):
    targets = frozenset(ResidueId(chain, 411) for chain in ("A", "B"))
    prepared = add_retained_non_polymer_hydrogens(
        source_1f4t, target_residue_ids=targets
    ).structure
    restored = read_structure_string(
        write_structure_string(prepared, file_format=file_format),
        file_format=file_format,
    )
    replayed = add_retained_non_polymer_hydrogens(
        restored, target_residue_ids=targets
    ).structure
    for chain in ("A", "B"):
        residue_id = ResidueId(chain, 411)
        index = replayed.constitution.residue_index(residue_id)
        assert (
            len(
                replayed.topology.residue_hydrogen_anchor_by_name(
                    constitution=replayed.constitution, residue_index=index
                )
            )
            == 8
        )
        refs = (AtomRef(residue_id, "N3"), AtomRef(ResidueId(chain, 410), "FE"))
        bond = replayed.topology.bond_between(
            *(replayed.constitution.atom_index(ref) for ref in refs)
        )
        assert bond is not None
        assert bond.relationship_type is BondRelationshipType.METAL_COORDINATION
        assert bond.order is None
        assert not bond.aromatic


def test_2d6c_automatic_bound_preparation_keeps_unknown_fe_and_source_atoms(
    source_2d6c,
):
    source = source_2d6c
    library = build_default_component_library()
    result = add_retained_non_polymer_hydrogens(source, component_library=library)
    prepared = result.structure
    for chain in ("A", "B"):
        for number, expected_hydrogens in ((154, 34), (155, 4)):
            residue_id = ResidueId(chain, number)
            residue = prepared.constitution.residue_or_ligand(residue_id)
            assert residue is not None
            assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
                expected_hydrogens
            )
        fe = prepared.constitution.atom_index(AtomRef(ResidueId(chain, 154), "FE"))
        assert prepared.topology.formal_charge(fe) is None
        assert (
            sum(
                bond.involves(fe)
                and bond.relationship_type is BondRelationshipType.METAL_COORDINATION
                for bond in prepared.topology.bonds
            )
            == 6
        )

    assert prepared.provenance.ingress is source.provenance.ingress
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(index)
    for bond in source.topology.bonds:
        first, second = (
            prepared.constitution.atom_index(source.constitution.atom_ref_at(index))
            for index in bond.endpoint_pair()
        )
        current = prepared.topology.bond_between(first, second)
        assert current is not None
        assert current.source_metadata == bond.source_metadata
        assert current.provenance is bond.provenance

    _, chemistry = StructureChemistryObservation(
        prepared, component_library=library
    ).whole_structure_facts()
    for fact in chemistry.retained_non_polymer_facts:
        if fact.component_id in {"HME", "IMD"}:
            assert not fact.requires_hydrogen_completion()
            assert not fact.requires_topology_resolution()

    repeated = add_retained_non_polymer_hydrogens(prepared, component_library=library)
    assert repeated.structure.constitution == prepared.constitution
    assert repeated.structure.geometry == prepared.geometry
    assert repeated.structure.topology == prepared.topology


@pytest.mark.parametrize("charge", (None, 2, 3))
def test_2d6c_preparation_retains_only_source_assigned_fe_charge(source_2d6c, charge):
    residue_id = ResidueId("A", 154)
    ref = AtomRef(residue_id, "FE")
    source = _charged_source(source_2d6c, ref, charge)
    preparation = coordinated_component_preparation(
        source,
        source.constitution.residue_site_at(
            source.constitution.residue_index(residue_id)
        ),
        build_default_component_library(),
    )
    assert preparation is not None
    assert dict(preparation.formal_charge_requirements()).get("FE") == charge
    assert len(preparation.hypotheses) == (2 if charge is None else 1)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((residue_id,))
    )
    prepared = result.structure
    assert (
        prepared.topology.formal_charge(prepared.constitution.atom_index(ref)) == charge
    )
    anchors = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution,
        residue_index=prepared.constitution.residue_index(residue_id),
    )
    assert len(anchors) == 34
    assert not {"FE", "NA", "NB", "NC", "ND", "O1A", "O2A", "O1D", "O2D"}.intersection(
        anchors.values()
    )
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(residue_id, "O1A"))
        )
        == -1
    )
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(residue_id, "O1D"))
        )
        == -1
    )


@pytest.mark.parametrize("oxygen", ("O1A", "O2A", "O1D", "O2D"))
def test_2d6c_compatible_source_carboxyl_deuterium_survives(source_2d6c, oxygen):
    residue_id = ResidueId("A", 154)
    source = _hydrogen_source(source_2d6c, AtomRef(residue_id, oxygen))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((residue_id,))
    )
    prepared = result.structure
    residue = prepared.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 35
    ref = AtomRef(residue_id, "SOURCEH")
    original = source.constitution.atom_index(ref)
    current = prepared.constitution.atom_index(ref)
    assert prepared.constitution.atom_site_at(
        current
    ) == source.constitution.atom_site_at(original)
    assert prepared.geometry.atom_geometry(current) == source.geometry.atom_geometry(
        original
    )
    anchors = prepared.topology.residue_hydrogen_anchor_by_name(
        constitution=prepared.constitution,
        residue_index=prepared.constitution.residue_index(residue_id),
    )
    assert anchors["SOURCEH"] == oxygen
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(residue_id, "FE"))
        )
        is None
    )


@pytest.mark.parametrize("oxygen", ("O1A", "O2A", "O1D", "O2D"))
def test_2d6c_source_carboxylate_charge_is_not_bound_to_one_oxygen_name(
    source_2d6c, oxygen
):
    residue_id = ResidueId("A", 154)
    ref = AtomRef(residue_id, oxygen)
    source = _charged_source(source_2d6c, ref, -1)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((residue_id,))
    )
    prepared = result.structure
    residue = prepared.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 34
    assert prepared.topology.formal_charge(prepared.constitution.atom_index(ref)) == -1
    carbon = "CGA" if oxygen.endswith("A") else "CGD"
    bond = prepared.topology.bond_between(
        prepared.constitution.atom_index(ref),
        prepared.constitution.atom_index(AtomRef(residue_id, carbon)),
    )
    assert bond is not None
    assert bond.order == 1


@pytest.mark.parametrize(
    "donor", (AtomRef(ResidueId("A", 155), "N1"), AtomRef(ResidueId("A", 93), "NE2"))
)
def test_2d6c_partner_donor_hydrogen_blocks_the_coupled_site(source_2d6c, donor):
    source = _hydrogen_source(source_2d6c, donor)
    target = frozenset((ResidueId("A", 154), ResidueId("A", 155)))
    result = add_retained_non_polymer_hydrogens(source, target_residue_ids=target)
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_2d6c_charged_h_on_non_donor_his_n_blocks_automatic_site_preparation(
    source_2d6c,
):
    parent = AtomRef(ResidueId("A", 93), "ND1")
    source = _hydrogen_source(source_2d6c, parent)
    source = _charged_source(source, AtomRef(parent.residue_id, "SOURCEH"), 1)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((ResidueId("A", 154), ResidueId("A", 155))),
    )
    assert result.structure is source
    assert not result.repairs
    assert result.issues


@pytest.mark.parametrize("failure", ("metal_charge", "donor_h"))
@pytest.mark.parametrize("fallback", (False, True))
def test_one_contradictory_site_does_not_veto_an_independent_site(
    source_2d6c, failure, fallback
):
    source = (
        _charged_source(source_2d6c, AtomRef(ResidueId("A", 154), "FE"), 4)
        if failure == "metal_charge"
        else _hydrogen_source(source_2d6c, AtomRef(ResidueId("A", 155), "N1"))
    )
    targets = frozenset(
        ResidueId(chain, number) for chain in ("A", "B") for number in (154, 155)
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=targets,
        allow_retained_non_polymer_rdkit_fallback=fallback,
    )
    for number in (154, 155):
        residue_id = ResidueId("A", number)
        assert result.structure.constitution.residue_or_ligand(
            residue_id
        ) == source.constitution.residue_or_ligand(residue_id)
        assert result.structure.residue_geometry(
            result.structure.constitution.residue_index(residue_id)
        ) == source.residue_geometry(source.constitution.residue_index(residue_id))
    for number, count in ((154, 34), (155, 4)):
        residue_id = ResidueId("B", number)
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None
        assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == count
    assert {
        issue.residue_id
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
    } == {ResidueId("A", 154), ResidueId("A", 155)}
    assert not any(
        issue.kind is ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED
        for issue in result.issues
    )


@pytest.mark.parametrize("charge", (0, 1, 4))
def test_2d6c_unsupported_source_fe_charge_does_not_enter_organic_fallback(
    source_2d6c, charge
):
    source = _charged_source(source_2d6c, AtomRef(ResidueId("A", 154), "FE"), charge)
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((ResidueId("A", 154), ResidueId("A", 155))),
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs
    assert all(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )


def test_2d6c_missing_partner_template_leaves_preparation_unresolved(source_2d6c):
    source = source_2d6c
    library = ComponentLibrary({})
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=library,
        target_residue_ids=frozenset((ResidueId("A", 154), ResidueId("A", 155))),
    )
    assert result.structure is source
    assert not result.repairs
    assert len(result.issues) == 2
    assert all(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )
    for number in (154, 155):
        residue = source.constitution.residue_site_at(
            source.constitution.residue_index(ResidueId("A", number))
        )
        resolution = resolve_retained_non_polymer_chemistry(
            source, residue, component_library=library
        )
        assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        assert resolution.failure_reason
        assert not resolution.expected_hydrogen_atom_names


def test_2d6c_public_h_goal_prepares_ligands_and_bound_histidine(source_2d6c):
    result = process_structure(
        source_2d6c,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert result.requested_goals_fully_satisfied()
    # Proximity-only PDB parsing cannot recover HME's typed metal relationships.
    assert all(
        issue.kind is ValidationIssueKind.PARSER_READABILITY
        for issue in result.issues
        if issue.severity.value == "error"
    )
    prepared = result.structure
    for chain in ("A", "B"):
        for number, count in ((154, 34), (155, 4)):
            residue = prepared.constitution.residue_or_ligand(ResidueId(chain, number))
            assert residue is not None
            assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == count
        histidine = ResidueId(chain, 93)
        anchors = prepared.topology.residue_hydrogen_anchor_by_name(
            constitution=prepared.constitution,
            residue_index=prepared.constitution.residue_index(histidine),
        )
        assert "ND1" in anchors.values()
        assert "NE2" not in anchors.values()


def test_2d6c_custom_partner_template_is_not_overridden_by_site_preference(source_2d6c):
    source = source_2d6c
    library = build_default_component_library().with_template(
        ResidueTemplate(
            ChemicalComponentDefinition("IMD", ("N1", "C2", "N3", "C4", "C5"))
        )
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=library,
        target_residue_ids=frozenset((ResidueId("A", 154),)),
    )
    assert result.structure is source
    assert not result.repairs
    assert result.issues
    assert all(
        issue.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        for issue in result.issues
    )


@pytest.mark.parametrize("iron_charge", (2, 3))
def test_2d6c_common_canonical_graph_is_native_valid_under_both_iron_hypotheses(
    source_2d6c, iron_charge
):
    residue_id = ResidueId("A", 154)
    prepared = add_retained_non_polymer_hydrogens(
        source_2d6c, target_residue_ids=frozenset((residue_id,))
    ).structure
    native = Chem.RWMol()
    indices = {}
    residue = prepared.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    for site in residue.atom_sites:
        index = prepared.constitution.atom_index(AtomRef(residue_id, site.name))
        charge = prepared.topology.formal_charge(index)
        atom = Chem.Atom(site.element_identity.chemical_symbol.title())
        atom.SetNoImplicit(True)
        if site.element == "FE":
            assert charge is None
            charge = iron_charge
        elif not site.is_hydrogen():
            assert charge is not None
        atom.SetFormalCharge(0 if charge is None else charge)
        indices[index] = native.AddAtom(atom)
    for bond in prepared.topology.bonds:
        first, second = bond.endpoint_pair()
        if first not in indices or second not in indices:
            continue
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION:
            if prepared.constitution.atom_site_at(first).element == "FE":
                first, second = second, first
            native.AddBond(indices[first], indices[second], Chem.BondType.DATIVE)
        else:
            assert bond.relationship_type is BondRelationshipType.COVALENT
            assert bond.order is not None
            native.AddBond(
                indices[first],
                indices[second],
                {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}[bond.order],
            )
    molecule = native.GetMol()
    Chem.SanitizeMol(molecule)
    assert sum(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()) == 34
    assert (
        sum(bond.GetBondType() == Chem.BondType.DATIVE for bond in molecule.GetBonds())
        == 4
    )


def test_2d6c_automatic_h_preparation_is_not_default_metal_ff_admission(source_2d6c):
    prepared = add_retained_non_polymer_hydrogens(source_2d6c).structure
    with pytest.raises(RefinementError, match="HME|IMD"):
        build_continuous_relaxation_problem(
            prepared,
            LocalScopeSpec.from_residues((ResidueId("A", 154),)),
            component_library=build_default_component_library(),
        )


def test_2d6c_mmcif_retains_unknown_fe_and_all_prepared_coordination(source_2d6c):
    prepared = add_retained_non_polymer_hydrogens(source_2d6c).structure
    reparsed = read_structure_string(
        write_structure_string(prepared, FileFormat.MMCIF), FileFormat.MMCIF
    )
    for chain in ("A", "B"):
        fe = reparsed.constitution.atom_index(AtomRef(ResidueId(chain, 154), "FE"))
        assert reparsed.topology.formal_charge(fe) is None
        assert (
            sum(
                bond.involves(fe)
                and bond.relationship_type is BondRelationshipType.METAL_COORDINATION
                for bond in reparsed.topology.bonds
            )
            == 6
        )
