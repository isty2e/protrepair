"""Contextual sulfur preferences preserve source chemistry and metal uncertainty."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.microstate import (
    place_polymer_microstate_hydrogens,
)
from protrepair.transformer.polymer_microstate import apply_polymer_microstate_patches

CYS = ResidueId("P", 17)
HEME = ResidueId("L", 8)
SG = AtomRef(CYS, "SG")
FE = AtomRef(HEME, "FE")


def _observe(source):
    return source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(
                source.provenance.ingress,
                observation=StructureObservation.from_source_facets(
                    constitution=source.constitution,
                    geometry=source.geometry,
                    topology=source.topology,
                ),
            ),
        )
    )


def _with_topology(source, topology):
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _with_bonds(source, bonds, *, observed=True):
    current = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(bonds),
        ),
    )
    return _observe(current) if observed else current


@pytest.fixture(scope="module")
def heme_source():
    library = build_default_component_library()
    template = library.require("HEM")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    atoms = []
    for atom in semantics.component.heavy_atoms():
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(atom.atom_name, atom.element, Vec3(*atom.ideal_position))
        )
    source = build_structure(
        chains=(
            chain_payload(
                "P",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=CYS,
                        atoms=tuple(
                            atom_payload(name, name[0], Vec3(*position))
                            for name, position in (
                                ("N", (-1.2, 0.3, 0.2)),
                                ("CA", (0, 0, 0)),
                                ("C", (1.4, 0, 0)),
                                ("O", (2.1, -0.9, 0.3)),
                                ("CB", (0.1, 1.4, 0)),
                                ("SG", (0, 2, 1.5)),
                            )
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="HEM", residue_id=HEME, is_hetero=True, atoms=tuple(atoms)
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bond = TopologyBond(
        source.constitution.atom_index(SG),
        source.constitution.atom_index(FE),
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="independent-heme-site",
            reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
        ),
    )
    cysteine = library.require("CYS")
    bonds = tuple(
        TopologyBond(
            source.constitution.atom_index(AtomRef(CYS, defined.atom_name_1)),
            source.constitution.atom_index(AtomRef(CYS, defined.atom_name_2)),
            order=defined.order,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for defined in cysteine.definition.bonds
        if source.constitution.resolve_atom_index(AtomRef(CYS, defined.atom_name_1))
        is not None
        and source.constitution.resolve_atom_index(AtomRef(CYS, defined.atom_name_2))
        is not None
    )
    return _with_bonds(source, (*bonds, bond))


def _charged(source, charge):
    atoms = list(source.topology.atom_topologies)
    atoms[source.constitution.atom_index(SG).value] = AtomTopology(charge)
    return _observe(
        _with_topology(
            source,
            StructureTopology(
                constitution=source.constitution,
                atom_topologies=tuple(atoms),
                bonds=source.topology.bonds,
            ),
        )
    )


def _target(source, *, library=None, requests=None, residue_id=CYS):
    prepared = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source),
        build_default_component_library() if library is None else library,
        requests=requests,
    )
    return next(
        target
        for target in prepared.targets_for(residue_id)
        if target.site.kind is PolymerChemicalSite.SIDECHAIN
    )


def _source_hydrogen(source, isotope):
    index = source.constitution.residue_index(CYS)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite("SOURCEH", isotope)),
        residue_geometry=geometry.with_atom_geometry(
            "SOURCEH",
            AtomGeometry(geometry.position("SG").with_offset(1.34, 0, 0), 0.7, 33),
        ),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    bond = TopologyBond(
        source.constitution.atom_index(SG),
        source.constitution.atom_index(AtomRef(CYS, "SOURCEH")),
        order=1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="observed-thiol",
            reported_order=1,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    return _with_bonds(source, (*source.topology.bonds, bond))


def _monatomic_partner(source, element):
    residue = source.constitution.residue_or_ligand(HEME)
    assert residue is not None
    current = source.without_atom_refs(
        tuple(
            AtomRef(HEME, atom.name) for atom in residue.atom_sites if atom.name != "FE"
        )
    )
    residue = current.constitution.residue_or_ligand(HEME)
    assert residue is not None
    current = current.with_updated_residue_facets(
        replace(residue, component_id=element, atom_sites=(AtomSite("FE", element),)),
        residue_geometry=current.residue_geometry(
            current.constitution.residue_index(HEME)
        ),
    )
    return _observe(current)


@pytest.mark.parametrize("charge", (None, 0, -1))
def test_heme_preference_preserves_known_source_charge(heme_source, charge):
    source = heme_source if charge is None else _charged(heme_source, charge)
    target = _target(source)
    graph = target.resolution.graph
    assert graph is not None
    expected = -1 if charge is None else charge
    assert (graph.atom("SG").charge, graph.atom("SG").hydrogens) == (
        expected,
        int(expected == 0),
    )
    assert target.resolution.basis is (
        MicrostateSelectionBasis.DEFAULT
        if charge is None
        else MicrostateSelectionBasis.EVIDENCE
    )
    assert target.resolution.superseded_source.is_empty()


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("charge", (None, 0, -1))
def test_heme_preference_does_not_discard_original_thiol(heme_source, isotope, charge):
    source = heme_source if charge is None else _charged(heme_source, charge)
    source = _source_hydrogen(source, isotope)
    target = _target(source)
    if charge == -1:
        assert target.resolution.status is MicrostateResolutionStatus.CONFLICT
        assert target.resolution.graph is None
        return
    assert target.resolution.graph is not None
    assert target.resolution.graph.atom("SG").charge == 0
    assert target.resolution.basis is MicrostateSelectionBasis.EVIDENCE
    prepared = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((CYS,))
    ).structure
    original = source.constitution.atom_index(AtomRef(CYS, "SOURCEH"))
    current = prepared.constitution.atom_index(AtomRef(CYS, "SOURCEH"))
    assert prepared.constitution.atom_site_at(
        current
    ) == source.constitution.atom_site_at(original)
    assert prepared.geometry.atom_geometry(current) == source.geometry.atom_geometry(
        original
    )
    assert _target(prepared).is_realized()


@pytest.mark.parametrize("element", ("FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG"))
@pytest.mark.parametrize("charge", (None, 0, -1))
def test_other_metal_donor_admission_is_not_a_charge_default(
    heme_source, element, charge
):
    source = _monatomic_partner(heme_source, element)
    if charge is not None:
        source = _charged(source, charge)
    resolution = _target(source).resolution
    if charge is None:
        assert resolution.status is MicrostateResolutionStatus.AMBIGUOUS
        assert resolution.graph is None
    else:
        assert resolution.graph is not None
        assert resolution.graph.atom("SG").charge == charge
        assert resolution.basis is MicrostateSelectionBasis.EVIDENCE


@pytest.mark.parametrize("element", ("CA", "MG", "C", "N"))
def test_unsupported_partner_cannot_be_prepared_as_heme(heme_source, element):
    resolution = _target(_monatomic_partner(heme_source, element)).resolution
    assert resolution.status is MicrostateResolutionStatus.UNSUPPORTED


@pytest.mark.parametrize("custom", (False, True))
def test_active_heme_library_authority_controls_contextual_preference(
    heme_source, custom
):
    library = build_default_component_library()
    if custom:
        stock = library.require("HEM")
        library = library.with_template(
            replace(
                stock, definition=replace(stock.definition, formal_charges={"FE": 2})
            )
        )
    else:
        library = ComponentLibrary(
            {
                name: template
                for name, template in library.templates.items()
                if name != "HEM"
            }
        )
    assert (
        _target(heme_source, library=library).resolution.status
        is MicrostateResolutionStatus.AMBIGUOUS
    )


@pytest.mark.parametrize("change", ("missing", "element", "extra"))
def test_current_heme_scaffold_must_support_the_preparation(heme_source, change):
    index = heme_source.constitution.residue_index(HEME)
    residue = heme_source.constitution.residue_site_at(index)
    if change == "missing":
        changed = heme_source.without_atom_refs((AtomRef(HEME, "CHA"),))
    else:
        name, element = ("CHA", "O") if change == "element" else ("EXTRA", "C")
        changed = heme_source.with_updated_residue_facets(
            residue.with_atom_site(AtomSite(name, element)),
            residue_geometry=heme_source.residue_geometry(index).with_atom_geometry(
                name, AtomGeometry(Vec3(10, 10, 10))
            ),
        )
    assert _target(changed).resolution.status is MicrostateResolutionStatus.AMBIGUOUS


@pytest.mark.parametrize("change", ("removed", "element"))
def test_current_heme_cannot_hide_changed_original_scaffold(heme_source, change):
    index = heme_source.constitution.residue_index(HEME)
    residue = heme_source.constitution.residue_site_at(index)
    name, element = ("SOURCEC", "C") if change == "removed" else ("CHA", "O")
    original = _observe(
        heme_source.with_updated_residue_facets(
            residue.with_atom_site(AtomSite(name, element)),
            residue_geometry=heme_source.residue_geometry(index).with_atom_geometry(
                name, AtomGeometry(Vec3(10, 10, 10))
            ),
        )
    )
    changed = heme_source.with_provenance(original.provenance)
    assert _target(changed).resolution.status is MicrostateResolutionStatus.AMBIGUOUS


def test_compatible_saved_choice_and_explicit_reset_keep_their_authority(heme_source):
    target = _target(heme_source)
    patch = place_polymer_microstate_hydrogens(
        target.context,
        CYS,
        target.site,
        override=MicrostateConstraints(charges=(("SG", 0),)),
    )
    neutral = apply_polymer_microstate_patches(heme_source, (patch,))
    saved = _target(neutral).resolution
    assert saved.graph is not None and saved.graph.atom("SG").charge == 0
    assert saved.basis is MicrostateSelectionBasis.OVERRIDE
    reset = _target(neutral, requests={(CYS, PolymerChemicalSite.SIDECHAIN): None})
    assert reset.resolution.graph is not None
    assert reset.resolution.graph.atom("SG").charge == -1
    assert reset.resolution.basis is MicrostateSelectionBasis.DEFAULT


def test_heme_default_is_realized_atomically_without_changing_source_or_iron(
    heme_source,
):
    result = materialize_hydrogens_core(
        heme_source, target_residue_ids=frozenset((CYS,))
    )
    prepared = result.structure
    assert prepared.topology.formal_charge(prepared.constitution.atom_index(SG)) == -1
    assert prepared.topology.formal_charge(prepared.constitution.atom_index(FE)) is None
    residue = prepared.constitution.residue_or_ligand(CYS)
    assert residue is not None and not residue.has_atom_site("HG")
    assert _target(prepared).is_realized()
    for index, _ in heme_source.geometry.iter_positions():
        ref = heme_source.constitution.atom_ref_at(index)
        current = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            current
        ) == heme_source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(
            current
        ) == heme_source.geometry.atom_geometry(index)
    assert (
        prepared.provenance.ingress.observation
        is heme_source.provenance.ingress.observation
    )


@pytest.mark.parametrize("chain", ("A", "B"))
def test_deposited_1f4t_cysteine_uses_heme_preparation(chain):
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb1f4t.ent")
    residue_id = ResidueId(chain, 317)
    target = _target(source, residue_id=residue_id)
    assert target.resolution.graph is not None
    assert target.resolution.graph.atom("SG").charge == -1
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue_id,))
    )
    residue = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None and not residue.has_atom_site("HG")
    assert _target(result.structure, residue_id=residue_id).is_realized()


@pytest.mark.parametrize("format", (FileFormat.PDB, FileFormat.MMCIF))
def test_heme_thiolate_roundtrip_keeps_charge_and_typed_connection(heme_source, format):
    prepared = materialize_hydrogens_core(
        heme_source, target_residue_ids=frozenset((CYS,))
    ).structure
    restored = read_structure_string(write_structure_string(prepared, format), format)
    assert restored.topology.formal_charge(restored.constitution.atom_index(SG)) == -1
    assert restored.topology.formal_charge(restored.constitution.atom_index(FE)) is None
    bond = restored.topology.bond_between(
        restored.constitution.atom_index(SG), restored.constitution.atom_index(FE)
    )
    assert (
        bond is not None
        and bond.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    residue = restored.constitution.residue_or_ligand(CYS)
    assert residue is not None and not residue.has_atom_site("HG")


@pytest.mark.parametrize(
    "basis", (MicrostateSelectionBasis.DEFAULT, MicrostateSelectionBasis.OVERRIDE)
)
def test_stale_site_choices_are_reconsidered_or_refused_at_their_original_authority(
    heme_source, basis
):
    target = _target(heme_source)
    patch = place_polymer_microstate_hydrogens(
        target.context,
        CYS,
        target.site,
        preferences=target.preferences,
    )
    prepared = apply_polymer_microstate_patches(heme_source, (patch,))
    choices = prepared.provenance.microstate_choices
    choice = next(entry for entry in choices if entry.residue_id == CYS)
    atom = replace(choice.graph.atom("SG"), charge=0, hydrogens=0, boundary_order=2)
    stale = replace(choice, graph=replace(choice.graph, atoms=(atom,)), basis=basis)
    changed = prepared.with_provenance(
        replace(prepared.provenance, microstate_choices=(stale,))
    )
    result = _target(changed).resolution
    if basis is MicrostateSelectionBasis.DEFAULT:
        assert result.graph is not None and result.graph.atom("SG").charge == -1
    else:
        assert result.status is MicrostateResolutionStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "context", ("absent_observation", "incomplete_original", "reordered")
)
def test_heme_applicability_uses_stable_identity_without_requiring_complete_source(
    heme_source, context
):
    if context == "absent_observation":
        source = heme_source.with_provenance(
            replace(
                heme_source.provenance,
                ingress=replace(heme_source.provenance.ingress, observation=None),
            )
        )
    elif context == "incomplete_original":
        original = _observe(heme_source.without_atom_refs((AtomRef(HEME, "CHA"),)))
        source = heme_source.with_provenance(original.provenance)
    else:
        index = heme_source.constitution.residue_index(HEME)
        residue = heme_source.constitution.residue_site_at(index)
        source = heme_source.with_updated_residue_facets(
            replace(residue, atom_sites=tuple(reversed(residue.atom_sites))),
            residue_geometry=heme_source.residue_geometry(index),
        )
    target = _target(source)
    assert target.resolution.graph is not None
    assert target.resolution.graph.atom("SG").charge == -1


@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    ),
)
def test_other_sulfur_relationships_do_not_enter_isolated_thiolate_preparation(
    heme_source, relationship
):
    bonds = tuple(
        replace(
            bond,
            relationship_type=relationship,
            order=1,
            source_metadata=replace(
                bond.source_metadata,
                reported_order=1,
                reported_relationship_type=relationship,
            ),
        )
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION
        else bond
        for bond in heme_source.topology.bonds
    )
    source = _with_bonds(heme_source, bonds)
    prepared = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    assert not any(
        t.site.kind is PolymerChemicalSite.SIDECHAIN for t in prepared.targets_for(CYS)
    )


def test_multiple_sulfur_donors_do_not_enable_the_heme_preference(heme_source):
    second = replace(
        heme_source.topology.bonds[-1],
        atom_index_2=heme_source.constitution.atom_index(AtomRef(HEME, "NA")),
        source_metadata=None,
    )
    source = _with_bonds(heme_source, (*heme_source.topology.bonds, second))
    assert _target(source).resolution.status is MicrostateResolutionStatus.UNSUPPORTED


def test_custom_cysteine_template_is_not_replaced_by_the_heme_assumption(heme_source):
    library = build_default_component_library()
    stock = library.require("CYS")
    custom = replace(
        stock, definition=replace(stock.definition, formal_charges={"SG": -1})
    )
    assert (
        _target(heme_source, library=library.with_template(custom)).resolution.status
        is MicrostateResolutionStatus.UNSUPPORTED
    )
