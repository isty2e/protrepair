"""Typed donor context constrains source-aware HIS preparation and publication."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.microstate.catalog import (
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import (
    PolymerMicrostatePreparation,
    pras_microstate_preferences,
)
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolutionStatus,
)
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.microstate import (
    place_polymer_microstate_hydrogens,
)
from protrepair.transformer.polymer_microstate import (
    PolymerMicrostatePatch,
    apply_polymer_microstate_patches,
)

HIS = ResidueId("A", 41)
METAL = ResidueId("Z", 1)


@pytest.fixture(params=("FE", "CA"))
def coordination_metal(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def histidine_pdb() -> str:
    path = Path(__file__).parents[1] / "fixtures/corpus/pdb1afc.ent"
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if line.startswith("ATOM") and line[21] == "A" and int(line[22:26]) == 41
    )


def _source(
    text: str,
    *,
    donor: str = "NE2",
    element: str = "FE",
    relationship: BondRelationshipType = BondRelationshipType.METAL_COORDINATION,
    observed: bool = True,
    reverse: bool = False,
) -> ProteinStructure:
    lines = text.splitlines()
    if reverse:
        lines.reverse()
    lines.append(
        f"HETATM99999 {element:^4} {element:>3} Z   1    "
        f"{30.0:8.3f}{30.0:8.3f}{30.0:8.3f}{0.7:6.2f}{19.0:6.2f}"
        f"          {element:>2}\n"
    )
    source = read_structure_string("\n".join(lines), FileFormat.PDB)
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(HIS, donor)),
        source.constitution.atom_index(AtomRef(METAL, element)),
        order=None,
        relationship_type=relationship,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="donor-control",
            reported_relationship_type=relationship,
            reported_distance_angstrom=2.2,
        ),
    )
    template = build_default_component_library().require("HIS")
    heavy_bonds = tuple(
        TopologyBond(
            source.constitution.atom_index(AtomRef(HIS, edge.atom_name_1)),
            source.constitution.atom_index(AtomRef(HIS, edge.atom_name_2)),
            order=edge.order,
            aromatic=edge.aromatic,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for edge in template.definition.bonds
        if source.constitution.resolve_atom_index(AtomRef(HIS, edge.atom_name_1))
        is not None
        and source.constitution.resolve_atom_index(AtomRef(HIS, edge.atom_name_2))
        is not None
    )
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=(*heavy_bonds, bond),
    )
    observation = (
        StructureObservation.from_source_facets(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=topology,
        )
        if observed
        else None
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        ),
        polymer_blueprint=source.polymer_blueprint,
    )


def _site() -> PolymerMicrostateSite:
    return PolymerMicrostateSite(
        build_default_component_library().require("HIS"),
        PolymerChemicalSite.SIDECHAIN,
    )


@pytest.mark.parametrize("donor", ("ND1", "NE2"))
@pytest.mark.parametrize(
    "element", ("FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG", "CA")
)
@pytest.mark.parametrize("observed", (True, False))
def test_typed_donor_selects_unprotonated_n_without_metal_charge_inference(
    histidine_pdb: str, donor: str, element: str, observed: bool
) -> None:
    source = _source(histidine_pdb, donor=donor, element=element, observed=observed)
    site = _site()
    result = PolymerMicrostateContext(source).resolve(
        HIS, site, preferences=pras_microstate_preferences(site)
    )
    assert result.status is MicrostateResolutionStatus.RESOLVED
    assert result.graph is not None
    assert result.graph.atom(donor).hydrogens == 0
    other = "NE2" if donor == "ND1" else "ND1"
    assert result.graph.atom(other).hydrogens == 1
    assert result.graph.protonation_key()[0] == 0
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(METAL, element))
        )
        is None
    )


@pytest.mark.parametrize("observed", (True, False))
def test_explicit_cation_request_cannot_override_physical_donor_requirement(
    histidine_pdb: str, observed: bool, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, observed=observed, element=coordination_metal)
    result = PolymerMicrostateContext(source).resolve(
        HIS,
        _site(),
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    assert result.status is MicrostateResolutionStatus.CONFLICT
    assert result.graph is None


@pytest.mark.parametrize("donor", ("ND1", "NE2"))
@pytest.mark.parametrize("observed", (True, False))
@pytest.mark.parametrize("reverse", (True, False))
def test_actual_placement_publishes_realized_graph_preserving_typed_partner(
    histidine_pdb: str,
    donor: str,
    observed: bool,
    reverse: bool,
    coordination_metal: str,
) -> None:
    source = _source(
        histidine_pdb,
        donor=donor,
        observed=observed,
        reverse=reverse,
        element=coordination_metal,
    )
    context = PolymerMicrostateContext(source)
    site = _site()
    patch = place_polymer_microstate_hydrogens(
        context, HIS, site, preferences=pras_microstate_preferences(site)
    )
    prepared = apply_polymer_microstate_patches(source, (patch,))
    assert (
        prepared.provenance.ingress.observation is source.provenance.ingress.observation
    )
    for index, _position in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        new_index = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            new_index
        ) == source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(
            new_index
        ) == source.geometry.atom_geometry(index)
    metal_index = prepared.constitution.atom_index(AtomRef(METAL, coordination_metal))
    assert prepared.topology.formal_charge(metal_index) is None
    preserved = prepared.topology.bond_between(
        prepared.constitution.atom_index(AtomRef(HIS, donor)), metal_index
    )
    assert preserved is not None
    assert preserved.relationship_type is BondRelationshipType.METAL_COORDINATION
    original = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(HIS, donor)),
        source.constitution.atom_index(AtomRef(METAL, coordination_metal)),
    )
    assert original is not None
    assert preserved.source_metadata == original.source_metadata
    assert preserved.provenance is BondProvenance.SOURCE_EXPLICIT
    target = PolymerMicrostatePreparation(
        PolymerMicrostateContext(prepared), build_default_component_library()
    ).targets_for(HIS)[0]
    assert target.is_realized()
    assert target.resolution.graph is not None
    assert target.resolution.graph.atom(donor).hydrogens == 0
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        roundtrip = read_structure_string(
            write_structure_string(prepared, file_format), file_format
        )
        restored = roundtrip.topology.bond_between(
            roundtrip.constitution.atom_index(AtomRef(HIS, donor)),
            roundtrip.constitution.atom_index(AtomRef(METAL, coordination_metal)),
        )
        assert restored is not None
        assert restored.relationship_type is BondRelationshipType.METAL_COORDINATION


def _with_observed_payload(
    source: ProteinStructure,
    *,
    parent: str | None = None,
    isotope: str = "H",
    charge: tuple[str, int] | None = None,
) -> ProteinStructure:
    residue_index = source.constitution.residue_index(HIS)
    residue = source.constitution.residue_site_at(residue_index)
    geometry = source.residue_geometry(residue_index)
    name = "HD1" if parent == "ND1" else "HE2"
    if parent is not None:
        residue = residue.with_atom_site(AtomSite(name, isotope))
        geometry = geometry.with_atom_geometry(
            name,
            AtomGeometry(
                geometry.position(parent).with_offset(0.4, 0.5, 0.7), 0.6, 17.0
            ),
        )
    charges = () if charge is None else (charge,)
    if parent is not None:
        charges = (*charges, (name, 0))
    changes = source.with_updated_residue_facets(
        residue_site=residue,
        residue_geometry=geometry,
        formal_charge_by_atom_name=charges,
    )
    bonds = list(changes.topology.bonds)
    if parent is not None:
        bonds.append(
            TopologyBond(
                changes.constitution.atom_index(AtomRef(HIS, parent)),
                changes.constitution.atom_index(AtomRef(HIS, name)),
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    reported_order=1,
                    reported_relationship_type=BondRelationshipType.COVALENT,
                ),
            )
        )
    topology = StructureTopology(
        constitution=changes.constitution,
        atom_topologies=changes.topology.atom_topologies,
        bonds=tuple(bonds),
    )
    return _with_topology(changes, topology, observed=True)


def _with_topology(
    source: ProteinStructure, topology: StructureTopology, *, observed: bool = False
) -> ProteinStructure:
    observation = (
        StructureObservation.from_source_facets(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=topology,
        )
        if observed
        else None
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        ),
        polymer_blueprint=source.polymer_blueprint,
    )


@pytest.mark.parametrize("donor", ("ND1", "NE2"))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_observed_donor_h_is_not_erased_to_make_coordination_compatible(
    histidine_pdb: str, donor: str, isotope: str, coordination_metal: str
) -> None:
    source = _with_observed_payload(
        _source(histidine_pdb, donor=donor, element=coordination_metal),
        parent=donor,
        isotope=isotope,
    )
    result = PolymerMicrostateContext(source).resolve(HIS, _site())
    assert result.status is MicrostateResolutionStatus.CONFLICT
    with pytest.raises(ValueError, match="conflict"):
        place_polymer_microstate_hydrogens(
            PolymerMicrostateContext(source), HIS, _site()
        )
    name = "HD1" if donor == "ND1" else "HE2"
    original = source.constitution.resolve_atom_site(AtomRef(HIS, name))
    assert original is not None and original.element == isotope


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_compatible_observed_h_and_explicit_metal_charge_are_preserved(
    histidine_pdb: str, isotope: str, coordination_metal: str
) -> None:
    source = _with_observed_payload(
        _source(histidine_pdb, element=coordination_metal),
        parent="ND1",
        isotope=isotope,
    )
    metal_index = source.constitution.residue_index(METAL)
    source = source.with_updated_residue_facets(
        residue_site=source.constitution.residue_site_at(metal_index),
        residue_geometry=source.residue_geometry(metal_index),
        formal_charge_by_atom_name=(
            (coordination_metal, 3 if coordination_metal == "FE" else 2),
        ),
    )
    source = _with_topology(source, source.topology, observed=True)
    patch = place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source), HIS, _site()
    )
    prepared = apply_polymer_microstate_patches(source, (patch,))
    for ref in (AtomRef(HIS, "HD1"), AtomRef(METAL, coordination_metal)):
        first = source.constitution.atom_index(ref)
        second = prepared.constitution.atom_index(ref)
        assert source.constitution.atom_site_at(
            first
        ) == prepared.constitution.atom_site_at(second)
        assert source.geometry.atom_geometry(first) == prepared.geometry.atom_geometry(
            second
        )
        assert source.topology.formal_charge(first) == prepared.topology.formal_charge(
            second
        )


@pytest.mark.parametrize("charge", (-1, 1))
def test_known_donor_charge_outside_catalog_is_not_neutralized(
    histidine_pdb: str, charge: int, coordination_metal: str
) -> None:
    source = _with_observed_payload(
        _source(histidine_pdb, element=coordination_metal), charge=("NE2", charge)
    )
    result = PolymerMicrostateContext(source).resolve(HIS, _site())
    assert result.status is MicrostateResolutionStatus.CONFLICT
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(HIS, "NE2"))
        )
        == charge
    )


@pytest.mark.parametrize("element", ("O", "C", "S", "MG"))
def test_wrong_or_unadmitted_partner_is_not_a_supported_ring_site(
    histidine_pdb: str, element: str
) -> None:
    source = _source(histidine_pdb, element=element)
    result = PolymerMicrostateContext(source).resolve(HIS, _site())
    assert result.status is MicrostateResolutionStatus.UNSUPPORTED


@pytest.mark.parametrize("observed", (True, False))
def test_double_donor_site_does_not_guess_imidazolate(
    histidine_pdb: str, observed: bool, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    original = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(HIS, "NE2")),
        source.constitution.atom_index(AtomRef(METAL, coordination_metal)),
    )
    assert original is not None
    second = replace(
        original, atom_index_1=source.constitution.atom_index(AtomRef(HIS, "ND1"))
    )
    source = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, second),
        ),
        observed=observed,
    )
    result = PolymerMicrostateContext(source).resolve(HIS, _site())
    assert result.status is MicrostateResolutionStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.UNKNOWN, BondRelationshipType.HYDROGEN_BOND)
)
def test_untyped_or_hbond_relationship_does_not_create_donor_requirements(
    histidine_pdb: str, relationship: BondRelationshipType
) -> None:
    source = _source(histidine_pdb, relationship=relationship, observed=False)
    context = PolymerMicrostateContext(source)
    result = context.resolve(
        HIS, _site(), preferences=pras_microstate_preferences(_site())
    )
    assert result.graph is not None
    assert result.graph.atom("NE2").hydrogens == 1
    if relationship is BondRelationshipType.UNKNOWN:
        with pytest.raises(ValueError, match="unsupported current"):
            place_polymer_microstate_hydrogens(
                context, HIS, _site(), preferences=pras_microstate_preferences(_site())
            )


def test_saved_cation_override_is_rechecked_after_new_coordination(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(
        histidine_pdb,
        relationship=BondRelationshipType.HYDROGEN_BOND,
        observed=False,
        element=coordination_metal,
    )
    patch = place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source),
        HIS,
        _site(),
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    prepared = apply_polymer_microstate_patches(source, (patch,))
    bonds = tuple(
        replace(bond, relationship_type=BondRelationshipType.METAL_COORDINATION)
        if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND
        else bond
        for bond in prepared.topology.bonds
    )
    changed = _with_topology(
        prepared,
        StructureTopology(
            constitution=prepared.constitution,
            atom_topologies=prepared.topology.atom_topologies,
            bonds=bonds,
        ),
    )
    context = PolymerMicrostateContext(changed)
    assert context.resolve(HIS, _site()).status is MicrostateResolutionStatus.CONFLICT
    with pytest.raises(ValueError, match="conflict"):
        PolymerMicrostatePatch(
            changed,
            HIS,
            _site(),
            patch.hydrogens,
            override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
            context=context,
        )


def test_patch_cannot_reuse_context_from_a_different_snapshot(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    context = PolymerMicrostateContext(source)
    patch = place_polymer_microstate_hydrogens(context, HIS, _site())
    changed = source.with_updated_residue_geometries(
        (
            (
                HIS,
                source.residue_geometry(
                    source.constitution.residue_index(HIS)
                ).with_atom_geometry(
                    "NE2",
                    AtomGeometry(Vec3(4.0, 5.0, 6.0)),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="exact source snapshot"):
        PolymerMicrostatePatch(changed, HIS, _site(), patch.hydrogens, context=context)


def test_bridged_ring_n_is_not_silently_treated_as_one_partner(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    index = source.constitution.residue_index(METAL)
    partner = source.constitution.residue_site_at(index).with_atom_site(
        AtomSite("SECOND", coordination_metal)
    )
    source = source.with_updated_residue_facets(
        partner,
        residue_geometry=source.residue_geometry(index).with_atom_geometry(
            "SECOND", AtomGeometry(Vec3(31.0, 30.0, 30.0))
        ),
    )
    original = next(
        bond
        for bond in source.topology.bonds
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    second = TopologyBond(
        source.constitution.atom_index(AtomRef(HIS, "NE2")),
        source.constitution.atom_index(AtomRef(METAL, "SECOND")),
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=original.provenance,
        source_metadata=original.source_metadata,
    )
    source = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, second),
        ),
        observed=True,
    )
    assert (
        PolymerMicrostateContext(source).resolve(HIS, _site()).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


@pytest.mark.parametrize("donor", ("CG", "CD2", "CE1", "CB"))
def test_typed_coordination_on_a_non_donor_is_unsupported(
    histidine_pdb: str, donor: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, donor=donor, element=coordination_metal)
    assert (
        PolymerMicrostateContext(source).resolve(HIS, _site()).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


@pytest.mark.parametrize("order,aromatic", ((1, False), (None, True)))
def test_coordinate_edge_does_not_admit_covalent_semantics(
    histidine_pdb: str, order: int | None, aromatic: bool, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, observed=False, element=coordination_metal)
    bonds = tuple(
        replace(bond, order=order, aromatic=aromatic)
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION
        else bond
        for bond in source.topology.bonds
    )
    changed = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
    )
    assert (
        PolymerMicrostateContext(changed).resolve(HIS, _site()).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


def test_original_integral_order_conflict_is_not_rewritten(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    pair = frozenset(
        (
            source.constitution.atom_index(AtomRef(HIS, "CG")),
            source.constitution.atom_index(AtomRef(HIS, "CD2")),
        )
    )
    bonds = tuple(
        replace(
            bond,
            order=3,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                reported_order=3,
                reported_relationship_type=BondRelationshipType.COVALENT,
            ),
        )
        if frozenset(bond.endpoint_pair()) == pair
        else bond
        for bond in source.topology.bonds
    )
    source = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        observed=True,
    )
    result = PolymerMicrostateContext(source).resolve(HIS, _site())
    assert result.status is MicrostateResolutionStatus.CONFLICT
    assert (
        next(
            bond
            for bond in source.topology.bonds
            if frozenset(bond.endpoint_pair()) == pair
        ).order
        == 3
    )


def test_caller_cannot_claim_realization_with_a_coordination_incompatible_graph(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, observed=False, element=coordination_metal)
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    cation = catalog.resolve(
        MicrostateConstraints(),
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    assert cation.graph is not None
    with pytest.raises(ValueError, match="coordination donor requirements"):
        PolymerMicrostateContext(source).hydrogen_attachments(HIS, _site(), cation)


@pytest.mark.parametrize("donor", ("ND1", "NE2"))
def test_requirements_constrain_override_catalog_before_source_reconciliation(
    donor: str,
) -> None:
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    result = catalog.resolve(
        MicrostateConstraints(),
        requirements=MicrostateConstraints(hydrogens=((donor, 0),)),
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    assert result.status is MicrostateResolutionStatus.CONFLICT


@pytest.mark.parametrize("donor,distance", (("ND1", 1.335), ("NE2", 1.316)))
def test_realized_coordinated_ring_uses_matching_dictionary(
    histidine_pdb: str, donor: str, distance: float, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, donor=donor, element=coordination_metal)
    patch = place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source), HIS, _site()
    )
    prepared = apply_polymer_microstate_patches(source, (patch,))
    dictionary = StructureRestraints(
        prepared, build_default_component_library(), build_default_restraint_library()
    ).for_residue(HIS)
    assert dictionary is not None
    target = dictionary.bond_target("CE1", "NE2")
    assert target is not None
    assert target.target_distance_angstrom == distance


def test_h_removal_rebinds_donor_context_without_original_observation(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, observed=False, element=coordination_metal)
    patch = place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source), HIS, _site()
    )
    prepared = apply_polymer_microstate_patches(source, (patch,))
    stripped = prepared.without_hydrogens()
    context = PolymerMicrostateContext(stripped)
    resolution = context.resolve(HIS, _site())
    assert resolution.graph is not None
    assert resolution.graph.atom("NE2").hydrogens == 0
    assert not context.is_realized(HIS, _site(), resolution)
    restored = apply_polymer_microstate_patches(
        stripped, (place_polymer_microstate_hydrogens(context, HIS, _site()),)
    )
    restored_context = PolymerMicrostateContext(restored)
    assert restored_context.is_realized(
        HIS, _site(), restored_context.resolve(HIS, _site())
    )


@pytest.mark.parametrize("missing", ("CB", "CE1"))
def test_coordination_does_not_bypass_missing_stock_scaffold(
    histidine_pdb: str, coordination_metal: str, missing: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    changed = source.without_atom_refs((AtomRef(HIS, missing),))
    resolution = PolymerMicrostateContext(changed).resolve(
        HIS, _site(), override=MicrostateConstraints(hydrogens=(("NE2", 0),))
    )
    assert resolution.status is MicrostateResolutionStatus.INSUFFICIENT


def test_canonical_name_does_not_admit_wrong_ring_donor_element(
    histidine_pdb: str, coordination_metal: str
) -> None:
    source = _source(histidine_pdb, element=coordination_metal)
    index = source.constitution.residue_index(HIS)
    changed = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index).with_atom_site(AtomSite("NE2", "O")),
        residue_geometry=source.residue_geometry(index),
    )
    assert (
        PolymerMicrostateContext(changed).resolve(HIS, _site()).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_removing_original_conflicting_h_does_not_clear_donor_evidence(
    histidine_pdb: str, coordination_metal: str, isotope: str
) -> None:
    source = _with_observed_payload(
        _source(histidine_pdb, element=coordination_metal),
        parent="NE2",
        isotope=isotope,
    )
    stripped = source.without_hydrogens()
    assert (
        stripped.provenance.ingress.observation is source.provenance.ingress.observation
    )
    resolution = PolymerMicrostateContext(stripped).resolve(HIS, _site())
    assert resolution.status is MicrostateResolutionStatus.CONFLICT
    with pytest.raises(ValueError, match="conflict"):
        place_polymer_microstate_hydrogens(
            PolymerMicrostateContext(stripped), HIS, _site()
        )


@pytest.mark.parametrize("change", ("none", "remove", "rebind", "retype"))
@pytest.mark.parametrize("reverse", (False, True))
def test_original_histidine_partner_survives_current_rebinding(
    histidine_pdb: str, change: str, reverse: bool
) -> None:
    source = _source(histidine_pdb, reverse=reverse)
    index = source.constitution.residue_index(METAL)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index).with_atom_site(
            AtomSite("FE2", "FE")
        ),
        residue_geometry=source.residue_geometry(index).with_atom_geometry(
            "FE2", AtomGeometry(Vec3(31, 30, 30))
        ),
    )
    site = _site()
    selected = PolymerMicrostateContext(source).resolve(HIS, site)
    assert selected.graph is not None
    original = next(
        b
        for b in source.topology.bonds
        if b.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    replacement = (
        TopologyBond(
            source.constitution.atom_index(AtomRef(HIS, "NE2")),
            source.constitution.atom_index(AtomRef(METAL, "FE2")),
            order=None,
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.REPAIR_INFERRED,
        )
        if change == "rebind"
        else replace(original, relationship_type=BondRelationshipType.HYDROGEN_BOND)
        if change == "retype"
        else original
    )
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(
            replacement if b == original else b
            for b in source.topology.bonds
            if change != "remove" or b != original
        ),
    )
    changed = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    context = PolymerMicrostateContext(changed)
    resolution = context.resolve(HIS, site)
    if change == "none":
        assert resolution.graph is not None
    else:
        assert resolution.status is MicrostateResolutionStatus.UNSUPPORTED
        target = next(
            target
            for target in PolymerMicrostatePreparation(
                context, build_default_component_library()
            ).targets_for(HIS)
            if target.site.kind is PolymerChemicalSite.SIDECHAIN
        )
        assert target.resolution.status is MicrostateResolutionStatus.UNSUPPORTED
        assert not target.is_realized()
        with pytest.raises(ValueError, match="original coordination"):
            context.hydrogen_attachments(HIS, site, selected)
        with pytest.raises(ValueError, match="original coordination"):
            PolymerMicrostatePatch(changed, HIS, site, ())
