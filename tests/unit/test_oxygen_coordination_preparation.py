"""O coordination validates context without supplying protonation evidence."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import pras_microstate_preferences
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string
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

TARGET = ResidueId("A", 15)
METAL = ResidueId("Z", 1)


def _with_topology(
    source: ProteinStructure,
    bonds: tuple[TopologyBond, ...],
    *,
    observed: bool | None = None,
) -> ProteinStructure:
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=bonds,
    )
    observation = source.provenance.ingress.observation
    if observed is not None:
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


def _source(
    component: str,
    donors: tuple[str, ...],
    *,
    element: str = "CA",
    observed: bool = True,
    carbonyl: bool = False,
    reverse: bool = False,
) -> ProteinStructure:
    filename, first = (
        ("pdb1b9o.ent", 84)
        if carbonyl
        else ("pdb2rnt.ent", 15)
        if component == "ASP"
        else ("pdb3cpa.ent", 72)
    )
    path = Path(__file__).parents[1] / "fixtures/corpus" / filename
    lines = [
        f"{line[:22]}{15 + int(line[22:26]) - first:4d}{line[26:]}"
        for line in path.read_text().splitlines()
        if line.startswith("ATOM")
        and line[21] == "A"
        and int(line[22:26]) in ({first, first + 1} if carbonyl else {first})
        and line[76:78].strip() not in {"H", "D", "T"}
    ]
    if reverse:
        lines.reverse()
    lines.append(
        f"HETATM99999 {element:^4} {element:>3} Z   1    "
        f"{30.0:8.3f}{30.0:8.3f}{30.0:8.3f}{0.7:6.2f}{19.0:6.2f}"
        f"          {element:>2}"
    )
    source = read_structure_string("\n".join(lines), FileFormat.PDB)
    library = build_default_component_library()
    bonds = []
    for residue in source.constitution.residue_slots:
        template = library.get(residue.component_id)
        if template is None:
            continue
        carboxyl_names = (
            {"CG", "OD1", "OD2"}
            if residue.component_id == "ASP"
            else {"CD", "OE1", "OE2"}
            if residue.component_id == "GLU"
            else set()
        )
        for edge in template.definition.bonds:
            first_index = source.constitution.resolve_atom_index(
                AtomRef(residue.residue_id, edge.atom_name_1)
            )
            second_index = source.constitution.resolve_atom_index(
                AtomRef(residue.residue_id, edge.atom_name_2)
            )
            if first_index is not None and second_index is not None:
                bonds.append(
                    TopologyBond(
                        first_index,
                        second_index,
                        order=(
                            None
                            if {edge.atom_name_1, edge.atom_name_2} <= carboxyl_names
                            else edge.order
                        ),
                        provenance=BondProvenance.TEMPLATE_RESOLVED,
                    )
                )
    if carbonyl:
        bonds.append(
            TopologyBond(
                source.constitution.atom_index(AtomRef(TARGET, "C")),
                source.constitution.atom_index(AtomRef(ResidueId("A", 16), "N")),
                provenance=BondProvenance.SEQUENCE_INFERRED,
            )
        )
    bonds.extend(
        TopologyBond(
            source.constitution.atom_index(AtomRef(TARGET, donor)),
            source.constitution.atom_index(AtomRef(METAL, element)),
            order=None,
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                source_id=f"donor-{donor}",
                reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
                reported_distance_angstrom=2.4,
            ),
        )
        for donor in donors
    )
    return _with_topology(source, tuple(bonds), observed=observed)


def _site(source: ProteinStructure, *, carbonyl: bool = False) -> PolymerMicrostateSite:
    residue = source.constitution.residue_or_ligand(TARGET)
    assert residue is not None
    template = build_default_component_library().require(residue.component_id)
    if carbonyl:
        return PolymerMicrostateContext(source).backbone_site(
            TARGET, template, PolymerChemicalSite.BACKBONE_C
        )
    return PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN)


def _prepare(
    source: ProteinStructure,
    *,
    carbonyl: bool = False,
    override: MicrostateConstraints | None = None,
) -> ProteinStructure:
    site = _site(source, carbonyl=carbonyl)
    patch = place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source),
        TARGET,
        site,
        preferences=pras_microstate_preferences(site),
        override=override,
    )
    return apply_polymer_microstate_patches(source, (patch,))


def _assert_source_preserved(
    source: ProteinStructure, prepared: ProteinStructure
) -> None:
    assert (
        prepared.provenance.ingress.observation is source.provenance.ingress.observation
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
        if charge is not None:
            assert prepared.topology.formal_charge(current) == charge
    for bond in source.topology.bonds:
        if bond.relationship_type is not BondRelationshipType.METAL_COORDINATION:
            continue
        current = prepared.topology.bond_between(
            *(
                prepared.constitution.atom_index(source.constitution.atom_ref_at(index))
                for index in bond.endpoint_pair()
            )
        )
        assert current is not None
        assert (
            replace(
                current, atom_index_1=bond.atom_index_1, atom_index_2=bond.atom_index_2
            )
            == bond
        )


@pytest.mark.parametrize(
    "component,donors",
    (
        ("ASP", ("OD1",)),
        ("ASP", ("OD2",)),
        ("ASP", ("OD1", "OD2")),
        ("GLU", ("OE1",)),
        ("GLU", ("OE2",)),
        ("GLU", ("OE1", "OE2")),
    ),
)
@pytest.mark.parametrize(
    "element", ("MG", "CA", "FE", "ZN", "CU", "CO", "NI", "MN", "CD", "HG")
)
@pytest.mark.parametrize("observed", (True, False))
def test_carboxyl_coordination_keeps_preference_without_guessing_metal_charge(
    component: str, donors: tuple[str, ...], element: str, observed: bool
) -> None:
    source = _source(component, donors, element=element, observed=observed)
    site = _site(source)
    result = PolymerMicrostateContext(source).resolve(
        TARGET, site, preferences=pras_microstate_preferences(site)
    )
    assert result.status is MicrostateResolutionStatus.RESOLVED
    assert result.graph is not None
    assert result.graph.protonation_key()[0] == -1
    assert all(atom.hydrogens == 0 for atom in result.graph.atoms)
    prepared = _prepare(source)
    _assert_source_preserved(source, prepared)
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(METAL, element))
        )
        is None
    )
    context = PolymerMicrostateContext(prepared)
    assert context.is_realized(TARGET, site, result)
    assert context.resolve(TARGET, site).graph == result.graph


@pytest.mark.parametrize("observed", (True, False))
@pytest.mark.parametrize("element", ("MG", "CA", "ZN"))
def test_linked_peptide_carbonyl_stays_neutral_double_bonded_and_hydrogen_free(
    observed: bool, element: str
) -> None:
    source = _source("ASP", ("O",), carbonyl=True, observed=observed, element=element)
    site = _site(source, carbonyl=True)
    assert site.linkage is PeptideLinkage.LINKED
    prepared = _prepare(source, carbonyl=True)
    _assert_source_preserved(source, prepared)
    context = PolymerMicrostateContext(prepared)
    result = context.resolve(TARGET, site)
    assert result.graph is not None
    assert result.graph.bond_order("C", "O") == 2
    assert result.graph.atom("O").charge == 0
    assert result.graph.atom("O").hydrogens == 0
    assert context.is_realized(TARGET, site, result)


def _with_source_h(
    source: ProteinStructure, parent: str, isotope: str
) -> ProteinStructure:
    index = source.constitution.residue_index(TARGET)
    residue = source.constitution.residue_site_at(index).with_atom_site(
        AtomSite("HOBS", isotope)
    )
    geometry = source.residue_geometry(index)
    updated = source.with_updated_residue_facets(
        residue_site=residue,
        residue_geometry=geometry.with_atom_geometry(
            "HOBS",
            AtomGeometry(
                geometry.position(parent).with_offset(0.4, 0.5, 0.7), 0.6, 17.0
            ),
        ),
        formal_charge_by_atom_name=(("HOBS", 0),),
    )
    edge = TopologyBond(
        updated.constitution.atom_index(AtomRef(TARGET, parent)),
        updated.constitution.atom_index(AtomRef(TARGET, "HOBS")),
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT,
            reported_order=1,
            reported_relationship_type=BondRelationshipType.COVALENT,
        ),
    )
    return _with_topology(updated, (*updated.topology.bonds, edge), observed=True)


@pytest.mark.parametrize(
    "component,parent", (("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"))
)
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("coordinating_h", (True, False))
def test_source_acid_is_not_deprotonated_by_coordination_or_default_preference(
    component: str, parent: str, isotope: str, coordinating_h: bool
) -> None:
    other = (
        ("OD2" if parent == "OD1" else "OD1")
        if component == "ASP"
        else ("OE2" if parent == "OE1" else "OE1")
    )
    source = _with_source_h(
        _source(component, (parent if coordinating_h else other,)), parent, isotope
    )
    prepared = _prepare(source)
    _assert_source_preserved(source, prepared)
    site = _site(prepared)
    context = PolymerMicrostateContext(prepared)
    result = context.resolve(TARGET, site)
    assert result.graph is not None
    assert result.graph.atom(parent).hydrogens == 1
    assert result.graph.atom(parent).charge == 0
    assert result.graph.atom(other).hydrogens == 0
    assert result.graph.protonation_key()[0] == 0
    assert context.is_realized(TARGET, site, result)
    assert prepared.constitution.resolve_atom_site(AtomRef(TARGET, "HOBS")) == AtomSite(
        "HOBS", isotope
    )


@pytest.mark.parametrize(
    "component,center,first,second",
    (("ASP", "CG", "OD1", "OD2"), ("GLU", "CD", "OE1", "OE2")),
)
@pytest.mark.parametrize("reverse", (True, False))
@pytest.mark.parametrize("charge", (-1, 0))
def test_source_charge_and_both_integral_resonances_survive_publication(
    component: str, center: str, first: str, second: str, reverse: bool, charge: int
) -> None:
    source = _source(component, (first, second), reverse=reverse)
    single, double = (first, second) if reverse else (second, first)
    index = source.constitution.residue_index(TARGET)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=((center, 0), (single, charge), (double, 0)),
    )
    order_by_pair = {frozenset((center, single)): 1, frozenset((center, double)): 2}
    bonds = tuple(
        replace(
            edge,
            order=order_by_pair[pair],
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                reported_order=order_by_pair[pair],
                reported_relationship_type=BondRelationshipType.COVALENT,
            ),
        )
        if (
            pair := frozenset(
                source.constitution.atom_ref_at(i).atom_name
                for i in edge.endpoint_pair()
            )
        )
        in order_by_pair
        else edge
        for edge in source.topology.bonds
    )
    source = _with_topology(source, bonds, observed=True)
    prepared = _prepare(source)
    _assert_source_preserved(source, prepared)
    result = PolymerMicrostateContext(prepared).resolve(TARGET, _site(prepared))
    assert result.graph is not None
    assert result.graph.atom(single).charge == charge
    assert result.graph.atom(single).hydrogens == 1 + charge
    assert result.graph.bond_order(center, single) == 1
    assert result.graph.bond_order(center, double) == 2
    for edge in bonds:
        if edge.provenance is not BondProvenance.SOURCE_EXPLICIT:
            continue
        current = prepared.topology.bond_between(
            *(
                prepared.constitution.atom_index(source.constitution.atom_ref_at(i))
                for i in edge.endpoint_pair()
            )
        )
        assert current is not None and current.order == edge.order
        assert current.source_metadata == edge.source_metadata


@pytest.mark.parametrize("component,donor", (("ASP", "OD1"), ("GLU", "OE2")))
def test_bridging_carboxyl_oxygen_keeps_each_partner_without_covalent_valence(
    component: str, donor: str
) -> None:
    source = _source(component, (donor,))
    index = source.constitution.residue_index(METAL)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index).with_atom_site(
            AtomSite("CA2", "CA")
        ),
        residue_geometry=source.residue_geometry(index).with_atom_geometry(
            "CA2", AtomGeometry(Vec3(31, 30, 30))
        ),
    )
    original = next(
        edge
        for edge in source.topology.bonds
        if edge.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    bridge = TopologyBond(
        source.constitution.atom_index(AtomRef(TARGET, donor)),
        source.constitution.atom_index(AtomRef(METAL, "CA2")),
        order=None,
        relationship_type=original.relationship_type,
        provenance=original.provenance,
        source_metadata=original.source_metadata,
    )
    source = _with_topology(source, (*source.topology.bonds, bridge), observed=True)
    prepared = _prepare(source)
    _assert_source_preserved(source, prepared)
    result = PolymerMicrostateContext(prepared).resolve(TARGET, _site(prepared))
    assert result.graph is not None and result.graph.atom(donor).boundary_order == 0
    assert (
        sum(
            edge.relationship_type is BondRelationshipType.METAL_COORDINATION
            for edge in prepared.topology.bonds
        )
        == 2
    )


@pytest.mark.parametrize("donor", ("CG", "CB"))
@pytest.mark.parametrize("observed", (True, False))
def test_wrong_carboxyl_donor_cannot_be_admitted_by_an_override(
    donor: str, observed: bool
) -> None:
    source = _source("ASP", (donor,), observed=observed)
    result = PolymerMicrostateContext(source).resolve(
        TARGET, _site(source), override=MicrostateConstraints(charges=(("OD2", -1),))
    )
    assert result.status is MicrostateResolutionStatus.UNSUPPORTED
    with pytest.raises(ValueError):
        _prepare(source, override=MicrostateConstraints(charges=(("OD2", -1),)))


@pytest.mark.parametrize("element", ("C", "N", "O", "S", "SE", "NA"))
def test_wrong_or_unadmitted_metal_partner_is_not_silently_ignored(
    element: str,
) -> None:
    source = _source("ASP", ("OD1",), element=element)
    assert (
        PolymerMicrostateContext(source).resolve(TARGET, _site(source)).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


@pytest.mark.parametrize(
    "field,value,original_only",
    (("order", 1, True), ("order", 1, False), ("aromatic", True, False)),
)
def test_invalid_coordination_edge_is_rechecked_in_original_and_current_context(
    field: str, value: int | bool, original_only: bool
) -> None:
    source = _source("ASP", ("OD1",))
    coordinate = next(
        edge
        for edge in source.topology.bonds
        if edge.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    assert coordinate.source_metadata is not None
    changed = replace(
        coordinate,
        **{field: value},
        source_metadata=replace(
            coordinate.source_metadata, reported_order=1 if field == "order" else None
        ),
    )
    bonds = tuple(
        changed if edge == coordinate else edge for edge in source.topology.bonds
    )
    source = _with_topology(source, bonds, observed=original_only)
    if original_only:
        source = _with_topology(
            source,
            tuple(
                edge
                for edge in bonds
                if edge.relationship_type is not BondRelationshipType.METAL_COORDINATION
            ),
        )
    result = PolymerMicrostateContext(source).resolve(
        TARGET, _site(source), override=MicrostateConstraints(charges=(("OD2", -1),))
    )
    assert result.status is MicrostateResolutionStatus.UNSUPPORTED


@pytest.mark.parametrize("charge", (-1, 1))
def test_source_oxygen_h_and_contradictory_charge_are_not_rewritten(
    charge: int,
) -> None:
    source = _with_source_h(_source("ASP", ("OD1",)), "OD1", "D")
    index = source.constitution.residue_index(TARGET)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=(("OD1", charge),),
    )
    source = _with_topology(source, source.topology.bonds, observed=True)
    assert (
        PolymerMicrostateContext(source).resolve(TARGET, _site(source)).status
        is MicrostateResolutionStatus.CONFLICT
    )
    with pytest.raises(ValueError):
        _prepare(source)
    assert source.constitution.resolve_atom_site(AtomRef(TARGET, "HOBS")) == AtomSite(
        "HOBS", "D"
    )


def test_saved_acid_override_survives_new_valid_coordination_but_not_bad_partner() -> (
    None
):
    source = _source("ASP", (), observed=False)
    prepared = _prepare(source, override=MicrostateConstraints(hydrogens=(("OD1", 1),)))
    donor = prepared.constitution.atom_index(AtomRef(TARGET, "OD1"))
    partner = prepared.constitution.atom_index(AtomRef(METAL, "CA"))
    edge = TopologyBond(
        donor,
        partner,
        order=None,
        relationship_type=BondRelationshipType.METAL_COORDINATION,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    changed = _with_topology(prepared, (*prepared.topology.bonds, edge))
    context = PolymerMicrostateContext(changed)
    result = context.resolve(TARGET, _site(changed))
    assert result.graph is not None and result.graph.atom("OD1").hydrogens == 1
    assert result.basis is MicrostateSelectionBasis.OVERRIDE
    assert context.is_realized(TARGET, _site(changed), result)
    replacement = context.resolve(
        TARGET, _site(changed), override=MicrostateConstraints(charges=(("OD2", -1),))
    )
    assert (
        replacement.graph is not None and replacement.graph.protonation_key()[0] == -1
    )
    bad = _with_topology(
        changed,
        tuple(
            replace(
                item,
                atom_index_1=changed.constitution.atom_index(AtomRef(TARGET, "CG")),
                atom_index_2=partner,
            )
            if item == edge
            else item
            for item in changed.topology.bonds
        ),
    )
    assert (
        PolymerMicrostateContext(bad).resolve(TARGET, _site(bad)).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


def test_oxygen_patch_rejects_a_context_bound_to_an_older_snapshot() -> None:
    source = _source("ASP", ("OD1",))
    context = PolymerMicrostateContext(source)
    patch = place_polymer_microstate_hydrogens(
        context,
        TARGET,
        _site(source),
        preferences=pras_microstate_preferences(_site(source)),
    )
    changed = source.with_updated_residue_geometries(
        (
            (
                TARGET,
                source.residue_geometry(
                    source.constitution.residue_index(TARGET)
                ).with_atom_geometry("OD1", AtomGeometry(Vec3(4, 5, 6))),
            ),
        )
    )
    with pytest.raises(ValueError, match="exact source snapshot"):
        PolymerMicrostatePatch(
            changed,
            TARGET,
            _site(changed),
            patch.hydrogens,
            preferences=pras_microstate_preferences(_site(changed)),
            context=context,
        )


@pytest.mark.parametrize(
    "linkage,expected",
    (
        (PeptideLinkage.FREE, MicrostateResolutionStatus.UNSUPPORTED),
        (PeptideLinkage.UNKNOWN, MicrostateResolutionStatus.INSUFFICIENT),
    ),
)
def test_carbonyl_admission_does_not_establish_a_free_or_unknown_terminus(
    linkage: PeptideLinkage,
    expected: MicrostateResolutionStatus,
) -> None:
    source = _source("ASP", ("O",), carbonyl=True)
    site = replace(_site(source, carbonyl=True), linkage=linkage)
    assert PolymerMicrostateContext(source).resolve(TARGET, site).status is expected


def test_coordination_alone_does_not_disambiguate_acid_or_resonance() -> None:
    source = _source("ASP", ("OD1", "OD2"), observed=False)
    result = PolymerMicrostateContext(source).resolve(TARGET, _site(source))
    assert result.status is MicrostateResolutionStatus.AMBIGUOUS
    assert len(result.candidates) == 4
    assert {graph.protonation_key()[0] for graph in result.candidates} == {-1, 0}


@pytest.mark.parametrize("missing", ("OD2", "CB"))
def test_missing_stock_scaffold_is_not_bypassed_by_oxygen_support(missing: str) -> None:
    source = _source("ASP", ("OD1",))
    index = source.constitution.residue_index(TARGET)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index).without_atom_sites((missing,)),
        residue_geometry=source.residue_geometry(index).without_atoms((missing,)),
    )
    result = PolymerMicrostateContext(source).resolve(
        TARGET, _site(source), override=MicrostateConstraints(charges=(("OD2", -1),))
    )
    assert result.status is MicrostateResolutionStatus.INSUFFICIENT


def test_wrong_donor_element_is_not_admitted_by_canonical_atom_name() -> None:
    source = _source("ASP", ("OD1",))
    index = source.constitution.residue_index(TARGET)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index).with_atom_site(AtomSite("OD1", "N")),
        residue_geometry=source.residue_geometry(index),
    )
    assert (
        PolymerMicrostateContext(source).resolve(TARGET, _site(source)).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


@pytest.mark.parametrize(
    "donor,expected",
    (
        ("C", MicrostateResolutionStatus.INSUFFICIENT),
        ("CA", MicrostateResolutionStatus.UNSUPPORTED),
    ),
)
def test_linked_carbonyl_does_not_admit_carbon_or_alpha_anchor_as_donor(
    donor: str,
    expected: MicrostateResolutionStatus,
) -> None:
    source = _source("ASP", (donor,), carbonyl=True)
    context = PolymerMicrostateContext(source)
    site = _site(source, carbonyl=True)
    assert context.resolve(TARGET, site).status is expected
    with pytest.raises(ValueError):
        context.coordination_requirements(
            TARGET, replace(site, linkage=PeptideLinkage.LINKED)
        )


def test_compatible_known_metal_charge_is_preserved_not_repredicted() -> None:
    source = _source("ASP", ("OD1", "OD2"))
    index = source.constitution.residue_index(METAL)
    source = source.with_updated_residue_facets(
        source.constitution.residue_site_at(index),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=(("CA", 2),),
    )
    source = _with_topology(source, source.topology.bonds, observed=True)
    prepared = _prepare(source)
    _assert_source_preserved(source, prepared)
    assert (
        prepared.topology.formal_charge(
            prepared.constitution.atom_index(AtomRef(METAL, "CA"))
        )
        == 2
    )


def test_carboxyl_and_backbone_coordination_remain_independent_decision_units() -> None:
    source = _source("ASP", ("OD1", "OD2", "O"), carbonyl=True)
    sidechain = _prepare(source)
    prepared = _prepare(sidechain, carbonyl=True)
    _assert_source_preserved(source, prepared)
    context = PolymerMicrostateContext(prepared)
    for carbonyl in (False, True):
        site = _site(prepared, carbonyl=carbonyl)
        resolution = context.resolve(TARGET, site)
        assert resolution.graph is not None
        assert context.is_realized(TARGET, site, resolution)


def test_source_acid_h_conflicting_with_declared_double_bond_is_not_suppressed() -> (
    None
):
    source = _with_source_h(_source("ASP", ("OD1",)), "OD1", "T")
    pair = frozenset(
        (
            source.constitution.atom_index(AtomRef(TARGET, "CG")),
            source.constitution.atom_index(AtomRef(TARGET, "OD1")),
        )
    )
    bonds = tuple(
        replace(
            edge,
            order=2,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                reported_order=2,
                reported_relationship_type=BondRelationshipType.COVALENT,
            ),
        )
        if frozenset(edge.endpoint_pair()) == pair
        else edge
        for edge in source.topology.bonds
    )
    source = _with_topology(source, bonds, observed=True)
    result = PolymerMicrostateContext(source).resolve(TARGET, _site(source))
    assert result.status is MicrostateResolutionStatus.CONFLICT
    with pytest.raises(ValueError):
        _prepare(source)
