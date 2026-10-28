"""Explicit sulfur evidence constrains coordinated CYS without a metal default."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import (
    AminoAcidPreparation,
    pras_microstate_preferences,
)
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
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
from protrepair.transformer.completion.hydrogen.microstate import (
    place_polymer_microstate_hydrogens,
)
from protrepair.transformer.polymer_microstate import apply_polymer_microstate_patches

CYS = ResidueId("A", 1)
METAL = ResidueId("Z", 1)


def _source(charge=None, *, element="ZN", isotope=None, observed=True):
    atoms = [
        atom_payload(name, name[0], Vec3(*position))
        for name, position in (
            ("N", (-1.2, 0, 0)),
            ("CA", (0, 0, 0)),
            ("C", (1.4, 0, 0)),
            ("O", (2.1, -0.9, 0.3)),
            ("CB", (0.1, 1.4, 0)),
            ("SG", (0, 2, 1.5)),
        )
    ]
    if isotope is not None:
        atoms.append(atom_payload("HG", isotope, Vec3(0, 2, 2.84)))
    source = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS", residue_id=CYS, atoms=tuple(atoms)
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="ION",
                residue_id=METAL,
                is_hetero=True,
                atoms=(atom_payload("X", element, Vec3(2, 2, 2)),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    constitution = source.constitution
    template = build_default_component_library().get("CYS")
    assert template is not None
    bonds = tuple(
        TopologyBond(
            constitution.atom_index(AtomRef(CYS, bond.atom_name_1)),
            constitution.atom_index(AtomRef(CYS, bond.atom_name_2)),
            order=bond.order,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for bond in template.definition.bonds
        if constitution.resolve_atom_index(AtomRef(CYS, bond.atom_name_1)) is not None
        and constitution.resolve_atom_index(AtomRef(CYS, bond.atom_name_2)) is not None
    )
    sulfur = constitution.atom_index(AtomRef(CYS, "SG"))
    bonds += (
        TopologyBond(
            sulfur,
            constitution.atom_index(AtomRef(METAL, "X")),
            order=None,
            relationship_type=BondRelationshipType.METAL_COORDINATION,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                reported_relationship_type=BondRelationshipType.METAL_COORDINATION,
            ),
        ),
    )
    if isotope is not None:
        bonds += (
            TopologyBond(
                sulfur,
                constitution.atom_index(AtomRef(CYS, "HG")),
                order=1,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        )
    topologies = list(source.topology.atom_topologies)
    topologies[sulfur.value] = AtomTopology(formal_charge=charge)
    topology = StructureTopology(
        constitution=constitution, atom_topologies=tuple(topologies), bonds=bonds
    )
    source = ProteinStructure.from_payload(
        constitution=constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    if observed:
        source = source.with_provenance(
            replace(
                source.provenance,
                ingress=replace(
                    source.provenance.ingress,
                    observation=StructureObservation.from_source_facets(
                        constitution=constitution,
                        geometry=source.geometry,
                        topology=topology,
                    ),
                ),
            )
        )
    return source


def _site():
    template = build_default_component_library().get("CYS")
    assert template is not None
    return PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN)


@pytest.mark.parametrize(
    "charge,observed", ((None, False), (None, True), (0, True), (-1, True))
)
def test_sulfur_selection_obeys_source_charge_without_assigning_metal(charge, observed):
    source = _source(charge, observed=observed)
    site = _site()
    context = PolymerMicrostateContext(source)
    patch = place_polymer_microstate_hydrogens(
        context, CYS, site, preferences=pras_microstate_preferences(site)
    )
    result = apply_polymer_microstate_patches(source, (patch,))
    sulfur = result.constitution.atom_index(AtomRef(CYS, "SG"))
    assert result.topology.formal_charge(sulfur) == (-1 if charge == -1 else 0)
    residue = result.constitution.residue_or_ligand(CYS)
    assert residue is not None
    assert residue.has_atom_site("HG") is (charge != -1)
    assert (
        result.topology.formal_charge(
            result.constitution.atom_index(AtomRef(METAL, "X"))
        )
        is None
    )
    assert (
        result.provenance.ingress.observation is source.provenance.ingress.observation
    )
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        assert result.geometry.atom_geometry(
            result.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(index)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_original_thiol_isotope_is_preserved_and_cannot_support_anionic_sulfur(isotope):
    source = _source(0, isotope=isotope)
    context = PolymerMicrostateContext(source)
    patch = place_polymer_microstate_hydrogens(context, CYS, _site())
    result = apply_polymer_microstate_patches(source, (patch,))
    ref = AtomRef(CYS, "HG")
    assert result.constitution.resolve_atom_site(ref) == AtomSite("HG", isotope)
    assert result.geometry.atom_geometry(
        result.constitution.atom_index(ref)
    ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
    conflict = _source(-1, isotope=isotope)
    assert (
        PolymerMicrostateContext(conflict).resolve(CYS, _site()).status
        is MicrostateResolutionStatus.CONFLICT
    )


def test_unsupported_partner_is_not_a_thiolate_or_hydrogen_default():
    source = _source(element="CA")
    resolution = PolymerMicrostateContext(source).resolve(CYS, _site())
    assert resolution.status is MicrostateResolutionStatus.UNSUPPORTED
    assert resolution.graph is None


def test_uncoordinated_cysteine_keeps_the_existing_fixed_hydrogen_path():
    source = _source()
    topology = StructureTopology(
        constitution=source.constitution,
        atom_topologies=source.topology.atom_topologies,
        bonds=tuple(
            bond
            for bond in source.topology.bonds
            if bond.relationship_type is not BondRelationshipType.METAL_COORDINATION
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=topology,
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        )
    )
    prepared = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    assert not any(
        target.site.kind is PolymerChemicalSite.SIDECHAIN
        for target in prepared.targets_for(CYS)
    )


@pytest.mark.parametrize("missing_partner", (False, True))
def test_original_cysteine_coordination_cannot_disappear(missing_partner):
    source = _source(charge=-1)
    if missing_partner:
        changed = source.without_atom_refs((AtomRef(METAL, "X"),))
    else:
        changed = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=tuple(
                    b
                    for b in source.topology.bonds
                    if b.relationship_type
                    is not BondRelationshipType.METAL_COORDINATION
                ),
            ),
            provenance=source.provenance,
            polymer_blueprint=source.polymer_blueprint,
        )
    resolution = PolymerMicrostateContext(changed).resolve(CYS, _site())
    assert resolution.status is MicrostateResolutionStatus.UNSUPPORTED
    assert any("original coordination" in detail for detail in resolution.details)
