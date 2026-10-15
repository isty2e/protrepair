"""Environment selection, chemistry authority and actual H realization."""

from dataclasses import replace
from math import cos, radians, sin

import pytest

from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateSelectionBasis,
)
from protrepair.chemistry.radii import (
    RDKIT_PERIODIC_TABLE_COVALENT_RADII_ANGSTROM,
    covalent_radius_angstrom,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens


def _point(name: str, distance: float = 0.0, bend: float = 0.0) -> Vec3:
    angle = radians({"ND1": 126, "NE2": -18}[name])
    return Vec3(
        1.15 * cos(angle) + distance * cos(angle + radians(bend)),
        1.15 * sin(angle) + distance * sin(angle + radians(bend)),
        0.0,
    )


def _source(
    *,
    contacts: tuple[tuple[str, float, float, str], ...] = (),
    source_h: str | None = None,
    isotope: str = "H",
    reverse: bool = False,
):
    atoms = [
        ("N", "N", Vec3(-3.6, 1.1, 0.8)),
        ("CA", "C", Vec3(-3.6, -0.32, 0.8)),
        ("C", "C", Vec3(-4.8, -1.1, 0.6)),
        ("O", "O", Vec3(-5.8, -0.5, 0.6)),
        ("OXT", "O", Vec3(-4.7, -2.3, 0.6)),
        ("CB", "C", Vec3(-2.52, -0.82, 0.0)),
    ]
    for name, angle in zip(
        ("CG", "ND1", "CE1", "NE2", "CD2"), (198, 126, 54, -18, -90), strict=True
    ):
        atoms.append(
            (
                name,
                "N" if name in {"ND1", "NE2"} else "C",
                Vec3(1.15 * cos(radians(angle)), 1.15 * sin(radians(angle)), 0.0),
            )
        )
    if source_h is not None:
        atoms.append(
            ("HD1" if source_h == "ND1" else "HE2", isotope, _point(source_h, 1.01))
        )
    entries = [(name, element, p, "HIS", "A", 1, "") for name, element, p in atoms]
    for number, (nitrogen, distance, bend, motif) in enumerate(contacts, 1):
        position = _point(nitrogen, distance, bend)
        outward = _point(nitrogen, distance + 1.23, bend)
        if motif == "carbonyl":
            entries.extend(
                (
                    ("O", "O", position, "GLY", "B", number, ""),
                    ("C", "C", outward, "GLY", "B", number, ""),
                )
            )
        elif motif == "amide":
            entries.extend(
                (
                    ("N", "N", position, "GLY", "B", number, ""),
                    ("CA", "C", outward, "GLY", "B", number, ""),
                )
            )
        elif motif == "charged_oxygen":
            entries.extend(
                (
                    ("O", "O", position, "GLY", "B", number, "1+"),
                    ("C", "C", outward, "GLY", "B", number, ""),
                )
            )
        elif motif == "donor":
            entries.extend(
                (
                    ("N", "N", position, "GLY", "B", number, ""),
                    (
                        "H",
                        "H",
                        _point(nitrogen, distance - 1.01, bend),
                        "GLY",
                        "B",
                        number,
                        "",
                    ),
                    ("CA", "C", outward, "GLY", "B", number, ""),
                )
            )
        elif motif == "unknown":
            entries.append(("N1", "N", position, "UNX", "B", number, ""))
        elif motif == "orphan_oxygen":
            entries.append(("O1", "O", position, "UNX", "B", number, "1-"))
        elif motif in {"imine", "iminium"}:
            entries.extend(
                (
                    (
                        "N1",
                        "N",
                        position,
                        "UNX",
                        "B",
                        number,
                        "1+" if motif == "iminium" else "",
                    ),
                    ("C1", "C", outward, "UNX", "B", number, ""),
                    ("C2", "C", outward.with_offset(0, 0, 1.5), "UNX", "B", number, ""),
                )
            )
        elif motif == "carbon":
            entries.append(("CA", "C", position, "GLY", "B", number, ""))
        else:
            raise AssertionError(motif)
    if reverse:
        entries.reverse()
    lines = [
        f"ATOM  {serial:5d} {name:^4s} {component:3s} {chain}{number:4d}    "
        f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}{charge:>2s}"
        for serial, (name, element, p, component, chain, number, charge) in enumerate(
            entries, 1
        )
    ]
    return read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)


def _preparation(source, requests=None):
    return PolymerMicrostatePreparation(
        PolymerMicrostateContext(source),
        build_standard_component_library(),
        requests=requests,
    )


def _target(preparation):
    return next(
        t
        for t in preparation.targets_for(ResidueId("A", 1))
        if t.site.kind is PolymerChemicalSite.SIDECHAIN
    )


def _selected(preparation):
    graph = _target(preparation).resolution.graph
    assert graph is not None
    return tuple(graph.atom(name).hydrogens for name in ("ND1", "NE2"))


def _move(source, ref, position):
    geometry = source.residue_geometry(
        source.constitution.residue_index(ref.residue_id)
    )
    return source.with_updated_residue_geometries(
        (
            (
                ref.residue_id,
                geometry.with_atom_geometry(
                    ref.atom_name,
                    replace(geometry.atom_geometry(ref.atom_name), position=position),
                ),
            ),
        )
    )


def _with_bonds(source, bonds):
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, *bonds),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize("nitrogen,expected", (("ND1", (1, 0)), ("NE2", (0, 1))))
def test_local_carbonyl_selects_donating_neutral_tautomer(nitrogen, expected):
    source = _source(contacts=((nitrogen, 2.8, 0, "carbonyl"),))
    assert _selected(_preparation(source)) == expected
    result = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset({ResidueId("A", 1)}),
    )
    target = _target(_preparation(result.structure))
    assert target.is_realized()
    assert _selected(_preparation(result.structure)) == expected
    assert not result.issues
    for index, position in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        assert (
            result.structure.geometry.position(
                result.structure.constitution.atom_index(ref)
            )
            == position
        )


@pytest.mark.parametrize(
    "motif", ("amide", "charged_oxygen", "unknown", "orphan_oxygen")
)
def test_nonacceptor_contact_does_not_fabricate_hydrogen_bond(motif):
    assert _selected(_preparation(_source(contacts=(("ND1", 2.8, 0, motif),)))) == (
        0,
        1,
    )


@pytest.mark.parametrize("motif,expected", (("imine", (1, 0)), ("iminium", (0, 1))))
def test_imine_acceptor_requires_nonpositive_charge(motif, expected):
    source = _source(contacts=(("ND1", 2.8, 0, motif),))
    n, c1, c2 = (
        source.constitution.atom_index(AtomRef(ResidueId("B", 1), name))
        for name in ("N1", "C1", "C2")
    )
    bonded = _with_bonds(source, (TopologyBond(n, c1, order=2), TopologyBond(n, c2)))
    assert _selected(_preparation(bonded)) == expected


@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.METAL_COORDINATION, BondRelationshipType.UNKNOWN),
)
def test_coordination_or_unknown_connection_disqualifies_acceptor(relationship):
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    assert _selected(_preparation(source)) == (1, 0)
    oxygen = source.constitution.atom_index(AtomRef(ResidueId("B", 1), "O"))
    other = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "O"))
    connected = _with_bonds(
        source, (TopologyBond(oxygen, other, relationship_type=relationship),)
    )
    assert _selected(_preparation(connected)) == (0, 1)


@pytest.mark.parametrize("distance,bend", ((3.7, 0), (2.8, 90), (1.4, 0)))
def test_far_bent_or_compressed_contact_is_not_hbond_support(distance, bend):
    assert _selected(
        _preparation(_source(contacts=(("ND1", distance, bend, "carbonyl"),)))
    ) == (0, 1)


def test_actual_donor_h_selects_accepting_ring_nitrogen():
    assert _selected(_preparation(_source(contacts=(("NE2", 2.8, 0, "donor"),)))) == (
        1,
        0,
    )


def test_unobserved_donor_orientation_is_not_guessed():
    assert _selected(_preparation(_source(contacts=(("NE2", 2.8, 0, "amide"),)))) == (
        0,
        1,
    )


def test_epsilon_tie_fallback_and_short_contact_priority():
    assert _selected(_preparation(_source())) == (0, 1)
    source = _source(contacts=(("NE2", 1.15, 0, "carbon"),))
    assert _selected(_preparation(source)) == (1, 0)


@pytest.mark.parametrize(
    "element", (*RDKIT_PERIODIC_TABLE_COVALENT_RADII_ANGSTROM, "D", "T")
)
@pytest.mark.parametrize("offset,expected", ((-0.02, (1, 0)), (0.02, (0, 1))))
def test_crowding_uses_covalent_radii_for_every_element_and_isotope(
    element, offset, expected
):
    separation = (
        covalent_radius_angstrom("H") + covalent_radius_angstrom(element) + 0.45
    )
    source = _source(contacts=(("NE2", 1.15, 0, "carbon"),))
    residue_id = ResidueId("B", 1)
    residue = source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    geometry = source.residue_geometry(source.constitution.residue_index(residue_id))
    source = source.with_updated_residue_facets(
        replace(residue, atom_sites=(replace(residue.atom_sites[0], element=element),)),
        residue_geometry=geometry,
    )
    source = _move(
        source,
        AtomRef(residue_id, "CA"),
        _point("NE2", 1.01 + separation + offset),
    )
    assert _selected(_preparation(source)) == expected


@pytest.mark.parametrize("element", ("X", "XX"))
def test_unknown_radius_neighbor_does_not_fabricate_crowding(element):
    source = _source(contacts=(("NE2", 1.15, 0, "carbon"),))
    residue_id = ResidueId("B", 1)
    residue = source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    source = source.with_updated_residue_facets(
        replace(residue, atom_sites=(replace(residue.atom_sites[0], element=element),)),
        residue_geometry=source.residue_geometry(
            source.constitution.residue_index(residue_id)
        ),
    )
    assert _selected(_preparation(source)) == (0, 1)


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_source_h_isotope_wins_over_environment(isotope):
    source = _source(
        contacts=(("ND1", 2.8, 0, "carbonyl"),), source_h="NE2", isotope=isotope
    )
    assert _selected(_preparation(source)) == (0, 1)
    result = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset({ResidueId("A", 1)}),
    )
    ref = AtomRef(ResidueId("A", 1), "HE2")
    before = source.constitution.atom_index(ref)
    after = result.structure.constitution.atom_index(ref)
    assert result.structure.constitution.atom_site_at(after).element == isotope
    assert result.structure.geometry.position(after) == source.geometry.position(before)


def test_explicit_cation_request_wins_and_environment_stays_a_default():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    request = {
        (ResidueId("A", 1), PolymerChemicalSite.SIDECHAIN): MicrostateConstraints(
            hydrogens=(("ND1", 1), ("NE2", 1))
        )
    }
    preparation = _preparation(source, request)
    assert _selected(preparation) == (1, 1)
    assert _target(preparation).resolution.basis is MicrostateSelectionBasis.OVERRIDE
    assert (
        _target(_preparation(source)).resolution.basis
        is MicrostateSelectionBasis.DEFAULT
    )


def test_selection_is_independent_of_atom_and_residue_order():
    contacts = (("ND1", 2.8, 0, "carbonyl"), ("NE2", 4.0, 0, "amide"))
    assert (
        _selected(_preparation(_source(contacts=contacts)))
        == _selected(_preparation(_source(contacts=contacts, reverse=True)))
        == (1, 0)
    )


def test_coordinate_change_invalidates_environment_but_not_applied_choice():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    before = _preparation(source)
    assert _selected(before) == (1, 0)
    ref = AtomRef(ResidueId("B", 1), "O")
    moved = _move(source, ref, Vec3(20, 20, 20))
    assert not before.matches_chemistry(moved)
    assert _selected(_preparation(moved)) == (0, 1)
    prepared = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset({ResidueId("A", 1)}),
    ).structure
    moved_prepared = _move(prepared, ref, Vec3(20, 20, 20))
    stripped = moved_prepared.without_hydrogens()
    assert _selected(_preparation(stripped)) == (1, 0)
    request = {(ResidueId("A", 1), PolymerChemicalSite.SIDECHAIN): None}
    assert _selected(_preparation(stripped, request)) == (0, 1)


def test_observation_rebinding_tracks_histidine_environment_and_applied_choice():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    observation = StructureChemistryObservation(source)
    original = observation.hydrogen_expectation_model
    assert _selected(original.polymer_preparation) == (1, 0)

    ref = AtomRef(ResidueId("B", 1), "O")
    moved = _move(source, ref, Vec3(20, 20, 20))
    rebound = observation.for_structure(moved).hydrogen_expectation_model
    assert rebound is not original
    assert _selected(rebound.polymer_preparation) == (0, 1)
    assert _selected(observation.hydrogen_expectation_model.polymer_preparation) == (
        1,
        0,
    )

    prepared = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset({ResidueId("A", 1)}),
    ).structure
    prepared_observation = observation.for_structure(prepared)
    prepared_model = prepared_observation.hydrogen_expectation_model
    assert prepared_model is not original
    stripped = _move(prepared, ref, Vec3(20, 20, 20)).without_hydrogens()
    stripped_model = prepared_observation.for_structure(
        stripped
    ).hydrogen_expectation_model
    assert stripped_model is not prepared_model
    assert _selected(stripped_model.polymer_preparation) == (1, 0)


def test_written_atoms_preserve_tautomer_on_reingress():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    prepared = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset({ResidueId("A", 1)}),
    ).structure
    reread = read_structure_string(
        write_structure_string(prepared, FileFormat.MMCIF), FileFormat.MMCIF
    )
    assert _selected(_preparation(reread)) == (1, 0)


def test_degenerate_ring_frame_abstains_without_warning():
    source = _source()
    cg = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "CG"))
    degenerate = _move(
        source, AtomRef(ResidueId("A", 1), "ND1"), source.geometry.position(cg)
    )
    assert _selected(_preparation(degenerate)) == (0, 1)


@pytest.mark.parametrize(
    "distance,expected",
    ((2.60, (0, 1)), (2.62, (1, 0)), (3.49, (1, 0)), (3.51, (0, 1))),
)
def test_contact_distance_boundaries(distance, expected):
    assert (
        _selected(_preparation(_source(contacts=(("ND1", distance, 0, "carbonyl"),))))
        == expected
    )


def test_rigid_translation_does_not_change_tautomer():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    transformed = source.with_updated_residue_geometries(
        tuple(
            (
                residue.residue_id,
                source.residue_geometry(
                    source.constitution.residue_index(residue.residue_id)
                ).with_atom_geometries(
                    tuple(
                        (
                            atom.name,
                            source.geometry.atom_geometry(
                                source.constitution.atom_index(
                                    AtomRef(residue.residue_id, atom.name)
                                )
                            ).with_position(
                                source.geometry.position(
                                    source.constitution.atom_index(
                                        AtomRef(residue.residue_id, atom.name)
                                    )
                                ).with_offset(13, -7, 2)
                            ),
                        )
                        for atom in residue.atom_sites
                    )
                ),
            )
            for chain in source.constitution.chains
            for residue in chain.residues
        )
    )
    assert (
        _selected(_preparation(transformed))
        == _selected(_preparation(source))
        == (1, 0)
    )


def test_custom_histidine_template_is_not_silently_replaced():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),))
    library = build_standard_component_library()
    changed = replace(library.require("HIS"), preferred_atom_order=("ND1", "NE2"))
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), library.with_template(changed)
    )
    assert _target(preparation).resolution.graph is None
    assert not _target(preparation).is_realized()


def test_reset_default_never_erases_original_histidine_h():
    source = _source(contacts=(("ND1", 2.8, 0, "carbonyl"),), source_h="NE2")
    preparation = _preparation(
        source, {(ResidueId("A", 1), PolymerChemicalSite.SIDECHAIN): None}
    )
    assert _selected(preparation) == (0, 1)


def test_unresolved_histidines_do_not_select_each_other():
    source = _source()
    lines = [
        line
        for line in write_structure_string(source, FileFormat.PDB).splitlines()
        if line.startswith("ATOM")
    ]
    target = _point("ND1", 2.8)
    peer_nitrogen = _point("ND1")
    for serial, line in enumerate(tuple(lines), 101):
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        x = -x + target.x + peer_nitrogen.x
        y = -y + target.y + peer_nitrogen.y
        lines.append(
            f"{line[:6]}{serial:5d}{line[11:21]}B{line[22:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        )
    pair = read_structure_string("\n".join(lines) + "\nEND\n", FileFormat.PDB)
    assert _selected(_preparation(pair)) == (0, 1)
    assert (
        _target(_preparation(pair)).resolution.basis is MicrostateSelectionBasis.DEFAULT
    )
