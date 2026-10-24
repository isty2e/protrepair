from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.faspr_output import relabel_faspr_output
from tests.unit.test_faspr_backend import (
    build_heavy_only_packed_structure,
    build_hydrogenated_test_structure,
    copy_template_faspr_executable,
)

from protrepair.chemistry.microstate.graph import MicrostateAtom, MicrostateGraph
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateSelectionBasis,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, write_structure_string
from protrepair.io.pdb_projection import pdb_without_conect
from protrepair.structure import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.packing import PackingPlan, PackingScope, PackingSpec
from protrepair.transformer.packing.faspr.backend import (
    FasprPackingBackend,
    PackingBackendError,
    PackingBackendExecutionError,
)


def _pack(
    tmp_path: Path,
    source: ProteinStructure,
    native: ProteinStructure,
    spec: PackingSpec | None = None,
    *,
    include_native_connections: bool = False,
) -> ProteinStructure:
    executable = copy_template_faspr_executable(tmp_path)
    (tmp_path / "dun2010bbdep.bin").write_text("stub", encoding="utf-8")
    (tmp_path / "packed-output.pdb").write_text(
        relabel_faspr_output(
            write_structure_string(native, FileFormat.PDB)
            if include_native_connections
            else pdb_without_conect(native),
            source,
        ),
        encoding="utf-8",
    )
    plan = PackingPlan.from_inputs(
        source, PackingSpec(backend_name="faspr") if spec is None else spec
    )
    return FasprPackingBackend(executable_path=executable).pack(plan).packed_structure


def _with_charge(source: ProteinStructure, charge: int | None) -> ProteinStructure:
    atoms = list(source.topology.atom_topologies)
    index = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "N"))
    atoms[index.value] = None if charge is None else AtomTopology(formal_charge=charge)
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=tuple(atoms),
            bonds=source.topology.bonds,
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _with_sites(
    source: ProteinStructure,
    *,
    component: str | None = None,
    remove: frozenset[str] = frozenset(),
    additions: tuple[tuple[AtomSite, AtomGeometry], ...] = (),
    reverse: bool = False,
) -> ProteinStructure:
    residue = source.constitution.chain("A").residues[0]
    sites = [site for site in residue.atom_sites if site.name not in remove]
    sites.extend(site for site, _ in additions)
    if reverse:
        sites.reverse()
    updated = ResidueSite(
        component_id=component or residue.component_id,
        residue_id=residue.residue_id,
        atom_sites=tuple(sites),
        is_hetero=residue.is_hetero,
    )
    chain = source.constitution.chain("A")
    constitution = source.constitution.with_chains(
        (chain.with_residues((updated, *chain.residues[1:])),)
    )
    geometry_by_ref = {
        source.constitution.atom_ref_at(AtomIndex(index)): geometry
        for index, geometry in enumerate(source.geometry.atom_geometries)
    }
    geometry_by_ref.update(
        (AtomRef(residue.residue_id, site.name), geometry)
        for site, geometry in additions
    )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution,
            atom_geometries=tuple(
                geometry_by_ref[constitution.atom_ref_at(AtomIndex(index))]
                for index in range(len(constitution.atom_slots))
            ),
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(
                source.topology.atom_topologies[old_index.value]
                if old_index is not None
                else None
                for index in range(len(constitution.atom_slots))
                for old_index in (
                    source.constitution.resolve_atom_index(
                        constitution.atom_ref_at(AtomIndex(index))
                    ),
                )
            ),
            bonds=source.topology.bonds_for_constitution(
                source_constitution=source.constitution,
                target_constitution=constitution,
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _saved_override(source: ProteinStructure) -> ProteinStructure:
    observation = StructureObservation.from_source_facets(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=source.topology,
    )
    choice = AppliedMicrostateChoice(
        ResidueId("A", 1),
        "ALA",
        MicrostateGraph((MicrostateAtom("N", "N", 1, 3, 1),), ()),
        MicrostateSelectionBasis.OVERRIDE,
    )
    return _with_charge(source, 1).with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
            microstate_choices=(choice,),
        )
    )


@pytest.mark.parametrize("charge", (None, 0, 1, -1))
def test_native_charge_omission_preserves_current_chemistry_and_hydrogens(
    tmp_path: Path, charge: int | None
) -> None:
    source = _with_charge(build_hydrogenated_test_structure(), charge)
    result = _pack(tmp_path, source, build_heavy_only_packed_structure())

    assert result.constitution == source.constitution
    assert result.geometry == source.geometry
    assert result.topology == source.topology
    assert result.provenance is source.provenance


def test_native_charge_is_not_a_protonation_override(tmp_path: Path) -> None:
    source = build_hydrogenated_test_structure()
    native = _with_charge(build_heavy_only_packed_structure(), 1)
    result = _pack(tmp_path, source, native)

    assert result.topology == source.topology
    assert result.constitution == source.constitution


def test_packing_preserves_scalar_metadata_and_exact_backbone(tmp_path: Path) -> None:
    source = build_hydrogenated_test_structure()
    geometries = tuple(
        AtomGeometry(
            position=g.position.with_offset(0.000321, -0.000234, 0.000123),
            occupancy=0.75,
            b_factor=12.345678,
            altloc="A",
        )
        for g in source.geometry.atom_geometries
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=StructureGeometry(
            constitution=source.constitution, atom_geometries=geometries
        ),
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    native = build_heavy_only_packed_structure(
        first_residue_atom_positions={
            name: source.geometry.atom_geometry(
                source.constitution.atom_index(AtomRef(ResidueId("A", 1), name))
            ).position
            for name in ("N", "CA", "C", "O", "CB")
        }
    )
    result = _pack(tmp_path, source, native)

    for name in ("N", "CA", "C", "O", "H", "HA", "HB1"):
        ref = AtomRef(ResidueId("A", 1), name)
        assert result.geometry.atom_geometry(
            result.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
    cb = result.geometry.atom_geometry(
        result.constitution.atom_index(AtomRef(ResidueId("A", 1), "CB"))
    )
    assert cb.occupancy == 0.75
    assert cb.b_factor == 12.345678
    assert cb.altloc == "A"


@pytest.mark.parametrize("scope", (PackingScope.FULL, PackingScope.LOCAL))
def test_explicit_mutable_set_preserves_exact_outside_geometry(
    tmp_path: Path, scope: PackingScope
) -> None:
    source = build_hydrogenated_test_structure(residue_count=2)
    geometries = tuple(
        g.with_position(g.position.with_offset(0.000321, 0.000234, 0.000123))
        for g in source.geometry.atom_geometries
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=StructureGeometry(
            constitution=source.constitution, atom_geometries=geometries
        ),
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    native = build_heavy_only_packed_structure(
        residue_count=2,
        atom_positions_by_residue_seq_num={
            number: {
                name: source.geometry.atom_geometry(
                    source.constitution.atom_index(
                        AtomRef(ResidueId("A", number), name)
                    )
                ).position
                for name in ("N", "CA", "C", "O", "CB")
            }
            for number in (1, 2)
        },
    )
    result = _pack(
        tmp_path,
        source,
        native,
        PackingSpec(
            backend_name="faspr",
            scope=scope,
            mutable_residue_ids=(ResidueId("A", 1),),
        ),
    )

    for i in range(len(source.constitution.atom_slots)):
        ref = source.constitution.atom_ref_at(AtomIndex(i))
        if ref.residue_id == ResidueId("A", 2):
            assert result.geometry.atom_geometry(
                result.constitution.atom_index(ref)
            ) == source.geometry.atom_geometry(AtomIndex(i))


@pytest.mark.parametrize("atom_name", ("N", "CA", "C", "O", "CB"))
def test_native_cannot_move_frozen_atoms(tmp_path: Path, atom_name: str) -> None:
    source = build_hydrogenated_test_structure()
    native = build_heavy_only_packed_structure(
        first_residue_atom_positions={atom_name: Vec3(9, 9, 9)}
    )
    with pytest.raises(PackingBackendExecutionError, match="protected atom"):
        _pack(
            tmp_path,
            source,
            native,
            PackingSpec(backend_name="faspr", frozen_residue_ids=(ResidueId("A", 1),)),
        )


def test_native_cannot_move_backbone_of_mutable_residue(tmp_path: Path) -> None:
    source = build_hydrogenated_test_structure()
    native = build_heavy_only_packed_structure(
        first_residue_atom_positions={"CA": Vec3(9, 9, 9)}
    )
    with pytest.raises(PackingBackendExecutionError, match="protected atom"):
        _pack(tmp_path, source, native)


def test_current_override_survives_without_replaying_original_charge(
    tmp_path: Path,
) -> None:
    source = _saved_override(_with_charge(build_hydrogenated_test_structure(), 0))
    result = _pack(tmp_path, source, build_heavy_only_packed_structure())
    index = result.constitution.atom_index(AtomRef(ResidueId("A", 1), "N"))

    assert result.topology.formal_charge(index) == 1
    assert result.provenance is source.provenance
    observation = result.provenance.ingress.observation
    assert observation is not None
    assert observation.topology.formal_charge(index) == 0


def test_requested_mutation_invalidates_old_charge_h_and_saved_choice(
    tmp_path: Path,
) -> None:
    source = _saved_override(build_hydrogenated_test_structure())
    native = _with_sites(
        build_heavy_only_packed_structure(), component="GLY", remove=frozenset({"CB"})
    )
    result = _pack(
        tmp_path, source, native, PackingSpec(backend_name="faspr", target_sequence="G")
    )

    residue = result.constitution.chain("A").residues[0]
    assert residue.component_id == "GLY"
    assert all(not site.is_hydrogen() for site in residue.atom_sites)
    assert all(charge is None for charge in result.topology.atom_topologies)
    assert result.provenance.microstate_choices == ()
    assert result.provenance.ingress is source.provenance.ingress
    assert (
        result.topology.bond_between(
            result.constitution.atom_index(AtomRef(residue.residue_id, "N")),
            result.constitution.atom_index(AtomRef(residue.residue_id, "CA")),
        )
        is not None
    )


def test_unrequested_component_change_is_rejected(tmp_path: Path) -> None:
    native = _with_sites(
        build_heavy_only_packed_structure(), component="GLY", remove=frozenset({"CB"})
    )
    with pytest.raises(PackingBackendExecutionError, match="requested sequence"):
        _pack(tmp_path, build_hydrogenated_test_structure(), native)


@pytest.mark.parametrize("atom_name", ("N", "CA", "C", "O", "CB"))
def test_native_cannot_omit_existing_required_heavy_atom(
    tmp_path: Path, atom_name: str
) -> None:
    native = _with_sites(
        build_heavy_only_packed_structure(), remove=frozenset({atom_name})
    )
    with pytest.raises(PackingBackendExecutionError, match="omitted required atom"):
        _pack(tmp_path, build_hydrogenated_test_structure(), native)


def test_native_atom_reordering_preserves_chemical_identity(tmp_path: Path) -> None:
    source = build_hydrogenated_test_structure()
    result = _pack(
        tmp_path, source, _with_sites(build_heavy_only_packed_structure(), reverse=True)
    )
    assert result.constitution == source.constitution
    assert result.topology == source.topology
    assert result.geometry == source.geometry


def test_native_omission_preserves_current_oxt_and_bond(tmp_path: Path) -> None:
    source = _with_sites(
        build_hydrogenated_test_structure(),
        additions=((AtomSite("OXT", "O"), AtomGeometry(Vec3(2, 2, 2))),),
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(ResidueId("A", 1), "C")),
        source.constitution.atom_index(AtomRef(ResidueId("A", 1), "OXT")),
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, bond),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    result = _pack(tmp_path, source, build_heavy_only_packed_structure())
    assert result.constitution == source.constitution
    assert result.geometry == source.geometry
    assert result.topology == source.topology


@pytest.mark.parametrize("element", ("D", "T"))
def test_native_projection_preserves_isotope_hydrogen_custody(
    tmp_path: Path, element: str
) -> None:
    original = build_hydrogenated_test_structure()
    hydrogen_ref = AtomRef(ResidueId("A", 1), "H")
    geometry = original.geometry.atom_geometry(
        original.constitution.atom_index(hydrogen_ref)
    )
    source = _with_sites(
        original,
        remove=frozenset({"H"}),
        additions=((AtomSite("H", element), geometry),),
    )
    result = _pack(tmp_path, source, build_heavy_only_packed_structure())
    assert result.constitution == source.constitution
    assert result.geometry == source.geometry
    assert result.topology == source.topology


def test_native_cannot_supply_unrequested_hydrogen_sites(tmp_path: Path) -> None:
    with pytest.raises(PackingBackendExecutionError, match="hydrogen"):
        _pack(
            tmp_path,
            build_hydrogenated_test_structure(),
            build_hydrogenated_test_structure(),
        )


@pytest.mark.parametrize("frozen", (False, True))
def test_native_sidechain_completion_respects_plan_authority(
    tmp_path: Path, frozen: bool
) -> None:
    source = _with_sites(
        build_hydrogenated_test_structure(), remove=frozenset({"CB", "HB1"})
    )
    native = build_heavy_only_packed_structure()
    if frozen:
        with pytest.raises(PackingBackendError, match="fixed residue A:1.*CB"):
            _pack(
                tmp_path,
                source,
                native,
                PackingSpec(
                    backend_name="faspr",
                    frozen_residue_ids=(ResidueId("A", 1),),
                ),
            )
        return
    result = _pack(tmp_path, source, native)
    ca = result.constitution.atom_index(AtomRef(ResidueId("A", 1), "CA"))
    cb = result.constitution.atom_index(AtomRef(ResidueId("A", 1), "CB"))
    bond = result.topology.bond_between(ca, cb)
    assert bond is not None and bond.order == 1
    assert bond.provenance is BondProvenance.TEMPLATE_RESOLVED
    assert not any(site.is_hydrogen() for site in result.constitution.atom_slots)


def test_complete_fixed_residue_does_not_require_optional_oxt(tmp_path: Path) -> None:
    source = build_hydrogenated_test_structure()
    result = _pack(
        tmp_path,
        source,
        build_heavy_only_packed_structure(),
        PackingSpec(backend_name="faspr", frozen_residue_ids=(ResidueId("A", 1),)),
    )
    assert result.constitution == source.constitution
    assert result.geometry == source.geometry
    assert result.topology == source.topology


def test_native_cannot_add_optional_oxt_to_complete_fixed_residue(
    tmp_path: Path,
) -> None:
    native = _with_sites(
        build_heavy_only_packed_structure(),
        additions=((AtomSite("OXT", "O"), AtomGeometry(Vec3(2, 2, 2))),),
    )
    with pytest.raises(
        PackingBackendExecutionError, match="added atoms to a fixed residue A:1"
    ):
        _pack(
            tmp_path,
            build_hydrogenated_test_structure(),
            native,
            PackingSpec(backend_name="faspr", frozen_residue_ids=(ResidueId("A", 1),)),
        )


@pytest.mark.parametrize("include_native_connections", (False, True))
def test_mutation_replaces_same_named_sidechain_graph(
    tmp_path: Path, include_native_connections: bool
) -> None:
    source = _with_sites(
        build_hydrogenated_test_structure(),
        component="ASP",
        additions=tuple(
            (AtomSite(name, element), AtomGeometry(Vec3(4 + index, 2, 1)))
            for index, (name, element) in enumerate(
                (("CG", "C"), ("OD1", "O"), ("OD2", "O"))
            )
        ),
    )
    cg_ref, od1_ref = (AtomRef(ResidueId("A", 1), name) for name in ("CG", "OD1"))
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                *source.topology.bonds,
                TopologyBond(
                    source.constitution.atom_index(cg_ref),
                    source.constitution.atom_index(od1_ref),
                    order=1,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    native = _with_sites(
        source,
        component="ASN",
        remove=frozenset({"OD2", "H", "HA", "HB1"}),
        additions=((AtomSite("ND2", "N"), AtomGeometry(Vec3(6, 2, 1))),),
    )
    result = _pack(
        tmp_path,
        source,
        native,
        PackingSpec(backend_name="faspr", target_sequence="N"),
        include_native_connections=include_native_connections,
    )
    bond = result.topology.bond_between(
        result.constitution.atom_index(cg_ref), result.constitution.atom_index(od1_ref)
    )
    assert bond is not None and bond.order == 2
    assert (
        result.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "OD2"))
        is None
    )
    assert (
        result.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "ND2"))
        is not None
    )


@pytest.mark.parametrize("cross_residue", (False, True))
def test_mutation_cannot_silently_discard_linked_sidechain_chemistry(
    tmp_path: Path, cross_residue: bool
) -> None:
    source = build_hydrogenated_test_structure(residue_count=2)
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                *source.topology.bonds,
                TopologyBond(
                    source.constitution.atom_index(AtomRef(ResidueId("A", 1), "CB")),
                    source.constitution.atom_index(
                        AtomRef(
                            ResidueId("A", 2 if cross_residue else 1),
                            "CA" if cross_residue else "O",
                        )
                    ),
                    provenance=(
                        BondProvenance.TEMPLATE_RESOLVED
                        if cross_residue
                        else BondProvenance.SOURCE_EXPLICIT
                    ),
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    native = _with_sites(
        build_heavy_only_packed_structure(residue_count=2),
        component="GLY",
        remove=frozenset({"CB"}),
    )
    with pytest.raises(PackingBackendExecutionError, match="linked chemistry"):
        _pack(
            tmp_path,
            source,
            native,
            PackingSpec(
                backend_name="faspr",
                mutable_residue_ids=(ResidueId("A", 1),),
                target_sequence="G",
            ),
        )
    assert (
        source.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CB"))
        is not None
    )


def test_native_cannot_add_unrecognized_sidechain_atom(tmp_path: Path) -> None:
    native = _with_sites(
        build_heavy_only_packed_structure(),
        additions=((AtomSite("CX", "C"), AtomGeometry(Vec3(3, 3, 3))),),
    )
    with pytest.raises(PackingBackendExecutionError, match="component scaffold"):
        _pack(tmp_path, build_hydrogenated_test_structure(), native)


def test_native_mutation_requires_complete_new_component_scaffold(
    tmp_path: Path,
) -> None:
    native = _with_sites(build_heavy_only_packed_structure(), component="SER")
    with pytest.raises(PackingBackendExecutionError, match="component scaffold"):
        _pack(
            tmp_path,
            build_hydrogenated_test_structure(),
            native,
            PackingSpec(backend_name="faspr", target_sequence="S"),
        )


@pytest.mark.parametrize("mutate", (False, True))
def test_native_new_atom_requires_component_element(
    tmp_path: Path, mutate: bool
) -> None:
    source = build_hydrogenated_test_structure()
    if mutate:
        native = _with_sites(
            build_heavy_only_packed_structure(),
            component="SER",
            additions=((AtomSite("OG", "N"), AtomGeometry(Vec3(3, 3, 3))),),
        )
        spec = PackingSpec(backend_name="faspr", target_sequence="S")
    else:
        source = _with_sites(source, remove=frozenset({"CB", "HB1"}))
        native = _with_sites(
            build_heavy_only_packed_structure(),
            remove=frozenset({"CB"}),
            additions=((AtomSite("CB", "N"), AtomGeometry(Vec3(3, 3, 3))),),
        )
        spec = PackingSpec(backend_name="faspr")
    with pytest.raises(PackingBackendExecutionError, match="component scaffold"):
        _pack(tmp_path, source, native, spec)


@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
)
def test_requested_mutation_uses_relationship_not_just_source_provenance(
    tmp_path: Path,
    relationship: BondRelationshipType,
) -> None:
    source = build_hydrogenated_test_structure()
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                *source.topology.bonds,
                TopologyBond(
                    source.constitution.atom_index(AtomRef(ResidueId("A", 1), "CB")),
                    source.constitution.atom_index(AtomRef(ResidueId("A", 1), "CA")),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    relationship_type=relationship,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    observation = StructureObservation.from_source_facets(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=source.topology,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        )
    )
    native = _with_sites(
        build_heavy_only_packed_structure(),
        component="GLY",
        remove=frozenset({"CB"}),
    )
    spec = PackingSpec(backend_name="faspr", target_sequence="G")
    if relationship is BondRelationshipType.UNKNOWN:
        with pytest.raises(PackingBackendExecutionError, match="linked chemistry"):
            _pack(tmp_path, source, native, spec)
        return
    result = _pack(
        tmp_path,
        source,
        native,
        spec,
    )
    assert result.constitution.chain("A").residues[0].component_id == "GLY"
    assert (
        result.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CB")) is None
    )
    assert result.provenance.ingress.observation is observation
    assert len(observation.topology.bonds) == 1
