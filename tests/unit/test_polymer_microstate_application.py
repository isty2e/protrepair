"""Atomic coupled-graph application, independent of placement algorithms."""

from dataclasses import replace

import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import pras_microstate_preferences
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.protonation import (
    PrasRatioHistidineProtonationRequest,
    resolve_histidine_protonation_assignments,
)
from protrepair.transformer.polymer_microstate import (
    MicrostateHydrogenPlacement,
    PolymerMicrostatePatch,
    apply_polymer_microstate_patches,
)


def _structure(
    component: str,
    *,
    hydrogens: tuple[tuple[str, str], ...] = (),
    charges: tuple[tuple[str, int], ...] = (),
    terminal: bool = False,
    extra_pdb: str = "",
    second_component: str | None = None,
) -> ProteinStructure:
    library = build_standard_component_library()
    lines = []
    serial = 0
    for number, comp in enumerate((component, second_component), 1):
        if comp is None:
            continue
        atoms = [
            (name, name[0])
            for name in library.require(comp).expected_heavy_atom_names()
        ]
        if terminal:
            atoms.append(("OXT", "O"))
        if number == 1:
            atoms.extend(hydrogens)
        for name, element in atoms:
            serial += 1
            charge = dict(charges).get(name) if number == 1 else None
            token = (
                "" if charge is None else f"{abs(charge)}{'-' if charge < 0 else '+'}"
            )
            lines.append(
                f"ATOM  {serial:5d} {name:^4} {comp:>3} A{number:4d}    "
                f"{serial * 1.1:8.3f}{serial % 3 * 0.3:8.3f}{serial % 2 * 0.2:8.3f}"
                f"{1:6.2f}{20:6.2f}          {element:>2}{token:>2}\n"
            )
    return read_structure_string("".join(lines) + extra_pdb + "END\n", FileFormat.PDB)


def _patch(
    structure: ProteinStructure,
    *,
    residue_number: int = 1,
    kind: PolymerChemicalSite = PolymerChemicalSite.SIDECHAIN,
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN,
    override: MicrostateConstraints | None = None,
) -> PolymerMicrostatePatch:
    residue_id = ResidueId("A", residue_number)
    residue = structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    site = PolymerMicrostateSite(
        build_standard_component_library().require(residue.component_id), kind, linkage
    )
    preferences = pras_microstate_preferences(site)
    result = site.resolve(
        residue,
        structure.provenance.ingress.observation,
        override=override,
        preferences=preferences,
    )
    assert result.graph is not None
    placements = []
    for atom in result.graph.atoms:
        observed = [
            entry.hydrogen.atom_name
            for entry in result.observed_hydrogens
            if entry.parent.atom_name == atom.name
        ]
        for index in range(atom.hydrogens):
            # Synthetic placements isolate the graph-commit contract. They are not
            # an H geometry algorithm and do not support pose-quality claims.
            name = observed[index] if index < len(observed) else f"H{atom.name}{index}"
            placements.append(
                MicrostateHydrogenPlacement(
                    AtomSite(name, "H"),
                    atom.name,
                    AtomGeometry(Vec3(1.0 + index, 2.0, 3.0), 0.6, 18.0),
                )
            )
    return PolymerMicrostatePatch(
        structure,
        residue_id,
        site,
        tuple(placements),
        override=override,
        preferences=preferences,
    )


def _native(structure: ProteinStructure) -> Chem.Mol:
    molecule = Chem.RWMol()
    for index, atom in enumerate(structure.constitution.atom_slots):
        native = Chem.Atom("H" if atom.is_hydrogen() else atom.element)
        native.SetFormalCharge(structure.topology.formal_charge(AtomIndex(index)) or 0)
        molecule.AddAtom(native)
    for bond in structure.topology.bonds:
        if bond.relationship_type is BondRelationshipType.COVALENT:
            assert bond.order in (1, 2)
            molecule.AddBond(
                bond.atom_index_1.value,
                bond.atom_index_2.value,
                Chem.BondType.SINGLE if bond.order == 1 else Chem.BondType.DOUBLE,
            )
    result = molecule.GetMol()
    Chem.SanitizeMol(result)
    return result


@pytest.mark.parametrize("component", ("ARG", "LYS", "ASP", "GLU", "HIS"))
def test_applies_h_charge_and_orders_preserving_heavy_geometry(component: str) -> None:
    source = _structure(component)
    patch = _patch(source)
    repaired = apply_polymer_microstate_patches(source, (patch,))
    assert repaired.provenance is source.provenance
    assert repaired.polymer_blueprint is source.polymer_blueprint
    assert source.constitution.residue_site_at(ResidueIndex(0)).atom_site_names() == (
        build_standard_component_library()
        .require(component)
        .expected_heavy_atom_names()
    )
    for index, geometry in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        target = repaired.constitution.resolve_atom_index(ref)
        assert target is not None and repaired.geometry.position(target) == geometry
    for atom in patch.graph.atoms:
        index = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, atom.name)
        )
        assert index is not None
        assert repaired.topology.formal_charge(index) == atom.charge
    for bond in patch.graph.bonds:
        first = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, bond.atom_name_1)
        )
        second = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, bond.atom_name_2)
        )
        assert first is not None and second is not None
        actual = repaired.topology.bond_between(first, second)
        assert actual is not None and actual.order == bond.order
    native = _native(repaired)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in native.GetAtoms())
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(native)
    for atom in patch.graph.atoms:
        index = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, atom.name)
        )
        assert index is not None
        assert (
            native.GetAtomWithIdx(index.value).GetTotalNumHs(includeNeighbors=True)
            == atom.hydrogens
        )
    if component in ("ARG", "HIS", "ASP", "GLU"):
        for atom in patch.graph.atoms:
            index = repaired.constitution.resolve_atom_index(
                AtomRef(patch.residue_id, atom.name)
            )
            assert index is not None
            assert (
                native.GetAtomWithIdx(index.value).GetHybridization()
                == Chem.HybridizationType.SP2
            )


@pytest.mark.parametrize("isotope,name", (("H", "HD1"), ("D", "DD1"), ("T", "TD1")))
@pytest.mark.parametrize("strip", (False, True))
def test_preserves_original_h_identity_geometry_and_isotope(
    isotope: str, name: str, strip: bool
) -> None:
    original = _structure("HIS", hydrogens=((name, isotope),))
    source = original.without_hydrogens() if strip else original
    repaired = apply_polymer_microstate_patches(source, (_patch(source),))
    ref = AtomRef(ResidueId("A", 1), name)
    before = original.constitution.resolve_atom_index(ref)
    after = repaired.constitution.resolve_atom_index(ref)
    assert before is not None and after is not None
    assert repaired.constitution.atom_site_at(after).element == isotope
    assert repaired.geometry.atom_geometry(after) == original.geometry.atom_geometry(
        before
    )
    assert (
        repaired.provenance.ingress.observation
        is original.provenance.ingress.observation
    )


def test_regeneration_is_idempotent_and_request_change_removes_generated_h() -> None:
    source = _structure("LYS")
    first = apply_polymer_microstate_patches(source, (_patch(source),))
    second = apply_polymer_microstate_patches(first, (_patch(first),))
    assert second == first
    neutral = _patch(first, override=MicrostateConstraints(charges=(("NZ", 0),)))
    changed = apply_polymer_microstate_patches(first, (neutral,))
    index = changed.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "NZ"))
    assert index is not None and changed.topology.formal_charge(index) == 0
    assert (
        len(changed.constitution.atom_slots) == len(first.constitution.atom_slots) - 1
    )
    assert len(changed.topology.bonds) == len(first.topology.bonds) - 1
    assert (
        changed.provenance.ingress.observation is source.provenance.ingress.observation
    )


def test_coverage_complete_wrong_charge_is_still_repaired_without_moving_h() -> None:
    source = _structure("LYS")
    first = apply_polymer_microstate_patches(source, (_patch(source),))
    wrong = first.with_updated_residue_facets(
        first.constitution.residue_site_at(ResidueIndex(0)),
        residue_geometry=first.residue_geometry(ResidueIndex(0)),
        formal_charge_by_atom_name=(("NZ", 0),),
    )
    repaired = apply_polymer_microstate_patches(wrong, (_patch(wrong),))
    assert repaired.geometry == first.geometry
    assert repaired.constitution == first.constitution
    assert repaired.topology == first.topology


def test_stale_and_overlapping_patches_are_rejected() -> None:
    source = _structure("LYS")
    patch = _patch(source)
    with pytest.raises(ValueError, match="overlap"):
        apply_polymer_microstate_patches(source, (patch, patch))
    different = source.without_hydrogens()
    if different is source:
        different = _structure("LYS")
    with pytest.raises(ValueError, match="snapshot"):
        apply_polymer_microstate_patches(different, (patch,))


@pytest.mark.parametrize(
    "kind", (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C)
)
def test_free_terminal_patch_and_sidechain_patch_commit_together(
    kind: PolymerChemicalSite,
) -> None:
    source = _structure("LYS", terminal=True)
    sidechain = _patch(source)
    terminal = _patch(source, kind=kind, linkage=PeptideLinkage.FREE)
    repaired = apply_polymer_microstate_patches(source, (sidechain, terminal))
    for patch in (sidechain, terminal):
        for atom in patch.graph.atoms:
            index = repaired.constitution.resolve_atom_index(
                AtomRef(patch.residue_id, atom.name)
            )
            assert (
                index is not None
                and repaired.topology.formal_charge(index) == atom.charge
            )


def _with_bonds(
    source: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize(
    "mode", ("missing", "wrong_parent", "metal", "unknown", "order")
)
def test_current_boundary_guard_checks_identity_not_just_valence(mode: str) -> None:
    original = _structure("LYS")
    nz = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "NZ"))
    ce = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CE"))
    cg = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CG"))
    assert nz is not None and ce is not None and cg is not None
    bonds = tuple(bond for bond in original.topology.bonds if not bond.involves(nz))
    if mode != "missing":
        bond = TopologyBond(
            nz,
            cg if mode == "wrong_parent" else ce,
            order=2 if mode == "order" else 1,
            relationship_type=(
                BondRelationshipType.METAL_COORDINATION
                if mode == "metal"
                else BondRelationshipType.UNKNOWN
                if mode == "unknown"
                else BondRelationshipType.COVALENT
            ),
        )
        bonds += (bond,)
    source = _with_bonds(original, bonds)
    before = source.topology
    with pytest.raises(ValueError, match="boundary|relationship"):
        apply_polymer_microstate_patches(source, (_patch(source),))
    assert source.topology is before


def test_source_h_can_only_be_removed_when_override_reports_its_conflict() -> None:
    source = _structure("HIS", hydrogens=(("HD1", "D"), ("HE2", "H")))
    patch = _patch(
        source, override=MicrostateConstraints(hydrogens=(("ND1", 0), ("NE2", 1)))
    )
    assert patch.resolution.superseded_source.minimum_hydrogens == (("ND1", 1),)
    result = apply_polymer_microstate_patches(source, (patch,))
    assert (
        result.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "HD1"))
        is None
    )
    assert (
        result.provenance.ingress.observation is source.provenance.ingress.observation
    )


def test_original_h_cannot_be_replaced_by_an_equivalent_new_name() -> None:
    source = _structure("HIS", hydrogens=(("HD1", "D"),))
    patch = _patch(source)
    replacements = tuple(
        replace(entry, atom=AtomSite("RENAME", "H"))
        if entry.atom.name == "HD1"
        else entry
        for entry in patch.hydrogens
    )
    with pytest.raises(ValueError, match="original H identities"):
        PolymerMicrostatePatch(
            source,
            patch.residue_id,
            patch.site,
            replacements,
            preferences=pras_microstate_preferences(patch.site),
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_egress_projects_effective_charge_and_orders(file_format: FileFormat) -> None:
    source = _structure("ARG")
    patch = _patch(source)
    repaired = apply_polymer_microstate_patches(source, (patch,))
    reread = read_structure_string(
        write_structure_string(repaired, file_format), file_format
    )
    for atom in patch.graph.atoms:
        index = reread.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, atom.name)
        )
        assert index is not None
        # PDB cannot distinguish an explicit neutral charge from unspecified.
        assert (reread.topology.formal_charge(index) or 0) == atom.charge
    for bond in patch.graph.bonds:
        first = reread.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, bond.atom_name_1)
        )
        second = reread.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, bond.atom_name_2)
        )
        assert first is not None and second is not None
        actual = reread.topology.bond_between(first, second)
        assert actual is not None and actual.order == bond.order


def test_native_guanidinium_uses_changed_uff_parameters_not_only_sanitizability() -> (
    None
):
    source = _structure("ARG")
    repaired = apply_polymer_microstate_patches(source, (_patch(source),))
    indices = tuple(
        source.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), name))
        for name in ("CZ", "NH2")
    )
    assert all(index is not None for index in indices)
    first, second = indices
    assert first is not None and second is not None
    before = rdForceFieldHelpers.GetUFFBondStretchParams(
        _native(source), first.value, second.value
    )
    after = rdForceFieldHelpers.GetUFFBondStretchParams(
        _native(repaired), first.value, second.value
    )
    assert before is not None and after is not None
    assert before != after


def test_original_bond_metadata_survives_effective_order_override() -> None:
    heavy = (
        build_standard_component_library().require("ARG").expected_heavy_atom_names()
    )
    cz, nh1 = (heavy.index(name) + 1 for name in ("CZ", "NH1"))
    source = _structure(
        "ARG", charges=(("NH1", 1),), extra_pdb=f"CONECT{cz:5d}{nh1:5d}{nh1:5d}\n"
    )
    patch = _patch(source, override=MicrostateConstraints(charges=(("NH2", 1),)))
    repaired = apply_polymer_microstate_patches(source, (patch,))
    first = repaired.constitution.resolve_atom_index(AtomRef(patch.residue_id, "CZ"))
    second = repaired.constitution.resolve_atom_index(AtomRef(patch.residue_id, "NH1"))
    assert first is not None and second is not None
    bond = repaired.topology.bond_between(first, second)
    assert bond is not None and bond.order == 1
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.source_metadata is not None and bond.source_metadata.reported_order == 2
    assert (
        repaired.provenance.ingress.observation is source.provenance.ingress.observation
    )


@pytest.mark.parametrize("strip", (False, True))
def test_source_h_preserves_connectivity_metadata(strip: bool) -> None:
    heavy = (
        build_standard_component_library().require("HIS").expected_heavy_atom_names()
    )
    nd1, hd1 = heavy.index("ND1") + 1, len(heavy) + 1
    original = _structure(
        "HIS", hydrogens=(("HD1", "D"),), extra_pdb=f"CONECT{nd1:5d}{hd1:5d}\n"
    )
    source = original.without_hydrogens() if strip else original
    repaired = apply_polymer_microstate_patches(source, (_patch(source),))
    first = repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "ND1"))
    second = repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "HD1"))
    assert first is not None and second is not None
    bond = repaired.topology.bond_between(first, second)
    assert bond is not None and bond.order == 1
    assert bond.relationship_type is BondRelationshipType.COVALENT
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.source_metadata is not None


def test_batch_remaps_untouched_peptide_bond_and_is_order_independent() -> None:
    source = _structure("LYS", second_component="GLU")
    patches = (_patch(source), _patch(source, residue_number=2))
    repaired = apply_polymer_microstate_patches(source, patches)
    reverse = apply_polymer_microstate_patches(source, patches[::-1])
    assert repaired == reverse
    first = repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "C"))
    second = repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 2), "N"))
    assert first is not None and second is not None
    peptide = repaired.topology.bond_between(first, second)
    assert peptide is not None and peptide.order == 1


def test_batch_failure_does_not_apply_an_earlier_valid_site() -> None:
    original = _structure("LYS", second_component="GLU")
    cd = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 2), "CD"))
    cg = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 2), "CG"))
    assert cd is not None and cg is not None
    source = _with_bonds(
        original,
        tuple(
            bond
            for bond in original.topology.bonds
            if set(bond.endpoint_pair()) != {cd, cg}
        ),
    )
    before = source.topology
    with pytest.raises(ValueError, match="boundary"):
        apply_polymer_microstate_patches(
            source, (_patch(source), _patch(source, residue_number=2))
        )
    assert source.topology is before
    assert source.constitution == original.constitution


def test_noncovalent_record_cannot_be_misrepresented_as_a_covalent_graph_edge() -> None:
    original = _structure("HIS")
    nd1 = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "ND1"))
    ce1 = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CE1"))
    assert nd1 is not None and ce1 is not None
    source = _with_bonds(
        original,
        tuple(
            replace(bond, relationship_type=BondRelationshipType.HYDROGEN_BOND)
            if set(bond.endpoint_pair()) == {nd1, ce1}
            else bond
            for bond in original.topology.bonds
        ),
    )
    with pytest.raises(ValueError, match="noncovalent"):
        apply_polymer_microstate_patches(source, (_patch(source),))


def test_original_h_rebonded_elsewhere_after_ingress_is_not_silently_bridged() -> None:
    original = _structure("HIS", hydrogens=(("HD1", "H"),))
    h = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "HD1"))
    cb = original.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "CB"))
    assert h is not None and cb is not None
    source = _with_bonds(
        original,
        (
            *(bond for bond in original.topology.bonds if not bond.involves(h)),
            TopologyBond(h, cb),
        ),
    )
    with pytest.raises(ValueError, match="attachment disagrees"):
        apply_polymer_microstate_patches(source, (_patch(source),))


@pytest.mark.parametrize("bonded", (False, True))
def test_charged_current_source_h_is_not_silently_neutralized(bonded: bool) -> None:
    original = _structure("HIS", hydrogens=(("HD1", "H"),))
    source = original.with_updated_residue_facets(
        original.constitution.residue_site_at(ResidueIndex(0)),
        residue_geometry=original.residue_geometry(ResidueIndex(0)),
        formal_charge_by_atom_name=(("HD1", 1),),
    )
    h = source.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "HD1"))
    nd1 = source.constitution.resolve_atom_index(AtomRef(ResidueId("A", 1), "ND1"))
    assert h is not None and nd1 is not None
    bonds = tuple(bond for bond in source.topology.bonds if not bond.involves(h))
    source = _with_bonds(source, bonds + ((TopologyBond(h, nd1),) if bonded else ()))

    with pytest.raises(ValueError, match="hydrogen chemistry"):
        apply_polymer_microstate_patches(source, (_patch(source),))
    assert source.topology.formal_charge(h) == 1
    assert (
        source.provenance.ingress.observation is original.provenance.ingress.observation
    )


@pytest.mark.parametrize("component", ("ALA", "PRO"))
def test_linked_backbone_context_requires_the_actual_peptide_edge(
    component: str,
) -> None:
    source = _structure("ALA", second_component=component)
    patch = _patch(
        source,
        residue_number=2,
        kind=PolymerChemicalSite.BACKBONE_N,
        linkage=PeptideLinkage.LINKED,
    )
    repaired = apply_polymer_microstate_patches(source, (patch,))
    index = repaired.constitution.resolve_atom_index(AtomRef(ResidueId("A", 2), "N"))
    assert index is not None and repaired.topology.formal_charge(index) == 0
    native = _native(repaired)
    assert (
        native.GetAtomWithIdx(index.value).GetHybridization()
        == Chem.HybridizationType.SP2
    )
    assert native.GetAtomWithIdx(index.value).GetTotalNumHs(includeNeighbors=True) == (
        0 if component == "PRO" else 1
    )

    standalone = _structure(component)
    with pytest.raises(ValueError, match="boundary"):
        apply_polymer_microstate_patches(
            standalone,
            (
                _patch(
                    standalone,
                    kind=PolymerChemicalSite.BACKBONE_N,
                    linkage=PeptideLinkage.LINKED,
                ),
            ),
        )


def test_ambiguous_or_conflicting_chemistry_cannot_construct_a_patch() -> None:
    source = _structure("LYS")
    site = PolymerMicrostateSite(
        build_standard_component_library().require("LYS"), PolymerChemicalSite.SIDECHAIN
    )
    with pytest.raises(ValueError, match="ambiguous"):
        PolymerMicrostatePatch(source, ResidueId("A", 1), site, ())
    conflicting = _structure("LYS", charges=(("NZ", -1),))
    with pytest.raises(ValueError, match="conflict"):
        PolymerMicrostatePatch(
            conflicting,
            ResidueId("A", 1),
            site,
            (),
            preferences=pras_microstate_preferences(site),
        )


def test_incomplete_or_duplicate_h_placements_are_rejected() -> None:
    source = _structure("LYS")
    patch = _patch(source)
    for placements, message in (
        (patch.hydrogens[:-1], "exactly match"),
        (patch.hydrogens + patch.hydrogens[:1], "unique"),
    ):
        with pytest.raises(ValueError, match=message):
            PolymerMicrostatePatch(
                source,
                patch.residue_id,
                patch.site,
                placements,
                preferences=pras_microstate_preferences(patch.site),
            )


@pytest.mark.parametrize(
    "geometry",
    (
        AtomGeometry(Vec3(0, 0, 0), occupancy=float("nan")),
        AtomGeometry(Vec3(0, 0, 0), occupancy=-0.1),
        AtomGeometry(Vec3(0, 0, 0), b_factor=float("inf")),
        AtomGeometry(Vec3(0, 0, 0), b_factor=-1.0),
    ),
)
def test_nonfinite_or_invalid_placement_scalars_are_rejected(
    geometry: AtomGeometry,
) -> None:
    with pytest.raises(ValueError, match="finite scalars"):
        MicrostateHydrogenPlacement(AtomSite("HZ1", "H"), "NZ", geometry)


def test_ratio_assignment_selects_the_complete_cation_not_only_an_hd1_append() -> None:
    source = _structure("HIS")
    (assignment,) = resolve_histidine_protonation_assignments(
        source.constitution.chains[0], PrasRatioHistidineProtonationRequest(ratio=1.0)
    )
    patch = _patch(source, override=assignment.microstate_constraints())
    repaired = apply_polymer_microstate_patches(source, (patch,))
    assert patch.graph.protonation_key()[0] == 1
    assert patch.graph.atom("ND1").hydrogens == patch.graph.atom("NE2").hydrogens == 1
    native = _native(repaired)
    assert Chem.GetFormalCharge(native) == 1
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in native.GetAtoms())
