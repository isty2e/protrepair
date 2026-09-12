"""Graph-controlled automatic placement, source identity and actual 3-D geometry."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

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
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType, StructureTopology
from protrepair.transformer.completion.heavy.core import repair_heavy_atoms_core
from protrepair.transformer.completion.hydrogen import microstate
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
    PrasRatioHistidineProtonationRequest,
    histidine_microstate_requests,
)
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.transformer.polymer_microstate import (
    PolymerMicrostatePatch,
    apply_polymer_microstate_patches,
)


@pytest.fixture(scope="module")
def source() -> ProteinStructure:
    path = Path(__file__).parents[1] / "fixtures/corpus/pdb1afc.ent"
    return read_structure_string(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if line.startswith("ATOM") and line[21] == "A"
        ),
        FileFormat.PDB,
    )


def _site(source: ProteinStructure, component: str):
    template = build_standard_component_library().require(component)
    residue = next(
        entry
        for chain in source.constitution.chains
        for entry in chain.residues
        if entry.component_id == component
        and set(template.expected_heavy_atom_names()) <= set(entry.atom_site_names())
    )
    return residue.residue_id, PolymerMicrostateSite(
        template, PolymerChemicalSite.SIDECHAIN
    )


def _place(
    source: ProteinStructure, component: str, **kwargs
) -> PolymerMicrostatePatch:
    residue_id, site = _site(source, component)
    return microstate.place_polymer_microstate_hydrogens(
        PolymerMicrostateContext(source),
        residue_id,
        site,
        preferences=pras_microstate_preferences(site),
        **kwargs,
    )


def test_preparation_distinguishes_saved_choice_from_changed_ratio(source):
    cation = _place(
        source,
        "HIS",
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    charged = _assert_realized(source, cation)
    library = build_standard_component_library()

    unchanged = PolymerMicrostatePreparation(
        PolymerMicrostateContext(charged),
        library,
        requests=histidine_microstate_requests(
            charged, DisabledHistidineProtonationRequest()
        ),
    ).targets_for(cation.residue_id)[0]
    assert unchanged.is_realized()
    assert unchanged.resolution.graph is not None
    assert unchanged.resolution.graph.protonation_key()[0] == 1

    reset = PolymerMicrostatePreparation(
        PolymerMicrostateContext(charged),
        library,
        requests=histidine_microstate_requests(
            charged, PrasRatioHistidineProtonationRequest(0.0)
        ),
    ).targets_for(cation.residue_id)[0]
    assert not reset.is_realized()
    assert reset.resolution.graph is not None
    assert reset.resolution.graph.protonation_key()[0] == 0
    patch = microstate.place_polymer_microstate_hydrogens(
        reset.context,
        reset.residue_id,
        reset.site,
        override=reset.override,
        retain_applied_override=reset.retain_applied_override,
        preferences=pras_microstate_preferences(reset.site),
    )
    neutral = _assert_realized(charged, patch)
    assert neutral.provenance.microstate_overrides == ()
    assert (
        PolymerMicrostatePreparation(PolymerMicrostateContext(neutral), library)
        .targets_for(cation.residue_id)[0]
        .is_realized()
    )


def test_preparation_retains_unresolved_site_instead_of_empty_success(source):
    missing = next(
        r
        for r in source.constitution.chains[0].residues
        if r.component_id == "LYS" and not r.has_atom_site("NZ")
    )
    targets = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_standard_component_library()
    ).targets_for(missing.residue_id)
    sidechain = targets[0]
    assert sidechain.controlled_parent_names() == frozenset({"NZ"})
    assert sidechain.resolution.graph is None
    assert not sidechain.is_realized()
    with pytest.raises(ValueError):
        sidechain.hydrogen_atom_sites()


def test_preparation_rejects_unmatched_request(source):
    first = next(
        r for r in source.constitution.chains[0].residues if r.component_id == "ALA"
    )
    with pytest.raises(ValueError, match="absent polymer site"):
        PolymerMicrostatePreparation(
            PolymerMicrostateContext(source),
            build_standard_component_library(),
            requests={(first.residue_id, PolymerChemicalSite.SIDECHAIN): None},
        )


def _assert_realized(
    source: ProteinStructure, patch: PolymerMicrostatePatch
) -> ProteinStructure:
    repaired = apply_polymer_microstate_patches(source, (patch,))
    context = PolymerMicrostateContext(repaired)
    resolution = context.resolve(
        patch.residue_id,
        patch.site,
        preferences=pras_microstate_preferences(patch.site),
    )
    assert context.is_realized(patch.residue_id, patch.site, resolution)
    # Compare by stable identity, not slots shifted by H insertion.
    for atom_ref in (
        source.constitution.atom_ref_at(index)
        for index, _ in source.geometry.iter_positions()
    ):
        before = source.constitution.resolve_atom_index(atom_ref)
        after = repaired.constitution.resolve_atom_index(atom_ref)
        assert before is not None
        if not source.constitution.atom_site_at(before).is_hydrogen():
            assert after is not None
            assert source.geometry.atom_geometry(
                before
            ) == repaired.geometry.atom_geometry(after)
    for entry in patch.hydrogens:
        parent = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, entry.parent_name)
        )
        assert parent is not None
        assert (
            0.8
            < entry.geometry.position.distance_to(repaired.geometry.position(parent))
            < 1.2
        )
    return repaired


@pytest.mark.parametrize("component", ("ARG", "LYS", "ASP", "GLU", "HIS"))
def test_all_catalog_graphs_have_complete_real_placements(source, component):
    residue_id, site = _site(source, component)
    catalog = standard_microstate_candidates(component, site.kind)
    assert catalog is not None
    for graph in catalog.candidates:
        patch = microstate.place_polymer_microstate_hydrogens(
            PolymerMicrostateContext(source),
            residue_id,
            site,
            override=MicrostateConstraints(
                charges=tuple((atom.name, atom.charge) for atom in graph.atoms),
                hydrogens=tuple((atom.name, atom.hydrogens) for atom in graph.atoms),
                bonds=graph.bonds,
            ),
        )
        assert patch.graph == graph
        _assert_realized(source, patch)


@pytest.mark.parametrize("component", ("ARG", "LYS", "ASP", "GLU", "HIS"))
def test_default_placement_and_replay_are_idempotent(source, component):
    patch = _place(source, component)
    repaired = _assert_realized(source, patch)
    replay = _place(repaired, component)
    assert apply_polymer_microstate_patches(repaired, (replay,)) == repaired


def test_linked_backbone_n_h_is_in_actual_peptide_plane(source):
    residue_id, sidechain = _site(source, "GLN")
    template = sidechain.template
    context = PolymerMicrostateContext(source)
    site = context.backbone_site(residue_id, template, PolymerChemicalSite.BACKBONE_N)
    patch = microstate.place_polymer_microstate_hydrogens(context, residue_id, site)
    assert len(patch.hydrogens) == 1

    def position(ref):
        index = source.constitution.resolve_atom_index(ref)
        assert index is not None
        return source.geometry.position(index).to_array()

    n = position(AtomRef(residue_id, "N"))
    ca = position(AtomRef(residue_id, "CA"))
    partner = next(
        ref
        for ref in context.covalent_neighbors(AtomRef(residue_id, "N"))
        if ref.residue_id != residue_id
    )
    c = position(partner)
    h = patch.hydrogens[0].geometry.position.to_array()
    normal = np.cross(ca - n, c - n)
    assert abs(np.dot(h - n, normal) / np.linalg.norm(normal)) < 1e-8
    assert np.dot(h - n, ca - n) < 0
    assert np.dot(h - n, c - n) < 0


@pytest.mark.parametrize(
    "kind", (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C)
)
def test_terminal_preparation_uses_coupled_graph(source, kind):
    source = repair_heavy_atoms_core(source, local_refinement=None).structure
    source = augment_c_terminal_oxt(source).structure
    residue = source.constitution.chains[0].residues[
        0 if kind is PolymerChemicalSite.BACKBONE_N else -1
    ]
    template = build_standard_component_library().require(residue.component_id)
    context = PolymerMicrostateContext(source)
    site = context.backbone_site(
        residue.residue_id, template, kind, assume_free_chain_ends=True
    )
    patch = microstate.place_polymer_microstate_hydrogens(
        context, residue.residue_id, site, preferences=pras_microstate_preferences(site)
    )
    assert site.free_terminal_assumption
    _assert_realized(source, patch)


def _with_original_h(source, component, name, element, parent):
    residue_id, _site_value = _site(source, component)
    residue_index = source.constitution.residue_index(residue_id)
    residue = source.constitution.residue_site_at(residue_index)
    geometry = source.residue_geometry(residue_index)
    h_geometry = AtomGeometry(
        Vec3.from_iterable(geometry.position(parent).to_array() + (1.0, 0.0, 0.0)),
        0.7,
        12.0,
    )
    changed = source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite(name, element)),
        residue_geometry=geometry.with_atom_geometry(name, h_geometry),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(
            residue_index
        ),
    )
    # A fresh original observation, not generated atoms masquerading as evidence.
    text = write_structure_string(changed, FileFormat.PDB)
    return read_structure_string(
        "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith(("CONECT", "LINK", "SSBOND"))
        ),
        FileFormat.PDB,
    ), residue_id


@pytest.mark.parametrize("name,element", (("HZ1", "H"), ("DZ1", "D"), ("TZ1", "T")))
@pytest.mark.parametrize("rebuild", (False, True))
def test_original_h_identity_scalars_and_geometry_authority(
    source, name, element, rebuild
):
    original, residue_id = _with_original_h(source, "LYS", name, element, "NZ")
    index = original.constitution.resolve_atom_index(AtomRef(residue_id, name))
    assert index is not None
    before = original.geometry.atom_geometry(index)
    patch = _place(original, "LYS", rebuild_existing=rebuild)
    repaired = apply_polymer_microstate_patches(original, (patch,))
    index = repaired.constitution.resolve_atom_index(AtomRef(residue_id, name))
    assert index is not None
    assert repaired.constitution.atom_site_at(index).element == element
    after = repaired.geometry.atom_geometry(index)
    assert after.occupancy == before.occupancy
    assert after.b_factor == before.b_factor
    assert (after.position != before.position) is rebuild
    assert (
        repaired.provenance.ingress.observation
        is original.provenance.ingress.observation
    )


def test_rebuild_does_not_restore_stale_original_h_coordinates(source):
    original, residue_id = _with_original_h(source, "LYS", "DZ1", "D", "NZ")
    index = original.constitution.residue_index(residue_id)
    geometry = original.residue_geometry(index).without_atoms({"DZ1"})
    shift = (3.0, 2.0, 1.0)
    geometry = geometry.with_atom_geometries(
        (
            name,
            value.with_position(Vec3.from_iterable(value.position.to_array() + shift)),
        )
        for name, value in geometry.atoms_by_name.items()
    )
    moved = original.with_updated_residue_facets(
        original.constitution.residue_site_at(index).without_atom_sites({"DZ1"}),
        residue_geometry=geometry,
        formal_charge_by_atom_name=original.residue_formal_charge_by_atom_name(index),
    )
    patch = _place(moved, "LYS", rebuild_existing=True)
    repaired = apply_polymer_microstate_patches(moved, (patch,))
    h = repaired.constitution.resolve_atom_index(AtomRef(residue_id, "DZ1"))
    n = repaired.constitution.resolve_atom_index(AtomRef(residue_id, "NZ"))
    assert (
        0.9
        < repaired.geometry.position(h).distance_to(repaired.geometry.position(n))
        < 1.2
    )
    assert repaired.constitution.atom_site_at(h).element == "D"


def test_source_h_tautomer_is_not_defaulted_away(source):
    original, _residue_id = _with_original_h(source, "HIS", "HD1", "H", "ND1")
    patch = _place(original, "HIS")
    assert patch.graph.atom("ND1").hydrogens == 1
    assert patch.graph.atom("NE2").hydrogens == 0


def test_explicit_cationic_his_then_new_neutral_request(source):
    cation = _place(
        source,
        "HIS",
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    repaired = apply_polymer_microstate_patches(source, (cation,))
    assert _place(repaired, "HIS").graph == cation.graph
    neutral = _place(
        repaired,
        "HIS",
        override=MicrostateConstraints(hydrogens=(("ND1", 0), ("NE2", 1))),
    )
    _assert_realized(repaired, neutral)
    assert sum(atom.charge for atom in neutral.graph.atoms) == 0


def test_unavailable_backend_and_invalid_rebuild_are_explicit(source, monkeypatch):
    monkeypatch.setattr(microstate, "Chem", None)
    with pytest.raises(RdkitUnavailableError):
        _place(source, "LYS")
    with pytest.raises(TypeError, match="boolean"):
        _place(source, "LYS", rebuild_existing="yes")


def test_repositioning_cannot_target_heavy_or_absent_atoms(source):
    patch = _place(source, "LYS")
    with pytest.raises(ValueError, match="final site H"):
        replace(
            patch,
            reposition_hydrogens=frozenset({"NZ"}),
            preferences=pras_microstate_preferences(patch.site),
        )


def test_degenerate_native_output_is_not_published(source, monkeypatch):
    assert microstate.Chem is not None
    original = microstate.Chem.AddHs

    def broken(*args, **kwargs):
        molecule = original(*args, **kwargs)
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 1:
                parent = atom.GetNeighbors()[0].GetIdx()
                molecule.GetConformer().SetAtomPosition(
                    atom.GetIdx(), molecule.GetConformer().GetAtomPosition(parent)
                )
        return molecule

    monkeypatch.setattr(microstate.Chem, "AddHs", broken)
    with pytest.raises(GeometryPlacementError, match="bond vector"):
        _place(source, "LYS")


def test_missing_terminal_oxygen_is_not_replaced_by_an_aldehyde_stencil(source):
    residue = source.constitution.chains[0].residues[-1]
    index = source.constitution.residue_index(residue.residue_id)
    missing = source.with_updated_residue_facets(
        residue.without_atom_sites({"OXT"}),
        residue_geometry=source.residue_geometry(index).without_atoms({"OXT"}),
        formal_charge_by_atom_name=tuple(
            (name, charge)
            for name, charge in source.residue_formal_charge_by_atom_name(index)
            if name != "OXT"
        ),
    )
    context = PolymerMicrostateContext(missing)
    site = context.backbone_site(
        residue.residue_id,
        build_standard_component_library().require(residue.component_id),
        PolymerChemicalSite.BACKBONE_C,
        assume_free_chain_ends=True,
    )
    with pytest.raises(ValueError, match="missing heavy atoms: OXT"):
        microstate.place_polymer_microstate_hydrogens(
            context,
            residue.residue_id,
            site,
            preferences=pras_microstate_preferences(site),
        )


def test_planar_amide_placement_cannot_invent_a_partner_carbonyl(source):
    residue_id, sidechain = _site(source, "GLN")
    context = PolymerMicrostateContext(source)
    partner = next(
        ref
        for ref in context.covalent_neighbors(AtomRef(residue_id, "N"))
        if ref.residue_id != residue_id
    )
    c_index = source.constitution.resolve_atom_index(partner)
    o_index = source.constitution.resolve_atom_index(AtomRef(partner.residue_id, "O"))
    bonds = tuple(
        replace(bond, order=1)
        if set(bond.endpoint_pair()) == {c_index, o_index}
        else bond
        for bond in source.topology.bonds
    )
    changed = ProteinStructure.from_payload(
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
    context = PolymerMicrostateContext(changed)
    site = context.backbone_site(
        residue_id, sidechain.template, PolymerChemicalSite.BACKBONE_N
    )
    with pytest.raises(GeometryPlacementError, match="actual partner C=O"):
        microstate.place_polymer_microstate_hydrogens(context, residue_id, site)


def test_original_unknown_h_attachment_is_not_assigned_by_proximity(source):
    original, _residue_id = _with_original_h(source, "LYS", "HQ", "H", "NZ")
    with pytest.raises(ValueError, match="insufficient"):
        _place(original, "LYS")


def test_cationic_guanidinium_h_remains_planar(source):
    patch = _place(source, "ARG")
    residue_geometry = source.residue_geometry(
        source.constitution.residue_index(patch.residue_id)
    )
    points = {
        name: residue_geometry.position(name).to_array()
        for name in ("CZ", "NE", "NH1", "NH2")
    }
    for parent in ("NH1", "NH2"):
        n = points[parent]
        normal = np.cross(points["CZ"] - n, points["NE"] - n)
        unit = normal / np.linalg.norm(normal)
        hydrogens = [
            entry.geometry.position.to_array()
            for entry in patch.hydrogens
            if entry.parent_name == parent
        ]
        assert len(hydrogens) == 2
        # The measured heavy group is not perfectly planar. Use its actual
        # local plane, rather than an absolute laboratory-axis assertion.
        assert all(abs(np.dot(position - n, unit)) < 0.1 for position in hydrogens)


def test_lysine_amine_has_tetrahedral_not_planar_h_directions(source):
    patch = _place(source, "LYS")
    geometry = source.residue_geometry(
        source.constitution.residue_index(patch.residue_id)
    )
    n = geometry.position("NZ").to_array()
    vectors = [geometry.position("CE").to_array() - n]
    vectors.extend(entry.geometry.position.to_array() - n for entry in patch.hydrogens)
    unit = [vector / np.linalg.norm(vector) for vector in vectors]
    for left in range(4):
        for right in range(left + 1, 4):
            assert np.dot(unit[left], unit[right]) == pytest.approx(-1 / 3, abs=0.002)


def test_preserved_h_uses_a_slot_instead_of_receiving_an_overlapping_new_h(source):
    placed = _place(source, "LYS")
    original, residue_id = _with_original_h(source, "LYS", "DZ1", "D", "NZ")
    index = original.constitution.residue_index(residue_id)
    geometry = original.residue_geometry(index)
    geometry = geometry.with_atom_geometry(
        "DZ1",
        geometry.atom_geometry("DZ1").with_position(
            placed.hydrogens[0].geometry.position
        ),
    )
    original = original.with_updated_residue_facets(
        original.constitution.residue_site_at(index),
        residue_geometry=geometry,
        formal_charge_by_atom_name=original.residue_formal_charge_by_atom_name(index),
    )
    patch = _place(original, "LYS")
    assert len(patch.hydrogens) == len(placed.hydrogens) == 3
    for left in range(3):
        for right in range(left + 1, 3):
            assert (
                patch.hydrogens[left].geometry.distance_to(
                    patch.hydrogens[right].geometry
                )
                > 1.0
            )


def _native_graph(source: ProteinStructure):
    molecule = Chem.RWMol()
    for number, atom in enumerate(source.constitution.atom_slots):
        native = Chem.Atom("H" if atom.is_hydrogen() else atom.element)
        native.SetFormalCharge(source.topology.formal_charge(AtomIndex(number)) or 0)
        molecule.AddAtom(native)
    for bond in source.topology.bonds:
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


def test_actual_graph_not_coordinate_stencil_controls_uff_parameters(source):
    patch = _place(source, "ARG")
    repaired = apply_polymer_microstate_patches(source, (patch,))
    before = _native_graph(source)
    after = _native_graph(repaired)
    before_indices = []
    after_indices = []
    for name in ("CZ", "NH2"):
        before_index = source.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, name)
        )
        after_index = repaired.constitution.resolve_atom_index(
            AtomRef(patch.residue_id, name)
        )
        assert before_index is not None and after_index is not None
        before_indices.append(before_index.value)
        after_indices.append(after_index.value)
    assert all(
        before.GetAtomWithIdx(index).GetHybridization() == Chem.HybridizationType.SP3
        for index in before_indices
    )
    assert all(
        after.GetAtomWithIdx(index).GetHybridization() == Chem.HybridizationType.SP2
        for index in after_indices
    )
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in after.GetAtoms())
    before_params = rdForceFieldHelpers.GetUFFBondStretchParams(before, *before_indices)
    after_params = rdForceFieldHelpers.GetUFFBondStretchParams(after, *after_indices)
    assert before_params is not None and after_params is not None
    assert after_params[0] > before_params[0]
    assert after_params[1] < before_params[1]


def test_native_backend_cannot_move_fixed_anchors(source, monkeypatch):
    original = Chem.AddHs

    def moved(*args, **kwargs):
        molecule = original(*args, **kwargs)
        molecule.GetConformer().SetAtomPosition(0, (999.0, 0.0, 0.0))
        return molecule

    monkeypatch.setattr(Chem, "AddHs", moved)
    with pytest.raises(GeometryPlacementError, match="fixed coordinate anchor"):
        _place(source, "LYS")


@pytest.mark.parametrize("component", ("ARG", "LYS", "HIS"))
def test_placement_is_keyed_by_identity_not_heavy_atom_storage_order(source, component):
    patch = _place(source, component)
    index = source.constitution.residue_index(patch.residue_id)
    residue = source.constitution.residue_site_at(index)
    changed = source.with_updated_residue_facets(
        replace(residue, atom_sites=tuple(reversed(residue.atom_sites))),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    reordered = _place(changed, component)
    assert reordered.graph == patch.graph
    assert reordered.hydrogens == patch.hydrogens


def test_original_deuterium_removal_requires_an_explicit_chemical_request(source):
    original, residue_id = _with_original_h(source, "HIS", "DD1", "D", "ND1")
    assert "DD1" in {entry.atom.name for entry in _place(original, "HIS").hydrogens}
    patch = _place(
        original,
        "HIS",
        override=MicrostateConstraints(hydrogens=(("ND1", 0), ("NE2", 1))),
    )
    assert dict(patch.resolution.superseded_source.minimum_hydrogens) == {"ND1": 1}
    repaired = apply_polymer_microstate_patches(original, (patch,))
    assert repaired.constitution.resolve_atom_index(AtomRef(residue_id, "DD1")) is None
    assert (
        repaired.constitution.resolve_atom_index(AtomRef(residue_id, "HE2")) is not None
    )
    assert (
        repaired.provenance.ingress.observation
        is original.provenance.ingress.observation
    )
