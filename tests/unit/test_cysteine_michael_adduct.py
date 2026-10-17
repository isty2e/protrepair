"""Source-preserving terminal-acrylamide addition to polymer cysteine."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.component_reference import assert_component_reference_matches
from tests.support.cysteine_michael_adduct import (
    CYSTEINE,
    LIGAND,
    residue_payload,
)
from tests.support.cysteine_michael_adduct import (
    assert_original_preserved as _assert_original_preserved,
)
from tests.support.cysteine_michael_adduct import (
    qs7_source as _source,
)
from tests.support.cysteine_michael_adduct import (
    with_bonds as _with_bonds,
)
from tests.support.cysteine_michael_adduct import (
    with_thiol_hydrogen as _with_hydrogen,
)

from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.chemistry.restraint.template import PlaneRestraintTarget
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import repair
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _binding(source: ProteinStructure) -> CysteineMichaelAdductBinding:
    library = build_default_component_library()
    residue = source.constitution.residue_or_ligand(LIGAND)
    assert residue is not None
    return CysteineMichaelAdductBinding(
        source, residue, library.require("QS7"), library
    )


def test_qs7_bundle_matches_frozen_primary_ccd():
    bundled = build_bundled_retained_non_polymer_registry().get("QS7")
    frozen = read_nonstandard_component_registry(
        FIXTURES / "chemistry/retained/QS7.cif"
    ).get("QS7")
    assert bundled is not None and frozen is not None
    assert_component_reference_matches(
        replace(bundled, plane_targets=()), replace(frozen, plane_targets=())
    )
    assert set(bundled.plane_targets) == set(frozen.plane_targets)


def test_binding_projects_product_without_mutating_isolated_template_or_source():
    source = _source()
    binding = _binding(source)
    isolated = build_default_component_library().require("QS7")
    assert len(isolated.expected_hydrogen_atom_names()) == 23
    assert len(binding.template.expected_hydrogen_atom_names()) == 24
    assert not binding.cysteine_is_realized()
    assert {
        b.order
        for b in isolated.definition.bonds
        if {b.atom_name_1, b.atom_name_2} == {"CBF", "CBG"}
    } == {2}
    assert {
        b.order
        for b in binding.template.definition.bonds
        if {b.atom_name_1, b.atom_name_2} == {"CBF", "CBG"}
    } == {1}


@pytest.mark.parametrize("strict", (False, True))
def test_actual_linked_qs7_realizes_product_atomically_and_repeats(strict):
    source = _source()
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=not strict
    )
    assert not result.issues
    _assert_original_preserved(source, result.structure)
    ligand = result.structure.constitution.residue_or_ligand(LIGAND)
    assert ligand is not None
    assert sum(a.is_hydrogen() for a in ligand.atom_sites) == 24
    assert _binding(result.structure).cysteine_is_realized()
    for parent, names in (("CBF", ("H21", "H22")), ("CBG", ("H23", "H24"))):
        geometry = result.structure.residue_geometry(
            result.structure.constitution.residue_index(LIGAND)
        )
        for name in names:
            assert geometry.position(name).distance_to(
                geometry.position(parent)
            ) == pytest.approx(1.09)
    before = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    after = PolymerMicrostatePreparation(
        PolymerMicrostateContext(result.structure), build_default_component_library()
    )
    unresolved = before.fixed_hydrogen_context(CYSTEINE)
    assert unresolved is not None and unresolved.unresolved_parent_names == {"SG"}
    assert {parent for _, parent in before.fixed_hydrogen_atom_sites(CYSTEINE)} == {
        "CA",
        "CB",
    }
    assert {parent for _, parent in after.fixed_hydrogen_atom_sites(CYSTEINE)} == {
        "CA",
        "CB",
    }
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert not repeated.issues and not repeated.repairs
    assert repeated.structure.constitution == result.structure.constitution
    assert repeated.structure.geometry == result.structure.geometry
    assert repeated.structure.topology == result.structure.topology
    topology_only = resolve_retained_non_polymer_topology(
        result.structure, build_default_component_library()
    )
    assert not topology_only.issues
    assert topology_only.structure == result.structure


def test_topology_only_does_not_half_publish_a_missing_product_h_inventory():
    source = _source()
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.issues and result.structure is source


@pytest.mark.parametrize(
    "provenance", (BondProvenance.TEMPLATE_RESOLVED, BondProvenance.SOURCE_EXPLICIT)
)
@pytest.mark.parametrize("topology_only", (False, True))
def test_admitted_unknown_cysteine_context_publishes_covalent_chemistry(
    provenance, topology_only
):
    source = _source()
    if topology_only:
        source = add_retained_non_polymer_hydrogens(source).structure
    indices = tuple(
        source.constitution.atom_index(AtomRef(CYSTEINE, name)) for name in ("CB", "SG")
    )
    original = source.topology.bond_between(*indices)
    assert original is not None
    metadata = (
        SourceBondMetadata(
            record_type=SourceBondRecordType.PDB_CONECT, reported_order=1
        )
        if provenance is BondProvenance.SOURCE_EXPLICIT
        else None
    )
    unknown = replace(
        original,
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=provenance,
        source_metadata=metadata,
    )
    source = _with_bonds(
        source,
        tuple(
            unknown if bond.endpoint_pair() == unknown.endpoint_pair() else bond
            for bond in source.topology.bonds
        ),
    )

    result = (
        resolve_retained_non_polymer_topology(source, build_default_component_library())
        if topology_only
        else add_retained_non_polymer_hydrogens(source)
    )

    assert not result.issues
    assert _binding(result.structure).cysteine_is_realized()
    published = result.structure.topology.bond_between(
        *(
            result.structure.constitution.atom_index(AtomRef(CYSTEINE, name))
            for name in ("CB", "SG")
        )
    )
    assert published is not None
    assert published.relationship_type is BondRelationshipType.COVALENT
    assert published.order == 1
    if metadata is not None:
        assert published.provenance is provenance
        assert published.source_metadata == metadata
    _assert_original_preserved(source, result.structure)


def test_unlinked_qs7_remains_the_isolated_alkene():
    source = _source(linked=False)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    ligand = result.structure.constitution.residue_or_ligand(LIGAND)
    assert ligand is not None and sum(a.is_hydrogen() for a in ligand.atom_sites) == 23
    bond = result.structure.topology.bond_between(
        *(
            result.structure.constitution.atom_index(AtomRef(LIGAND, name))
            for name in ("CBF", "CBG")
        )
    )
    assert bond is not None and bond.order == 2


@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.METAL_COORDINATION, BondRelationshipType.HYDROGEN_BOND),
)
@pytest.mark.parametrize(
    "provenance",
    (
        BondProvenance.TEMPLATE_RESOLVED,
        BondProvenance.EVIDENCE_RESOLVED,
        BondProvenance.SOURCE_EXPLICIT,
    ),
)
@pytest.mark.parametrize("topology_only", (False, True))
def test_noncovalent_cysteine_context_is_not_silently_replaced(
    relationship, provenance, topology_only
):
    source = _source()
    if topology_only:
        source = add_retained_non_polymer_hydrogens(source).structure
    indices = tuple(
        source.constitution.atom_index(AtomRef(CYSTEINE, name)) for name in ("CB", "SG")
    )
    original = source.topology.bond_between(*indices)
    assert original is not None
    contradictory = replace(
        original,
        relationship_type=relationship,
        provenance=provenance,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_relationship_type=relationship,
        )
        if provenance is BondProvenance.SOURCE_EXPLICIT
        else None,
    )
    source = _with_bonds(
        source,
        tuple(
            contradictory
            if bond.endpoint_pair() == contradictory.endpoint_pair()
            else bond
            for bond in source.topology.bonds
        ),
    )

    result = (
        resolve_retained_non_polymer_topology(source, build_default_component_library())
        if topology_only
        else add_retained_non_polymer_hydrogens(source)
    )

    assert result.issues
    assert result.structure is source
    with pytest.raises(ValueError):
        _binding(source)


def test_realized_product_uses_only_applicable_default_restraints():
    library = build_default_component_library()
    restraints = build_default_restraint_library()
    source = _source()
    default = restraints.require("QS7")
    assert (
        StructureRestraints(source, library, restraints).for_residue(LIGAND) == default
    )

    prepared = add_retained_non_polymer_hydrogens(source).structure
    selected = StructureRestraints(prepared, library, restraints).for_residue(LIGAND)
    assert selected is not None
    centers = {"CBF", "CBG"}
    assert selected.bond_targets == tuple(
        target
        for target in default.bond_targets
        if not centers.intersection((target.atom_name_1, target.atom_name_2))
    )
    assert selected.angle_targets == tuple(
        target
        for target in default.angle_targets
        if target.center_atom_name not in centers
    )
    assert selected.chirality_targets == default.chirality_targets
    assert selected.plane_targets == default.plane_targets
    assert len(selected.bond_targets) < len(default.bond_targets)
    assert len(selected.angle_targets) < len(default.angle_targets)
    assert restraints.require("QS7") == default


def test_product_preserves_custom_restraint_dictionary():
    library = build_default_component_library()
    restraints = build_default_restraint_library()
    default = restraints.require("QS7")
    custom = replace(
        default,
        bond_targets=tuple(
            replace(target, target_distance_angstrom=1.54)
            if target.matches("CBF", "CBG")
            else target
            for target in default.bond_targets
        ),
    )
    prepared = add_retained_non_polymer_hydrogens(_source()).structure
    assert (
        StructureRestraints(
            prepared, library, restraints.with_template(custom)
        ).for_residue(LIGAND)
        == custom
    )


def test_product_restraint_projection_drops_only_reaction_site_planes():
    prepared = add_retained_non_polymer_hydrogens(_source()).structure
    default = build_default_restraint_library().require("QS7")
    reaction_plane = PlaneRestraintTarget(("NBD", "CBE", "CBF", "CBG"))
    augmented = replace(default, plane_targets=(*default.plane_targets, reaction_plane))
    selected = _binding(prepared).applicable_restraints(augmented)
    assert selected.plane_targets == default.plane_targets
    assert selected.chirality_targets == default.chirality_targets


@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing_ligand_atom",
        "missing_cysteine_atom",
        "unknown_link",
        "double_link",
        "wrong_endpoint",
        "extra_attachment",
        "explicit_alkene",
        "sulfur_charge",
        "ligand_charge",
        "reflected_stereo",
        "degenerate_stereo",
    ),
)
def test_adduct_contradiction_preserves_both_participants(mutation, topology_only):
    source = _source()
    sulfur = AtomRef(CYSTEINE, "SG")
    carbon = AtomRef(LIGAND, "CBG")
    link = source.topology.bond_between(
        source.constitution.atom_index(sulfur), source.constitution.atom_index(carbon)
    )
    assert link is not None
    if mutation.startswith("missing"):
        source = source.without_atom_refs(
            (
                AtomRef(CYSTEINE, "CB")
                if mutation == "missing_cysteine_atom"
                else AtomRef(LIGAND, "CBE"),
            )
        )
    elif mutation in {"sulfur_charge", "ligand_charge"}:
        residue_id, name = (
            (CYSTEINE, "SG") if mutation == "sulfur_charge" else (LIGAND, "CBF")
        )
        payload = residue_payload(source, residue_id)
        charges = dict(payload.formal_charge_by_atom_name)
        charges[name] = -1
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
            formal_charge_by_atom_name=tuple(charges.items()),
        )
    elif mutation in {"reflected_stereo", "degenerate_stereo"}:
        payload = residue_payload(source, LIGAND)
        geometry = payload.residue_geometry
        if mutation == "reflected_stereo":
            geometry = type(geometry)(
                {
                    name: replace(
                        g, position=Vec3(-g.position.x, g.position.y, g.position.z)
                    )
                    for name, g in geometry.atoms_by_name.items()
                }
            )
        else:
            geometry = geometry.with_atom_geometry(
                "NAX",
                replace(
                    geometry.atom_geometry("NAX"), position=geometry.position("CAY")
                ),
            )
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=geometry,
            formal_charge_by_atom_name=payload.formal_charge_by_atom_name,
        )
    else:
        changed = link
        bonds = source.topology.bonds
        if mutation == "unknown_link":
            changed = replace(
                link,
                relationship_type=BondRelationshipType.UNKNOWN,
                source_metadata=None,
            )
        elif mutation == "double_link":
            changed = replace(link, order=2, source_metadata=None)
        elif mutation == "wrong_endpoint":
            changed = replace(
                link,
                atom_index_2=source.constitution.atom_index(AtomRef(LIGAND, "CBF")),
                source_metadata=None,
            )
        elif mutation == "extra_attachment":
            bonds = (
                *bonds,
                TopologyBond(
                    source.constitution.atom_index(sulfur),
                    source.constitution.atom_index(AtomRef(LIGAND, "CBE")),
                ),
            )
        elif mutation == "explicit_alkene":
            original = source.topology.bond_between(
                *(
                    source.constitution.atom_index(AtomRef(LIGAND, name))
                    for name in ("CBF", "CBG")
                )
            )
            assert original is not None and original.source_metadata is not None
            explicit = replace(
                original,
                order=2,
                source_metadata=replace(original.source_metadata, reported_order=2),
            )
            bonds = tuple(explicit if old == original else old for old in bonds)
        source = _with_bonds(
            source, tuple(changed if old == link else old for old in bonds)
        )
    transform = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transform(source, build_default_component_library())
    assert result.issues and result.structure is source


@pytest.mark.parametrize("element", ("H", "D", "T"))
def test_original_thiol_h_is_never_deleted_to_force_the_product(element):
    source = _with_hydrogen(_source(), original=True, element=element)
    result = add_retained_non_polymer_hydrogens(source)
    assert result.issues and result.structure is source


@pytest.mark.parametrize("failure", ("placement", "charged_h", "topology_only"))
def test_failed_admission_does_not_remove_generated_partner_thiol_h(
    failure, monkeypatch
):
    source = _with_hydrogen(_source(), charge=1 if failure == "charged_h" else None)
    if failure == "placement":
        monkeypatch.setattr(
            repair, "generate_component_hydrogen_patch", lambda **kwargs: None
        )
    transform = (
        resolve_retained_non_polymer_topology
        if failure == "topology_only"
        else add_retained_non_polymer_hydrogens
    )
    result = transform(source, build_default_component_library())
    assert result.issues and result.structure is source


def test_generated_partner_thiol_h_is_removed_only_with_complete_ligand_product():
    source = _source()
    generated = _with_hydrogen(source)
    result = add_retained_non_polymer_hydrogens(generated)
    assert not result.issues and _binding(result.structure).cysteine_is_realized()
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(CYSTEINE, "HG"))
        is None
    )
    _assert_original_preserved(source, result.structure)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_product_survives_serialization_and_source_hydrogen_readmission(file_format):
    prepared = add_retained_non_polymer_hydrogens(_source()).structure
    recovered = read_structure_string(
        write_structure_string(prepared, file_format), file_format
    )
    result = resolve_retained_non_polymer_topology(
        recovered, build_default_component_library()
    )
    assert not result.issues
    assert _binding(result.structure).cysteine_is_realized()
    _assert_original_preserved(recovered, result.structure)


@pytest.mark.parametrize("element", ("D", "T"))
def test_product_preserves_noncanonical_isotope_name_and_actual_chirality(element):
    prepared = add_retained_non_polymer_hydrogens(_source()).structure
    lines = write_structure_string(prepared, FileFormat.PDB).splitlines()
    text = (
        "\n".join(
            line[:12] + " DX1" + line[16:76] + f"{element:>2}" + line[78:]
            if line.startswith("HETATM") and line[12:16].strip() == "H14"
            else line
            for line in lines
        )
        + "\n"
    )
    source = read_structure_string(text, FileFormat.PDB)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues and _binding(result.structure).cysteine_is_realized()
    _assert_original_preserved(source, result.structure)
    assert (
        result.structure.constitution.atom_site_at(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "DX1"))
        ).element
        == element
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_warhead_bond",
        "charged_warhead",
        "wrong_stereo",
        "missing_reference",
        "cysteine_skeleton",
        "cysteine_beta_bond",
    ),
)
def test_custom_template_cannot_mislabel_an_unsupported_product(mutation):
    library = build_default_component_library()
    template = library.require("QS7")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    if mutation in ("cysteine_skeleton", "cysteine_beta_bond"):
        cysteine = library.require("CYS")
        changed_pair = frozenset(
            ("CB", "SG") if mutation == "cysteine_skeleton" else ("CA", "CB")
        )
        cysteine = replace(
            cysteine,
            definition=replace(
                cysteine.definition,
                bonds=tuple(
                    replace(b, order=2)
                    if frozenset((b.atom_name_1, b.atom_name_2)) == changed_pair
                    else b
                    for b in cysteine.definition.bonds
                ),
            ),
        )
        library = library.with_template(cysteine)
    else:
        if mutation == "extra_warhead_bond":
            component = replace(
                component, bonds=(*component.bonds, BondDefinition("CBG", "CBE"))
            )
        else:
            name = "CBF" if mutation == "charged_warhead" else "CAY"
            component = replace(
                component,
                atoms=tuple(
                    replace(a, formal_charge=1)
                    if a.atom_name == name and mutation == "charged_warhead"
                    else replace(a, ideal_position=None)
                    if a.atom_name == name and mutation == "missing_reference"
                    else replace(a, stereo_config="R")
                    if a.atom_name == name
                    else a
                    for a in component.atoms
                ),
            )
        template = replace(
            template, hydrogen_semantics=IdealGeometryHydrogenSemantics(component)
        )
        library = library.with_template(template)
    source = _source()
    result = add_retained_non_polymer_hydrogens(source, library)
    assert result.issues and result.structure is source


@pytest.mark.parametrize("atom_names", (("CB", "SG"), ("CA", "CB")))
@pytest.mark.parametrize(
    "order, relationship_type",
    ((2, BondRelationshipType.COVALENT), (1, BondRelationshipType.DISULFIDE)),
)
def test_context_partner_reported_chemistry_cannot_be_overwritten(
    atom_names, order, relationship_type
):
    source = _source()
    explicit = TopologyBond(
        *(source.constitution.atom_index(AtomRef(CYSTEINE, n)) for n in atom_names),
        order=order,
        relationship_type=relationship_type,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            SourceBondRecordType.MMCIF_STRUCT_CONN,
            reported_order=order,
            reported_relationship_type=relationship_type,
        ),
    )
    source = _with_bonds(
        source,
        (
            *(
                bond
                for bond in source.topology.bonds
                if bond.endpoint_pair() != explicit.endpoint_pair()
            ),
            explicit,
        ),
    )
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, build_default_component_library())
        assert result.issues and result.structure is source


@pytest.mark.parametrize(
    "relationship_type",
    (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN),
)
def test_extra_beta_carbon_attachment_does_not_admit_the_coupled_product(
    relationship_type,
):
    source = _source()
    extra = TopologyBond(
        source.constitution.atom_index(AtomRef(CYSTEINE, "CB")),
        source.constitution.atom_index(AtomRef(CYSTEINE, "N")),
        relationship_type=relationship_type,
    )
    source = _with_bonds(source, (*source.topology.bonds, extra))
    result = add_retained_non_polymer_hydrogens(source)
    assert result.issues and result.structure is source


@pytest.mark.parametrize("original", (False, True))
def test_multiply_attached_thiol_h_cannot_be_silently_removed(original):
    source = _with_hydrogen(_source())
    ambiguous = TopologyBond(
        source.constitution.atom_index(AtomRef(CYSTEINE, "CA")),
        source.constitution.atom_index(AtomRef(CYSTEINE, "HG")),
    )
    source = _with_bonds(source, (*source.topology.bonds, ambiguous))
    if original:
        source = read_structure_string(
            write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
        )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.issues and result.structure is source


def test_absent_original_observation_does_not_authorize_thiol_h_removal():
    source = _with_hydrogen(_source())
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=source.topology,
        polymer_blueprint=source.polymer_blueprint,
        provenance=replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        ),
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.issues and result.structure is source


def test_original_charge_is_not_hidden_by_later_neutralization():
    source = _source()
    payload = residue_payload(source, LIGAND)
    charged = source.with_updated_residue_facets(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
        formal_charge_by_atom_name=(("CBF", 1),),
    )
    source = read_structure_string(
        write_structure_string(charged, FileFormat.MMCIF), FileFormat.MMCIF
    )
    payload = residue_payload(source, LIGAND)
    neutralized = source.with_updated_residue_facets(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
        formal_charge_by_atom_name=(("CBF", 0),),
    )
    result = add_retained_non_polymer_hydrogens(neutralized)
    assert result.issues and result.structure is neutralized


def test_rigid_rotation_preserves_admission_and_reaction_h_geometry():
    source = _source()
    for residue_id in (CYSTEINE, LIGAND):
        payload = residue_payload(source, residue_id)
        geometry = type(payload.residue_geometry)(
            {
                name: replace(
                    g,
                    position=Vec3(
                        g.position.y + 10, -g.position.x - 7, g.position.z + 3
                    ),
                )
                for name, g in payload.residue_geometry.atoms_by_name.items()
            }
        )
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=geometry,
            formal_charge_by_atom_name=payload.formal_charge_by_atom_name,
        )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues and _binding(result.structure).cysteine_is_realized()
    _assert_original_preserved(source, result.structure)
    ligand_geometry = result.structure.residue_geometry(
        result.structure.constitution.residue_index(LIGAND)
    )
    for parent, names in (("CBF", ("H21", "H22")), ("CBG", ("H23", "H24"))):
        for name in names:
            assert ligand_geometry.position(parent).distance_to(
                ligand_geometry.position(name)
            ) == pytest.approx(1.09)
