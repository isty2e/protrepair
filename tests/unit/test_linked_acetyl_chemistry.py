"""Context-conditioned acetyl chemistry preserves the deposited graph."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.chemistry.retained_non_polymer.template_binding import (
    is_resolved_acyl_amide_link,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.structure_axes import ComponentSupportState
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)

CAP = ResidueId("A", 0)
FIRST = ResidueId("A", 1)
PDB_PATH = Path(__file__).parents[1] / "fixtures/corpus/pdb2zta.ent"


def _source(
    *, linked=True, anchor="C", partner="N", missing=(), elements=(), charges=()
):
    records = []
    element_map = dict(elements)
    charge_map = dict(charges)
    for line in PDB_PATH.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == "A" and int(line[22:26]) == 1:
            key = (FIRST, line[12:16].strip())
            if key in charge_map:
                charge = charge_map[key]
                line = (
                    line.ljust(80)[:78] + f"{abs(charge)}{'+' if charge >= 0 else '-'}"
                )
            records.append(line)
        elif line.startswith("HETATM") and line[17:20] == "ACE" and line[21] == "A":
            name = line[12:16].strip()
            if name in missing:
                continue
            if name in element_map:
                line = line[:76] + element_map[name].rjust(2) + line[78:]
            key = (CAP, name)
            if key in charge_map:
                charge = charge_map[key]
                line = (
                    line.ljust(80)[:78] + f"{abs(charge)}{'+' if charge >= 0 else '-'}"
                )
            records.append(line)
        elif linked and line.startswith("LINK  ") and line[21] == "A":
            line = (
                line[:12] + anchor.rjust(4) + line[16:42] + partner.rjust(4) + line[46:]
            )
            records.append(line)
    return read_structure_string("\n".join(records) + "\nEND\n", FileFormat.PDB)


def _linkage(structure):
    template = build_default_component_library().get("ARG")
    assert template is not None
    return (
        PolymerMicrostateContext(structure)
        .backbone_site(
            FIRST, template, PolymerChemicalSite.BACKBONE_N, assume_free_chain_ends=True
        )
        .linkage
    )


def test_linked_acetyl_has_three_methyl_h_and_realizes_source_amide_order():
    source = _source()
    assert _linkage(source) is PeptideLinkage.UNKNOWN
    source_bond = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(CAP, "C")),
        source.constitution.atom_index(AtomRef(FIRST, "N")),
    )
    assert source_bond is not None and source_bond.order is None

    result = add_retained_non_polymer_hydrogens(source)
    structure = result.structure
    index = structure.constitution.residue_index(CAP)
    residue = structure.constitution.residue_site_at(index)
    hydrogens = tuple(a.name for a in residue.atom_sites if a.is_hydrogen())
    assert len(hydrogens) == 3
    assert not result.issues
    assert set(
        structure.topology.residue_hydrogen_anchor_by_name(
            constitution=structure.constitution, residue_index=index
        ).values()
    ) == {"CH3"}
    bond = structure.topology.bond_between(
        structure.constitution.atom_index(AtomRef(CAP, "C")),
        structure.constitution.atom_index(AtomRef(FIRST, "N")),
    )
    assert bond is not None and bond.order == 1
    assert bond.source_metadata == source_bond.source_metadata
    assert bond.provenance == source_bond.provenance
    assert _linkage(structure) is PeptideLinkage.LINKED
    for old_index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old_index)
        new_index = structure.constitution.atom_index(ref)
        assert structure.constitution.atom_site_at(
            new_index
        ) == source.constitution.atom_site_at(old_index)
        assert structure.geometry.atom_geometry(
            new_index
        ) == source.geometry.atom_geometry(old_index)


def test_unlinked_acetyl_keeps_isolated_ccd_four_hydrogen_inventory():
    result = add_retained_non_polymer_hydrogens(_source(linked=False))
    assert not result.issues
    index = result.structure.constitution.residue_index(CAP)
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution, residue_index=index
    )
    assert sorted(anchors.values()) == ["C", "CH3", "CH3", "CH3"]


def test_acyl_binding_follows_graph_and_context_not_the_component_identifier():
    library = build_default_component_library()
    source = _source()
    index = source.constitution.residue_index(CAP)
    residue = source.constitution.residue_site_at(index)
    renamed = replace(residue, component_id="ALT")
    source = source.with_updated_residue_facets(
        renamed,
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    template = library.require("ACE")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    renamed_template = replace(
        template,
        definition=replace(template.definition, component_id="ALT"),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(
            replace(semantics.component, component_id="ALT")
        ),
    )
    custom = ComponentLibrary({**library.templates, "ALT": renamed_template})
    result = add_retained_non_polymer_hydrogens(source, custom)
    assert not result.issues
    cap = result.structure.constitution.residue_site_at(index)
    assert cap.component_id == "ALT"
    assert sum(atom.is_hydrogen() for atom in cap.atom_sites) == 3
    assert _linkage(result.structure) is PeptideLinkage.LINKED


def _with_bonds(source, bonds):
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


def _cap_with_occupied_nitrogen(relationship, provenance, *, realized=False):
    original = _source()
    if realized:
        original = add_retained_non_polymer_hydrogens(original).structure

    def payload(residue_id):
        index = original.constitution.residue_index(residue_id)
        return (
            original.constitution.residue_site_at(index),
            original.residue_geometry(index),
            original.residue_formal_charge_by_atom_name(index),
        )

    previous = ResidueId("A", -1)
    source = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=previous,
                        atoms=(atom_payload("C", "C", Vec3(-10, 0, 0)),),
                    ),
                    payload(FIRST),
                ),
            ),
        ),
        ligands=(payload(CAP),),
        source_format=FileFormat.PDB,
    )
    remapped = tuple(
        replace(
            bond,
            atom_index_1=source.constitution.atom_index(
                original.constitution.atom_ref_at(bond.atom_index_1)
            ),
            atom_index_2=source.constitution.atom_index(
                original.constitution.atom_ref_at(bond.atom_index_2)
            ),
        )
        for bond in original.topology.bonds
    )
    return _with_bonds(
        source,
        (
            *remapped,
            TopologyBond(
                source.constitution.atom_index(AtomRef(previous, "C")),
                source.constitution.atom_index(AtomRef(FIRST, "N")),
                order=1,
                relationship_type=relationship,
                provenance=provenance,
            ),
        ),
    )


@pytest.mark.parametrize("provenance", tuple(BondProvenance))
@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN),
)
def test_cap_refuses_an_already_occupied_polymer_nitrogen(relationship, provenance):
    source = _cap_with_occupied_nitrogen(relationship, provenance)
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, build_default_component_library())
        assert result.structure is source and not result.repairs
        assert any(
            issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
            for issue in result.issues
        )
    assert (
        _fact(source).component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )


def test_realized_cap_with_second_nitrogen_attachment_is_not_an_amide_boundary():
    source = _cap_with_occupied_nitrogen(
        BondRelationshipType.COVALENT,
        BondProvenance.SEQUENCE_INFERRED,
        realized=True,
    )
    assert not is_resolved_acyl_amide_link(
        source, AtomRef(CAP, "C"), AtomRef(FIRST, "N")
    )
    assert _linkage(source) is PeptideLinkage.UNKNOWN


@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.HYDROGEN_BOND, BondRelationshipType.METAL_COORDINATION),
)
def test_non_covalent_partner_relationship_does_not_occupy_the_cap_nitrogen(
    relationship,
):
    source = _cap_with_occupied_nitrogen(relationship, BondProvenance.SOURCE_EXPLICIT)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert is_resolved_acyl_amide_link(
        result.structure, AtomRef(CAP, "C"), AtomRef(FIRST, "N")
    )


def _fact(structure):
    return (
        StructureChemistryObservation(
            structure, component_library=build_default_component_library()
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )


def _with_source_cap_hydrogens(*, isotope="H", carbonyl=False, count=1):
    source = _source()
    generated = add_retained_non_polymer_hydrogens(
        _source(linked=not carbonyl)
    ).structure
    index = generated.constitution.residue_index(CAP)
    cap = generated.constitution.residue_site_at(index)
    anchors = generated.topology.residue_hydrogen_anchor_by_name(
        constitution=generated.constitution, residue_index=index
    )
    names = tuple(
        name
        for name, anchor in anchors.items()
        if anchor == ("C" if carbonyl else "CH3")
    )[:count]
    keep = {atom.name for atom in cap.atom_sites if not atom.is_hydrogen()} | set(names)
    cap = cap.without_atom_sites(set(cap.atom_site_names()) - keep)
    cap = cap.with_atom_sites(
        replace(atom, element=isotope) if atom.name in names else atom
        for atom in cap.atom_sites
    )
    updated = source.with_updated_residue_facets(
        cap,
        residue_geometry=generated.residue_geometry(index).without_atoms(
            set(anchors) - set(names)
        ),
    )
    # Read a deposited fixture: newly generated H is now original source evidence.
    return read_structure_string(
        write_structure_string(updated, FileFormat.MMCIF), FileFormat.MMCIF
    )


def test_default_acetyl_keeps_neutral_isolated_four_hydrogen_inventory():
    frozen = build_bundled_retained_non_polymer_registry().get("ACE")
    assert frozen is not None
    template = build_default_component_library().get("ACE")
    assert template is not None and len(template.expected_hydrogen_atom_names()) == 4
    assert sum(a.formal_charge for a in frozen.to_idealized_component().atoms) == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"anchor": "CH3"},
        {"partner": "CA"},
        {"partner": "NE"},
        {"missing": ("O",)},
        {"elements": (("O", "N"),)},
        {"charges": (((CAP, "C"), 1),)},
        {"charges": (((FIRST, "N"), 1),)},
    ),
)
def test_unsupported_or_conflicting_cap_application_is_atomic(kwargs):
    source = _source(**kwargs)
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(source, build_default_component_library())
        assert result.structure is source and not result.repairs
        assert any(
            i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION for i in result.issues
        )
    assert (
        _fact(source).component_support_state
        is ComponentSupportState.UNSUPPORTED_COMPONENTS_PRESENT
    )


@pytest.mark.parametrize(
    "relationship,order",
    (
        (BondRelationshipType.UNKNOWN, None),
        (BondRelationshipType.COVALENT, 2),
        (BondRelationshipType.DISULFIDE, 1),
    ),
)
def test_non_amide_relationship_or_order_cannot_trigger_substitution(
    relationship, order
):
    source = _source()
    bond = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(CAP, "C")),
        source.constitution.atom_index(AtomRef(FIRST, "N")),
    )
    assert bond is not None
    source = _with_bonds(
        source,
        tuple(
            replace(
                existing,
                relationship_type=relationship,
                order=order,
                source_metadata=None,
            )
            if existing == bond
            else existing
            for existing in source.topology.bonds
        ),
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and result.issues
    assert _linkage(source) is PeptideLinkage.UNKNOWN


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_carbonyl_source_h_cannot_be_removed_by_linked_template(isotope):
    source = _with_source_cap_hydrogens(isotope=isotope, carbonyl=True)
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and not result.repairs
    assert any(
        i.kind is ValidationIssueKind.UNSUPPORTED_HYDROGENATION for i in result.issues
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_compatible_source_methyl_h_identity_and_geometry_are_preserved(isotope):
    source = _with_source_cap_hydrogens(isotope=isotope)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = result.structure.constitution.atom_index(ref)
        assert result.structure.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
        assert result.structure.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(index)
    assert len(result.structure.constitution.ligands[0].atom_sites) == 6


def test_topology_only_realizes_complete_linked_inventory_and_is_idempotent():
    source = _with_source_cap_hydrogens(count=3)
    before = _fact(source)
    assert (
        before.heavy_atom_topology_availability_state
        is TopologyAvailabilityState.ABSENT
    )
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert not result.issues and result.repairs
    assert result.structure.constitution is source.constitution
    assert result.structure.geometry is source.geometry
    assert (
        _fact(result.structure).heavy_atom_topology_availability_state
        is TopologyAvailabilityState.PRESENT
    )
    assert _linkage(result.structure) is PeptideLinkage.LINKED
    repeated = resolve_retained_non_polymer_topology(
        result.structure, build_default_component_library()
    )
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


def test_explicit_isolated_smiles_override_refuses_a_linked_cap_without_fallback():
    source = _source()
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(
            RetainedNonPolymerChemistryEvidence(
                residue_id=CAP, smiles="CC=O", heavy_atom_names=("CH3", "C", "O")
            ),
        ),
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any("external" in issue.message for issue in result.issues)
    assert _linkage(result.structure) is PeptideLinkage.UNKNOWN


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_linked_cap_egress_roundtrip_preserves_amide_relationship(file_format):
    structure = add_retained_non_polymer_hydrogens(_source()).structure
    recovered = read_structure_string(
        write_structure_string(structure, file_format), file_format
    )
    bond = recovered.topology.bond_between(
        recovered.constitution.atom_index(AtomRef(CAP, "C")),
        recovered.constitution.atom_index(AtomRef(FIRST, "N")),
    )
    assert bond is not None and bond.relationship_type is BondRelationshipType.COVALENT
    # PDB LINK reports no order; its retained context is re-adjudicated on ingest.
    if file_format is FileFormat.MMCIF:
        assert bond.order == 1 and _linkage(recovered) is PeptideLinkage.LINKED
    rebuilt = add_retained_non_polymer_hydrogens(recovered).structure
    assert _linkage(rebuilt) is PeptideLinkage.LINKED
    assert len(rebuilt.constitution.ligands[0].atom_sites) == 6


def test_workflow_reobserves_realized_cap_before_selecting_linked_n_hydrogen():
    result = process_structure(
        _source(),
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    assert result.requested_goals_fully_satisfied()
    assert _linkage(result.structure) is PeptideLinkage.LINKED
    nitrogen = result.structure.constitution.atom_index(AtomRef(FIRST, "N"))
    local_h = tuple(
        index
        for bond in result.structure.topology.bonds
        if nitrogen in bond.endpoint_pair()
        for index in bond.endpoint_pair()
        if index != nitrogen
        and result.structure.constitution.atom_site_at(index).is_hydrogen()
    )
    assert len(local_h) == 1
    assert result.structure.topology.formal_charge(nitrogen) in (None, 0)


def _link(source):
    bond = source.topology.bond_between(
        source.constitution.atom_index(AtomRef(CAP, "C")),
        source.constitution.atom_index(AtomRef(FIRST, "N")),
    )
    assert bond is not None
    return bond


def test_source_link_reported_order_conflict_is_rejected_before_graph_commit():
    source = _source()
    link = _link(source)
    assert link.source_metadata is not None
    conflicted = replace(
        link, source_metadata=replace(link.source_metadata, reported_order=2)
    )
    source = _with_bonds(
        source, tuple(conflicted if b == link else b for b in source.topology.bonds)
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and not result.repairs
    assert any(
        i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION for i in result.issues
    )


def test_multiple_deposited_cap_attachments_are_not_partially_realized():
    source = _source()
    link = _link(source)
    extra = replace(
        link,
        atom_index_1=source.constitution.atom_index(AtomRef(CAP, "CH3")),
        atom_index_2=source.constitution.atom_index(AtomRef(FIRST, "N")),
        source_metadata=None,
    )
    source = _with_bonds(source, (*source.topology.bonds, extra))
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and result.issues and not result.repairs


@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.METAL_COORDINATION, BondRelationshipType.HYDROGEN_BOND),
)
def test_non_covalent_connection_is_not_an_acyl_substitution(relationship):
    source = _source()
    link = _link(source)
    changed = replace(link, relationship_type=relationship, source_metadata=None)
    source = _with_bonds(
        source, tuple(changed if b == link else b for b in source.topology.bonds)
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert len(result.structure.constitution.ligands[0].atom_sites) == 7
    assert _link(result.structure).relationship_type is relationship
    assert _linkage(result.structure) is not PeptideLinkage.LINKED


def test_noncanonical_source_methyl_h_keeps_isotope_scalars_and_explicit_charge():
    source = _with_source_cap_hydrogens(isotope="D")
    cap = source.constitution.ligands[0]
    name = next(a.name for a in cap.atom_sites if a.is_hydrogen())
    index = source.constitution.atom_index(AtomRef(CAP, name))
    cap = cap.without_atom_sites((name,)).with_atom_site(AtomSite("DX", "D"))
    geometry = (
        source.residue_geometry(source.constitution.residue_index(CAP))
        .without_atoms((name,))
        .with_atom_geometry(
            "DX",
            replace(
                source.geometry.atom_geometry(index), occupancy=0.75, b_factor=13.0
            ),
        )
    )
    source = source.with_updated_residue_facets(
        cap, residue_geometry=geometry, formal_charge_by_atom_name=(("DX", 0),)
    )
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    result = add_retained_non_polymer_hydrogens(source)
    old_index = source.constitution.atom_index(AtomRef(CAP, "DX"))
    new_index = result.structure.constitution.atom_index(AtomRef(CAP, "DX"))
    assert result.structure.constitution.atom_site_at(new_index).element == "D"
    assert result.structure.geometry.atom_geometry(
        new_index
    ) == source.geometry.atom_geometry(old_index)
    assert result.structure.topology.formal_charge(new_index) == 0


def test_reordered_source_slots_preserve_external_link_endpoints_after_h_rebuild():
    source = _source()
    cap = source.constitution.ligands[0]
    index = source.constitution.residue_index(CAP)
    source = source.with_updated_residue_facets(
        cap.reordered_atom_sites(tuple(reversed(cap.atom_site_names()))),
        residue_geometry=source.residue_geometry(index),
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues and _link(result.structure).order == 1
    assert _link(result.structure).source_metadata == _link(source).source_metadata
    assert _linkage(result.structure) is PeptideLinkage.LINKED


def test_cap_application_requires_supported_partner_template():
    source = _source()
    template = build_default_component_library().get("ACE")
    assert template is not None
    result = add_retained_non_polymer_hydrogens(
        source, ComponentLibrary(templates={"ACE": template})
    )
    assert result.structure is source and result.issues


def test_actual_graph_predicate_does_not_accept_absent_or_unresolved_amide():
    source = _source()
    assert not is_resolved_acyl_amide_link(
        source, AtomRef(CAP, "C"), AtomRef(FIRST, "N")
    )
    assert not is_resolved_acyl_amide_link(
        source, AtomRef(CAP, "MISSING"), AtomRef(FIRST, "N")
    )
    assert not is_resolved_acyl_amide_link(
        source, AtomRef(CAP, "C"), AtomRef(FIRST, "NE")
    )
    fixed = add_retained_non_polymer_hydrogens(source).structure
    assert is_resolved_acyl_amide_link(fixed, AtomRef(CAP, "C"), AtomRef(FIRST, "N"))
    link = _link(fixed)
    unlinked = _with_bonds(fixed, tuple(b for b in fixed.topology.bonds if b != link))
    assert not is_resolved_acyl_amide_link(
        unlinked, AtomRef(CAP, "C"), AtomRef(FIRST, "N")
    )


def test_original_polymer_n_overprotonation_is_not_erased_to_admit_cap():
    source = _source()
    index = source.constitution.residue_index(FIRST)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    n_position = geometry.position("N")
    for i, name in enumerate(("H1", "H2", "H3")):
        residue = residue.with_atom_site(AtomSite(name, "H"))
        geometry = geometry.with_atom_geometry(
            name,
            AtomGeometry(Vec3(n_position.x, n_position.y, n_position.z + 1 + i * 0.05)),
        )
    source = source.with_updated_residue_facets(residue, residue_geometry=geometry)
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source and result.issues


def test_topology_only_does_not_add_missing_methyl_h_or_resolve_half_a_link():
    source = _source()
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues
    assert _link(source).order is None


def test_cap_failure_is_local_to_its_residue_in_a_mixed_batch():
    records = []
    for line in PDB_PATH.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] in "AB" and int(line[22:26]) == 1:
            if line[21] == "B" and line[12:16].strip() == "N":
                line = line.ljust(80)[:78] + "1+"
            records.append(line)
        elif line.startswith("HETATM") and line[17:20] == "ACE":
            records.append(line)
        elif line.startswith("LINK  "):
            records.append(line)
    source = read_structure_string("\n".join(records) + "\nEND\n", FileFormat.PDB)
    failed_cap = ResidueId("B", 0)
    old_index = source.constitution.residue_index(failed_cap)
    result = add_retained_non_polymer_hydrogens(source)
    assert len(result.structure.constitution.ligands[0].atom_sites) == 6
    new_index = result.structure.constitution.residue_index(failed_cap)
    assert result.structure.constitution.residue_site_at(
        new_index
    ) == source.constitution.residue_site_at(old_index)
    assert result.structure.residue_geometry(new_index) == source.residue_geometry(
        old_index
    )
    assert any(
        i.residue_id == failed_cap
        and i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for i in result.issues
    )


def test_strict_fallback_mode_uses_contextual_template_without_native_inference():
    result = add_retained_non_polymer_hydrogens(
        _source(), allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    assert len(result.structure.constitution.ligands[0].atom_sites) == 6
    assert _linkage(result.structure) is PeptideLinkage.LINKED
