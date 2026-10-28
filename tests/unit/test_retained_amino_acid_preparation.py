"""Retained amino acids share state selection, not polymer membership."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import whole_structure_requested_goals
from tests.support.retained_reference_fixtures import reference_structure

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.chemistry.microstate.resolution import MicrostateSelectionBasis
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure,
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.observation import StructureObservation
from protrepair.structure.peptide import PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
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
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

LIGAND = ResidueId("L", 600)


@pytest.fixture(scope="module")
def protein() -> ProteinStructure:
    return read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb1afc.ent")


def _free_amino_acid(protein: ProteinStructure, component: str) -> ProteinStructure:
    template = build_default_component_library().require(component)
    if component in {"CSO", "HYP", "MSE", "PTR", "SEP", "TPO"}:
        source = reference_structure(component, with_h=False)
        residue = source.constitution.ligands[0]
    else:
        source = protein
        residue = next(
            residue
            for chain in protein.constitution.chains
            for residue in chain.residues
            if residue.component_id == component
            and set(template.expected_polymer_heavy_atom_names())
            <= set(residue.atom_site_names())
        )
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    atoms = tuple(
        atom_payload(
            atom.name, atom.element, geometry.position(atom.name), b_factor=23.0
        )
        for atom in residue.atom_sites
        if not atom.is_hydrogen()
    )
    polymer = build_bonded_structure(
        chains=(
            chain_payload(
                "L",
                (
                    residue_payload(
                        component_id=component, residue_id=LIGAND, atoms=atoms
                    ),
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    terminal = augment_c_terminal_oxt(polymer).structure
    terminal_geometry = terminal.residue_geometry(
        terminal.constitution.residue_index(LIGAND)
    )
    return build_bonded_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
        ligands=(
            residue_payload(
                component_id=component,
                residue_id=LIGAND,
                atoms=tuple(
                    atom_payload(
                        atom.name,
                        atom.element,
                        terminal_geometry.position(atom.name),
                        b_factor=23.0,
                    )
                    for atom in terminal.constitution.chains[0].residues[0].atom_sites
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "component",
    (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "CSO",
        "HYP",
        "MSE",
        "PTR",
        "SEP",
        "TPO",
    ),
)
def test_retained_amino_acid_shares_free_terminal_policy(protein, component):
    source = _free_amino_acid(protein, component)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    residue = result.structure.constitution.ligands[0]
    assert not result.structure.constitution.chains
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    counts = Counter(parents.values())
    assert counts["N"] == (2 if component in {"PRO", "HYP"} else 3)
    assert counts["OXT"] == 0
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
        )
        == 1
    )
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "OXT"))
        )
        == -1
    )
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert fact.is_supported()
    assert not fact.requires_hydrogen_completion()
    molecule = to_rdkit_molecule(result.structure)
    assert molecule.GetNumAtoms() == len(result.structure.constitution.atom_slots)
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    again = add_retained_non_polymer_hydrogens(result.structure)
    assert again.structure == result.structure
    assert not again.repairs
    for atom in source.constitution.ligands[0].atom_sites:
        ref = AtomRef(LIGAND, atom.name)
        assert result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))
    assert residue.component_id == component
    assert not AminoAcidPreparation(
        PolymerMicrostateContext(result.structure), build_default_component_library()
    ).has_pending_hydrogen_updates(LIGAND)


@pytest.mark.parametrize("component", ("TYR", "ASP", "PRO", "HIS", "MSE", "HYP"))
def test_public_h_goal_prepares_retained_amino_acid(protein, component):
    source = _free_amino_acid(protein, component)
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goals_fully_satisfied()
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in result.issues
    )
    assert not result.relaxation_executions
    to_rdkit_molecule(result.structure)


@pytest.mark.parametrize(
    "component,omitted", (("TYR", "OXT"), ("TYR", "OH"), ("MSE", "CE"))
)
def test_partial_scaffold_keeps_independent_site_progress(protein, component, omitted):
    source = _free_amino_acid(protein, component)
    index = source.constitution.residue_index(LIGAND)
    residue = source.constitution.residue_site_at(index).without_atom_sites((omitted,))
    source = source.with_updated_residue_facets_batch(
        ((residue, source.residue_geometry(index).without_atoms((omitted,)), ()),)
    )
    before = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert not before.hydrogen_generation_supported
    assert before.hydrogen_updates_pending and before.requires_hydrogen_completion()
    result = add_retained_non_polymer_hydrogens(source)
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == 3
    assert any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in result.issues
    )
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    again = add_retained_non_polymer_hydrogens(result.structure)
    assert again.structure == result.structure
    assert not again.repairs


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_source_terminal_h_identity_and_scalars_survive(protein, isotope):
    source = _free_amino_acid(protein, "TYR")
    index = source.constitution.residue_index(LIGAND)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    name = "H" if isotope == "H" else isotope
    h_geometry = AtomGeometry(
        Vec3(*(geometry.position("N").to_array() + (0.7, 0.7, 0.2))),
        occupancy=0.6,
        b_factor=31.0,
        altloc="B",
    )
    source = source.with_updated_residue_facets_batch(
        (
            (
                residue.with_atom_site(AtomSite(name, isotope)),
                geometry.with_atom_geometry(name, h_geometry),
                (),
            ),
        )
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
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    ref = AtomRef(LIGAND, name)
    assert result.structure.constitution.resolve_atom_site(
        ref
    ) == source.constitution.resolve_atom_site(ref)
    assert (
        result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        )
        == h_geometry
    )
    assert result.structure.provenance.ingress.observation is observation


def test_known_current_amine_charge_is_not_replaced_by_default(protein):
    source = _free_amino_acid(protein, "TYR")
    index = source.constitution.residue_index(LIGAND)
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index),
                source.residue_geometry(index),
                (("N", 0), ("O", 0), ("OXT", -1)),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == 2
    assert Counter(parents.values())["OXT"] == 0
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
        )
        == 0
    )
    to_rdkit_molecule(result.structure)


def test_unexplained_terminal_link_preserves_independent_hydrogens(protein):
    first = _free_amino_acid(protein, "TYR")
    second = _free_amino_acid(protein, "ALA")
    payloads = []
    for source, number in ((first, 600), (second, 900)):
        residue = source.constitution.ligands[0]
        geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
        payloads.append(
            residue_payload(
                component_id=residue.component_id,
                residue_id=ResidueId("L", number),
                atoms=tuple(
                    atom_payload(atom.name, atom.element, geometry.position(atom.name))
                    for atom in residue.atom_sites
                ),
            )
        )
    source = build_bonded_structure(
        chains=(), ligands=tuple(payloads), source_format=FileFormat.MMCIF
    )
    external = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND, "N")),
        source.constitution.atom_index(AtomRef(ResidueId("L", 900), "CB")),
        order=None,
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, external),
        ),
        provenance=source.provenance,
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,))
    )
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert "N" not in parents.values()
    assert "CA" in parents.values()
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert not fact.requires_hydrogen_completion()
    assert (
        len(
            [
                issue
                for issue in result.issues
                if issue.residue_id == LIGAND
                and issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
            ]
        )
        == 1
    )


def test_topology_only_does_not_generate_hydrogens(protein):
    source = _free_amino_acid(protein, "TYR")
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry


@pytest.mark.parametrize("component", ("HYP", "MSE", "SEP"))
def test_mirrored_reference_pose_cannot_be_prepared(protein, component):
    source = _free_amino_acid(protein, component)
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    for name, atom in geometry.atoms_by_name.items():
        position = atom.position
        geometry = geometry.with_atom_geometry(
            name, atom.with_position(Vec3(-position.x, position.y, position.z))
        )
    source = source.with_updated_residue_facets_batch(
        ((source.constitution.residue_site_at(index), geometry, ()),)
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert result.issues
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert not fact.requires_hydrogen_completion()


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("component", ("TYR", "MSE", "HYP"))
def test_retained_amino_acid_file_inputs_complete_without_role_rewrite(
    protein, component, file_format
):
    source = _free_amino_acid(protein, component)
    reread = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    assert not reread.constitution.chains
    result = process_structure(
        reread,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goals_fully_satisfied(), [
        issue.message for issue in result.issues
    ]
    to_rdkit_molecule(result.structure)
    for index, geometry in reread.geometry.iter_positions():
        ref = reread.constitution.atom_ref_at(index)
        assert (
            result.structure.geometry.position(
                result.structure.constitution.atom_index(ref)
            )
            == geometry
        )


def test_free_tyr_from_deposited_coordinates_is_prepared():
    text = (Path(__file__).parents[1] / "fixtures/corpus/pdb3cpa.ent").read_text()
    # This control deliberately declares a free occurrence: the original
    # GLY--TYR peptide LINK is not part of the extracted input.
    source = read_structure_string(
        "\n".join(
            line[:22] + " 600" + line[26:]
            for line in text.splitlines()
            if line.startswith("HETATM") and line[17:20] == "TYR"
        )
        + "\nEND\n",
        FileFormat.PDB,
    )
    assert len(source.constitution.ligands) == 1
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goals_fully_satisfied()
    to_rdkit_molecule(result.structure)


def test_missing_remote_atom_does_not_hide_confirmed_opposite_stereo(protein):
    source = _free_amino_acid(protein, "MSE")
    index = source.constitution.residue_index(LIGAND)
    residue = source.constitution.residue_site_at(index).without_atom_sites(("CE",))
    geometry = source.residue_geometry(index).without_atoms(("CE",))
    for name, atom in geometry.atoms_by_name.items():
        position = atom.position
        geometry = geometry.with_atom_geometry(
            name, atom.with_position(Vec3(-position.x, position.y, position.z))
        )
    source = source.with_updated_residue_facets_batch(((residue, geometry, ()),))
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert result.issues


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_partial_source_body_h_does_not_duplicate_a_generated_slot(protein, isotope):
    source = _free_amino_acid(protein, "TYR")
    complete = add_retained_non_polymer_hydrogens(source).structure
    first = complete.geometry.atom_geometry(
        complete.constitution.atom_index(AtomRef(LIGAND, "HB1"))
    )
    other = complete.geometry.atom_geometry(
        complete.constitution.atom_index(AtomRef(LIGAND, "HB2"))
    )
    index = source.constitution.residue_index(LIGAND)
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index).with_atom_site(
                    AtomSite("HB2", isotope)
                ),
                source.residue_geometry(index).with_atom_geometry("HB2", first),
                (),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert (
        result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "HB2"))
        )
        == first
    )
    assert (
        result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "HB1"))
        )
        == other
    )
    retained = result.structure.constitution.resolve_atom_site(AtomRef(LIGAND, "HB2"))
    assert retained is not None and retained.element == isotope


def test_declared_untyped_body_bond_is_resolved_without_losing_source_custody(protein):
    source = _free_amino_acid(protein, "CYS")
    first = source.constitution.atom_index(AtomRef(LIGAND, "CB"))
    second = source.constitution.atom_index(AtomRef(LIGAND, "SG"))
    pair = frozenset((first, second))
    untyped = TopologyBond(
        first,
        second,
        order=None,
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                bond
                for bond in source.topology.bonds
                if frozenset(bond.endpoint_pair()) != pair
            )
            + (untyped,),
        ),
        provenance=source.provenance,
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    first = result.structure.constitution.atom_index(AtomRef(LIGAND, "CB"))
    second = result.structure.constitution.atom_index(AtomRef(LIGAND, "SG"))
    bond = result.structure.topology.bond_between(first, second)
    assert bond is not None
    assert bond.relationship_type is BondRelationshipType.COVALENT and bond.order == 1
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    to_rdkit_molecule(result.structure)


def test_body_template_does_not_replace_the_selected_carboxyl_state(protein):
    source = _free_amino_acid(protein, "TYR")
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index).with_atom_site(
                    AtomSite("HO", "H")
                ),
                geometry.with_atom_geometry(
                    "HO",
                    AtomGeometry(
                        Vec3(*(geometry.position("O").to_array() + (0.7, 0.7, 0.2)))
                    ),
                ),
                (("N", 0), ("O", 0), ("OXT", 0)),
            ),
        )
    )
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND, "O")),
        source.constitution.atom_index(AtomRef(LIGAND, "HO")),
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, bond),
        ),
        provenance=source.provenance,
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    constitution = result.structure.constitution
    c = constitution.atom_index(AtomRef(LIGAND, "C"))
    oxygen = constitution.atom_index(AtomRef(LIGAND, "O"))
    oxt = constitution.atom_index(AtomRef(LIGAND, "OXT"))
    first = result.structure.topology.bond_between(c, oxygen)
    second = result.structure.topology.bond_between(c, oxt)
    assert first is not None and first.order == 1
    assert second is not None and second.order == 2
    to_rdkit_molecule(result.structure)


def test_body_placement_failure_does_not_discard_valid_terminal_sites(protein):
    source = _free_amino_acid(protein, "TYR")
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    position = Vec3(*(geometry.position("CB").to_array() + (3.0, 0.0, 0.0)))
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index).with_atom_site(
                    AtomSite("HB2", "H")
                ),
                geometry.with_atom_geometry("HB2", AtomGeometry(position)),
                (),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == 3
    assert (
        result.structure.geometry.position(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "HB2"))
        )
        == position
    )
    assert (
        len(
            [
                issue
                for issue in result.issues
                if issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
            ]
        )
        == 1
    )


def _amino_acid_with_neighbor(protein, endpoint, distance, *, polymer_neighbor):
    source = _free_amino_acid(protein, "ALA")
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    origin = geometry.position(endpoint)
    target = residue_payload(
        component_id="ALA",
        residue_id=LIGAND,
        atoms=tuple(
            atom_payload(
                atom.name,
                atom.element,
                geometry.position(atom.name).with_offset(
                    -origin.x, -origin.y, -origin.z
                ),
            )
            for atom in residue.atom_sites
        ),
    )
    opposite = "C" if endpoint == "N" else "N"
    neighbor = residue_payload(
        component_id="GLY",
        residue_id=ResidueId("P", 7),
        atoms=(atom_payload(opposite, opposite, Vec3(distance, 0.0, 0.0)),),
    )
    return build_bonded_structure(
        chains=(chain_payload("P", (neighbor,)),) if polymer_neighbor else (),
        ligands=(target,) if polymer_neighbor else (target, neighbor),
        source_format=FileFormat.MMCIF,
    )


@pytest.mark.parametrize("endpoint", ("N", "C"))
@pytest.mark.parametrize("polymer_neighbor", (False, True))
@pytest.mark.parametrize(
    "distance",
    (0.0, PEPTIDE_CN_DISTANCE_MAX_ANGSTROM, PEPTIDE_CN_DISTANCE_MAX_ANGSTROM + 0.1),
)
def test_unrecorded_peptide_neighbor_withdraws_only_free_terminal_assumption(
    protein, endpoint, polymer_neighbor, distance
):
    source = _amino_acid_with_neighbor(
        protein, endpoint, distance, polymer_neighbor=polymer_neighbor
    )
    expected_unknown = distance <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    kind = (
        PolymerChemicalSite.BACKBONE_N
        if endpoint == "N"
        else PolymerChemicalSite.BACKBONE_C
    )
    target = next(t for t in preparation.targets_for(LIGAND) if t.site.kind is kind)
    assert target.site.linkage is (
        PeptideLinkage.UNKNOWN if expected_unknown else PeptideLinkage.FREE
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((LIGAND,))
    )
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (fact.hydrogen_coverage_state is None) is expected_unknown
    if not expected_unknown:
        assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    else:
        parents = result.structure.topology.residue_hydrogen_anchor_by_name(
            constitution=result.structure.constitution,
            residue_index=result.structure.constitution.residue_index(LIGAND),
        )
        assert "CA" in parents.values()
        assert Counter(parents.values())["N"] == (0 if endpoint == "N" else 3)
        assert any(
            issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
            and issue.residue_id == LIGAND
            for issue in result.issues
        )
        charge_name = "N" if endpoint == "N" else "OXT"
        assert (
            result.structure.topology.formal_charge(
                result.structure.constitution.atom_index(AtomRef(LIGAND, charge_name))
            )
            is None
        )
    for bond in result.structure.topology.bonds:
        refs = tuple(
            result.structure.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
        )
        assert refs[0].residue_id == refs[1].residue_id
    for original in source.constitution.ligands:
        for atom in original.atom_sites:
            ref = AtomRef(original.residue_id, atom.name)
            assert result.structure.geometry.atom_geometry(
                result.structure.constitution.atom_index(ref)
            ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))


def test_unrecorded_peptide_neighbor_requires_geometry_rebinding(protein):
    source = _amino_acid_with_neighbor(protein, "N", 1.3, polymer_neighbor=False)
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    index = source.constitution.residue_index(ResidueId("P", 7))
    geometry = source.residue_geometry(index)
    moved = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index),
                geometry.with_atom_geometry("C", AtomGeometry(Vec3(10.0, 0.0, 0.0))),
                (),
            ),
        )
    )
    assert not preparation.matches_chemistry(moved)
    rebound = preparation.for_inputs(moved, preparation.component_library)
    target = next(
        t
        for t in rebound.targets_for(LIGAND)
        if t.site.kind is PolymerChemicalSite.BACKBONE_N
    )
    assert target.site.linkage is PeptideLinkage.FREE


@pytest.mark.parametrize("endpoint", ("N", "C"))
@pytest.mark.parametrize("wrong_target", (False, True))
def test_peptide_proximity_does_not_reinterpret_nonmatching_elements(
    protein, endpoint, wrong_target
):
    source = _amino_acid_with_neighbor(protein, endpoint, 1.3, polymer_neighbor=False)
    residue_id = LIGAND if wrong_target else ResidueId("P", 7)
    atom_name = endpoint if wrong_target else ("C" if endpoint == "N" else "N")
    index = source.constitution.residue_index(residue_id)
    residue = source.constitution.residue_site_at(index)
    source = source.with_updated_residue_facets_batch(
        (
            (
                residue.with_atom_site(
                    replace(residue.atom_site(atom_name), element="O")
                ),
                source.residue_geometry(index),
                (),
            ),
        )
    )
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    kind = (
        PolymerChemicalSite.BACKBONE_N
        if endpoint == "N"
        else PolymerChemicalSite.BACKBONE_C
    )
    target = next(t for t in preparation.targets_for(LIGAND) if t.site.kind is kind)
    assert target.site.linkage is PeptideLinkage.FREE
    assert (target.resolution.graph is None) is wrong_target


def _existing_neutral_amino_acid(protein, component):
    source = _free_amino_acid(protein, component)
    index = source.constitution.residue_index(LIGAND)
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index),
                source.residue_geometry(index),
                (("N", 0), ("OXT", 0)),
            ),
        )
    )
    carbon = source.constitution.atom_index(AtomRef(LIGAND, "C"))
    oxygen = source.constitution.atom_index(AtomRef(LIGAND, "O"))
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                replace(bond, order=2, provenance=BondProvenance.SOURCE_EXPLICIT)
                if set(bond.endpoint_pair()) == {carbon, oxygen}
                else bond
                for bond in source.topology.bonds
            ),
        ),
        provenance=source.provenance,
    )
    prepared = add_retained_non_polymer_hydrogens(source)
    assert not prepared.issues, [issue.message for issue in prepared.issues]
    text = write_structure_string(prepared.structure, FileFormat.PDB)
    lines = []
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = line[:78] + "  " + line[80:]
            if line[12:16].strip() == "H1":
                line = line[:12] + " H  " + line[16:]
        lines.append(line)
    return read_structure_string("\n".join(lines) + "\n", FileFormat.PDB)


@pytest.mark.parametrize(
    "component", ("ALA", "PRO", "ASP", "LYS", "ARG", "MSE", "HYP", "SEP")
)
def test_valid_existing_retained_state_precedes_generic_defaults(protein, component):
    source = _existing_neutral_amino_acid(protein, component)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    assert result.structure.constitution.atom_slots == source.constitution.atom_slots
    assert result.structure.geometry == source.geometry
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == (1 if component in {"PRO", "HYP"} else 2)
    assert Counter(parents.values())["OXT"] == 1
    nitrogen = result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
    assert result.structure.topology.formal_charge(nitrogen) == 0
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.formal_charge(AtomRef(LIGAND, "N")) is None
    choice = next(
        choice
        for choice in result.structure.provenance.microstate_choices
        if {atom.name for atom in choice.graph.atoms} == {"N"}
    )
    assert choice.basis is MicrostateSelectionBasis.DEFAULT
    molecule = to_rdkit_molecule(result.structure)
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    again = add_retained_non_polymer_hydrogens(result.structure)
    assert again.structure == result.structure
    assert not again.repairs


@pytest.mark.parametrize("component", ("ALA", "PRO", "MSE"))
def test_public_h_goal_keeps_valid_existing_amino_acid_state(protein, component):
    source = _existing_neutral_amino_acid(protein, component)
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure.constitution.atom_slots == source.constitution.atom_slots
    assert result.structure.geometry == source.geometry
    assert result.requested_goals_fully_satisfied()
    molecule = to_rdkit_molecule(result.structure)
    nitrogen = result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
    assert molecule.GetAtomWithIdx(nitrogen.value).GetFormalCharge() == 0


@pytest.mark.parametrize("original_charge", (False, True))
def test_declared_cation_can_complete_h_despite_existing_neutral_inventory(
    protein, original_charge
):
    source = _existing_neutral_amino_acid(protein, "ALA")
    index = source.constitution.residue_index(LIGAND)
    source = source.with_updated_residue_facets_batch(
        (
            (
                source.constitution.residue_site_at(index),
                source.residue_geometry(index),
                (("N", 1),),
            ),
        )
    )
    if original_charge:
        source = source.with_provenance(
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
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == 3
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
        )
        == 1
    )
    for atom in source.constitution.ligands[0].atom_sites:
        ref = AtomRef(LIGAND, atom.name)
        assert result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(source.constitution.atom_index(ref))


def test_existing_state_preference_accepts_api_input_without_observation(protein):
    source = _existing_neutral_amino_acid(protein, "ALA")
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                bond.with_resolved_chemistry(
                    replace(
                        bond, order=1, relationship_type=BondRelationshipType.COVALENT
                    )
                )
                if any(
                    source.constitution.atom_site_at(i).is_hydrogen()
                    for i in bond.endpoint_pair()
                )
                else bond
                for bond in source.topology.bonds
            ),
        ),
        provenance=source.provenance,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    assert result.structure is source
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    target = next(
        t
        for t in preparation.targets_for(LIGAND)
        if t.site.kind is PolymerChemicalSite.BACKBONE_N
    )
    assert target.resolution.graph is not None
    assert target.resolution.graph.atom("N").charge == 0
    molecule = to_rdkit_molecule(result.structure)
    nitrogen = source.constitution.atom_index(AtomRef(LIGAND, "N"))
    assert molecule.GetAtomWithIdx(nitrogen.value).GetFormalCharge() == 0


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_existing_state_preference_preserves_source_isotopes(protein, isotope):
    source = _existing_neutral_amino_acid(protein, "ALA")
    index = source.constitution.residue_index(LIGAND)
    residue = source.constitution.residue_site_at(index)
    for name in ("H", "H2"):
        residue = residue.with_atom_site(
            replace(residue.atom_site(name), element=isotope)
        )
    source = source.with_updated_residue_facets_batch(
        ((residue, source.residue_geometry(index), ()),)
    )
    source = source.with_provenance(
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
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure.constitution.atom_slots == source.constitution.atom_slots
    assert result.structure.geometry == source.geometry
    assert (
        result.structure.topology.formal_charge(
            result.structure.constitution.atom_index(AtomRef(LIGAND, "N"))
        )
        == 0
    )


def test_saved_amino_acid_choice_does_not_become_new_source_h_evidence(protein):
    source = add_retained_non_polymer_hydrogens(
        _free_amino_acid(protein, "ALA")
    ).structure
    index = source.constitution.residue_index(LIGAND)
    residue = source.constitution.residue_site_at(index).without_atom_sites(("H3",))
    source = source.with_updated_residue_facets_batch(
        (
            (
                residue,
                source.residue_geometry(index).without_atoms(("H3",)),
                source.residue_formal_charge_by_atom_name(index),
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues, [issue.message for issue in result.issues]
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert Counter(parents.values())["N"] == 3
