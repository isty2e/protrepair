"""Source coordination losses remain evidence, not invented canonical atoms."""

from dataclasses import replace
from pathlib import Path

import gemmi
import pytest
from tests.support.request_builders import ingress_options

from protrepair.api import process_structure
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.diagnostics.source_coordination import (
    diagnose_source_coordination_omissions,
)
from protrepair.errors import ModelInvariantError
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.source_connection import SourceAtomIdentity, SourceConnection
from protrepair.structure.topology import BondRelationshipType
from protrepair.workflow.contracts import LigandPolicy


def _raw_site(
    *,
    metal: str = "ZN",
    donor: str = "HOH",
    donor_chain: str = "A",
    metal_altloc: str = "",
    donor_altloc: str = "",
) -> gemmi.Structure:
    raw = gemmi.Structure()
    raw.name = "coordination"
    model = gemmi.Model(1)
    raw.add_model(model)
    model = raw[0]
    for chain_id in dict.fromkeys(("A", donor_chain)):
        model.add_chain(gemmi.Chain(chain_id))
    for component, chain_id, seq, atom_name, element, x, altloc in (
        (metal, "A", 1, metal, metal, 0.0, metal_altloc),
        (donor, donor_chain, 2, "O", "O", 2.13, donor_altloc),
    ):
        residue = gemmi.Residue()
        residue.name = component
        residue.seqid = gemmi.SeqId(seq, " ")
        residue.het_flag = "H"
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x, 0, 0)
        atom.occ = 1.0
        atom.b_iso = 20.0
        atom.charge = 2 if seq == 1 else 0
        atom.altloc = altloc or "\x00"
        residue.add_atom(atom)
        model[chain_id].add_residue(residue)

    connection = gemmi.Connection()
    connection.name = "site1"
    connection.type = gemmi.ConnectionType.MetalC
    connection.reported_distance = 2.13
    connection.partner1 = gemmi.AtomAddress(
        "A", gemmi.SeqId(1, " "), metal, metal, metal_altloc or "\x00"
    )
    connection.partner2 = gemmi.AtomAddress(
        donor_chain, gemmi.SeqId(2, " "), donor, "O", donor_altloc or "\x00"
    )
    raw.connections.append(connection)
    return raw


def _text(raw: gemmi.Structure, file_format: FileFormat) -> str:
    return (
        raw.make_pdb_string()
        if file_format is FileFormat.PDB
        else raw.make_mmcif_document().as_string()
    )


def _add_polymer_anchor(raw: gemmi.Structure) -> None:
    residue = gemmi.Residue()
    residue.name = "ALA"
    residue.seqid = gemmi.SeqId(10, " ")
    residue.het_flag = "A"
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.element = gemmi.Element("C")
    atom.pos = gemmi.Position(-10, 0, 0)
    residue.add_atom(atom)
    raw[0]["A"].add_residue(residue)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("metal", ("ZN", "CA", "FE"))
def test_filtered_water_evidence_does_not_change_retained_facets(
    file_format: FileFormat, metal: str
) -> None:
    raw = _raw_site(metal=metal)
    source = read_structure_string(_text(raw, file_format), file_format)
    observation = source.provenance.ingress.observation
    assert observation is not None
    evidence = observation.excluded_coordination_connections
    assert len(evidence) == 1
    assert evidence[0].source_metadata.reported_distance_angstrom == pytest.approx(2.13)
    assert (
        evidence[0].source_metadata.reported_relationship_type
        is BondRelationshipType.METAL_COORDINATION
    )
    assert {endpoint.component_id for endpoint in evidence[0].endpoint_pair()} == {
        metal,
        "HOH",
    }
    assert len(source.constitution.atom_slots) == 1
    assert not source.topology.bonds
    assert observation.topology.bonds == ()

    raw.connections.clear()
    without_declaration = read_structure_string(_text(raw, file_format), file_format)
    assert source.constitution == without_declaration.constitution
    assert source.geometry == without_declaration.geometry
    assert source.topology == without_declaration.topology
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(ResidueId("A", 1), metal))
        )
        == 2
    )
    issues = diagnose_source_coordination_omissions(source)
    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.SOURCE_COORDINATION_PARTNER_OMITTED
    assert issues[0].severity is IssueSeverity.WARNING
    assert f"HOH {AtomRef(ResidueId('A', 2), 'O').display_token()}" in issues[0].message
    assert not diagnose_source_coordination_omissions(without_declaration)
    with pytest.raises(ModelInvariantError, match="typed declaration"):
        replace(
            observation,
            excluded_coordination_connections=(
                replace(
                    evidence[0],
                    source_metadata=replace(
                        evidence[0].source_metadata,
                        reported_relationship_type=BondRelationshipType.UNKNOWN,
                    ),
                ),
            ),
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_retained_partner_is_not_an_ingress_omission(file_format: FileFormat) -> None:
    source = read_structure_string(
        _text(_raw_site(donor="LIG"), file_format), file_format
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.excluded_coordination_connections == ()
    assert len(observation.coordination_connections()) == 1
    assert len(source.topology.bonds) == 1
    assert not diagnose_source_coordination_omissions(source)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_public_process_reports_loss_after_file_and_canonical_reads(
    tmp_path: Path, file_format: FileFormat
) -> None:
    path = tmp_path / ("site.pdb" if file_format is FileFormat.PDB else "site.cif")
    path.write_text(_text(_raw_site(), file_format))
    source = read_structure(path)
    from_text = read_structure_string(path.read_text(), file_format)
    assert (
        source.provenance.ingress.observation
        == from_text.provenance.ingress.observation
    )
    for input_source in (path, source):
        result = process_structure(
            input_source, ingress=ingress_options(ligand_policy=LigandPolicy.KEEP)
        )
        assert result.structure.constitution == source.constitution
        assert result.structure.geometry == source.geometry
        assert result.structure.topology == source.topology
        issues = tuple(
            issue
            for issue in result.issues
            if issue.kind is ValidationIssueKind.SOURCE_COORDINATION_PARTNER_OMITTED
        )
        assert len(issues) == 1
        assert issues[0].severity is IssueSeverity.WARNING
        assert not result.has_errors()
        assert result.structure.provenance.ingress.observation == (
            source.provenance.ingress.observation
        )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_later_chain_filter_uses_original_retained_source_evidence(
    file_format: FileFormat,
) -> None:
    raw = _raw_site(donor="LIG", donor_chain="B")
    _add_polymer_anchor(raw)
    source = read_structure_string(_text(raw, file_format), file_format)
    reduced = apply_structure_normalization_policy(
        source, policy=StructureNormalizationPolicy(selected_chain_ids=("A",))
    )
    assert (
        reduced.provenance.ingress.observation is source.provenance.ingress.observation
    )
    assert not reduced.topology.bonds
    issues = diagnose_source_coordination_omissions(reduced)
    assert len(issues) == 1
    assert "original source identity" in issues[0].message
    assert "excluded during source normalization" not in issues[0].message
    result = process_structure(
        reduced, ingress=ingress_options(ligand_policy=LigandPolicy.KEEP)
    )
    assert len(diagnose_source_coordination_omissions(result.structure)) == 1


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_filtered_chain_records_actual_declared_partner_identity(
    file_format: FileFormat,
) -> None:
    source = read_structure_string(
        _text(_raw_site(donor="LIG", donor_chain="B"), file_format),
        file_format,
        policy=StructureNormalizationPolicy(selected_chain_ids=("A",)),
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert len(observation.excluded_coordination_connections) == 1
    assert len(diagnose_source_coordination_omissions(source)) == 1


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_removed_anchor_does_not_keep_a_stale_current_loss_warning(
    file_format: FileFormat,
) -> None:
    source = read_structure_string(_text(_raw_site(), file_format), file_format)
    empty = source.with_ligand_facets(ligand_sites=(), ligand_geometries=())
    assert empty.provenance.ingress.observation is source.provenance.ingress.observation
    assert not diagnose_source_coordination_omissions(empty)
    assert not tuple(
        issue
        for issue in process_structure(empty).issues
        if issue.kind is ValidationIssueKind.SOURCE_COORDINATION_PARTNER_OMITTED
    )


def test_current_restoration_does_not_rewrite_historical_omission() -> None:
    source = read_structure_string(
        _text(_raw_site(), FileFormat.MMCIF), FileFormat.MMCIF
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    original_evidence = observation.excluded_coordination_connections
    water_id = ResidueId("A", 2)
    water_site = ResidueSite("HOH", water_id, (AtomSite("O", "O"),), is_hetero=True)
    restored = source.with_ligand_facets(
        ligand_sites=(*source.constitution.ligands, water_site),
        ligand_geometries=(
            *(
                source.residue_geometry(source.constitution.residue_index(r.residue_id))
                for r in source.constitution.ligands
            ),
            ResidueGeometry({"O": AtomGeometry(position=Vec3(2.13, 0, 0))}),
        ),
    )
    assert restored.provenance.ingress.observation is observation
    assert observation.excluded_coordination_connections == original_evidence
    assert not diagnose_source_coordination_omissions(restored)
    assert not restored.topology.bonds


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("missing", ("atom", "component", "model"))
def test_missing_or_other_model_raw_partner_is_not_policy_exclusion(
    file_format: FileFormat, missing: str
) -> None:
    raw = _raw_site()
    if missing == "atom":
        del raw[0]["A"][1][0]
    elif missing == "component":
        raw[0]["A"][1].name = "DOD"
    else:
        second = gemmi.Model(2)
        chain = gemmi.Chain("A")
        chain.add_residue(raw[0]["A"][1])
        second.add_chain(chain)
        raw.add_model(second)
        del raw[0]["A"][1]
    source = read_structure_string(_text(raw, file_format), file_format)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.excluded_coordination_connections == ()
    assert not diagnose_source_coordination_omissions(source)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("which", ("metal", "donor"))
def test_unchosen_altloc_is_not_a_source_partner_omission(
    file_format: FileFormat, which: str
) -> None:
    raw = _raw_site(
        donor="LIG" if which == "donor" else "HOH",
        metal_altloc="A" if which == "metal" else "",
        donor_altloc="A" if which == "donor" else "",
    )
    residue = raw[0]["A"][0 if which == "metal" else 1]
    residue[0].occ = 0.1
    alternate = residue[0].clone()
    alternate.altloc = "B"
    alternate.occ = 0.9
    residue.add_atom(alternate)
    source = read_structure_string(_text(raw, file_format), file_format)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert observation.excluded_coordination_connections == ()
    assert not diagnose_source_coordination_omissions(source)


def test_unknown_conect_is_not_promoted_into_coordination_evidence() -> None:
    raw = _raw_site()
    raw.connections.clear()
    text = _text(raw, FileFormat.PDB).replace("END\n", "CONECT    1    2\nEND\n")
    source = read_structure_string(text, FileFormat.PDB)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert not observation.coordination_connections()
    assert not diagnose_source_coordination_omissions(source)


def test_off_graph_conflicting_declarations_preserve_both_reported_distances() -> None:
    raw = _raw_site()
    other = gemmi.Connection()
    other.name = "site2"
    other.type = gemmi.ConnectionType.MetalC
    other.partner1 = raw.connections[0].partner2
    other.partner2 = raw.connections[0].partner1
    other.reported_distance = 2.4
    raw.connections.append(other)
    document = raw.make_mmcif_document()
    distances = document.sole_block().find_values("_struct_conn.pdbx_dist_value")
    assert len(distances) == 2
    distances[1] = "2.4"
    source = read_structure_string(document.as_string(), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert {
        connection.source_metadata.reported_distance_angstrom
        for connection in observation.excluded_coordination_connections
    } == {2.13, 2.4}
    assert len(diagnose_source_coordination_omissions(source)) == 2
    assert not source.topology.bonds
    assert len(source.constitution.atom_slots) == 1


def test_coordinate_motion_preserves_source_boundary_evidence() -> None:
    source = read_structure_string(_text(_raw_site(), FileFormat.PDB), FileFormat.PDB)
    metal_id = ResidueId("A", 1)
    original = source.residue_geometry(source.constitution.residue_index(metal_id))
    moved = source.with_updated_residue_geometries(
        (
            (
                metal_id,
                original.with_atom_geometry(
                    "ZN", original.atom_geometry("ZN").with_position(Vec3(4, 5, 6))
                ),
            ),
        )
    )
    assert moved.provenance.ingress.observation is source.provenance.ingress.observation
    assert diagnose_source_coordination_omissions(
        moved
    ) == diagnose_source_coordination_omissions(source)


def test_egress_never_serializes_an_excluded_endpoint_as_a_bond() -> None:
    source = read_structure_string(
        _text(_raw_site(), FileFormat.MMCIF), FileFormat.MMCIF
    )
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        reparsed = read_structure_string(
            write_structure_string(source, file_format), file_format
        )
        assert reparsed.constitution == source.constitution
        assert reparsed.topology == source.topology
        assert not diagnose_source_coordination_omissions(reparsed)


def test_observation_rejects_off_graph_evidence_with_no_retained_endpoint() -> None:
    source = read_structure_string(
        _text(_raw_site(), FileFormat.MMCIF), FileFormat.MMCIF
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    connection = observation.excluded_coordination_connections[0]
    absent = SourceAtomIdentity(AtomRef(ResidueId("Z", 99), "FE"), "FE")
    with pytest.raises(ModelInvariantError, match="one selected endpoint"):
        replace(
            observation,
            excluded_coordination_connections=(
                SourceConnection(
                    absent, connection.endpoint_2, connection.source_metadata
                ),
            ),
        )


@pytest.mark.filterwarnings("error")
def test_excluded_water_numeric_payload_is_not_newly_validated() -> None:
    raw = _raw_site()
    water = raw[0]["A"][1][0]
    water.pos = gemmi.Position(float("nan"), float("inf"), 0)
    water.occ = -1.0
    water.b_iso = -5.0
    document = raw.make_mmcif_document()
    document.sole_block().find_values("_struct_conn.pdbx_dist_value")[0] = "2.13"
    source = read_structure_string(document.as_string(), FileFormat.MMCIF)
    assert len(source.constitution.atom_slots) == 1
    assert len(diagnose_source_coordination_omissions(source)) == 1


def test_excluded_water_does_not_invent_its_own_selected_conformer() -> None:
    raw = _raw_site(donor_altloc="A")
    raw[0]["A"][1][0].occ = 0.1
    alternate = raw[0]["A"][1][0].clone()
    alternate.altloc = "B"
    alternate.occ = 0.9
    raw[0]["A"][1].add_atom(alternate)
    source = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert len(observation.excluded_coordination_connections) == 1
    assert observation.excluded_coordination_connections[0].endpoint_2.altloc == "A"
    assert len(diagnose_source_coordination_omissions(source)) == 1


@pytest.mark.parametrize("which", ("metal", "donor"))
def test_inactive_component_variant_does_not_create_source_loss(which: str) -> None:
    raw = _raw_site(donor="LIG" if which == "donor" else "HOH")
    original = raw[0]["A"][0 if which == "metal" else 1]
    original[0].occ = 0.1
    alternate = original.clone()
    alternate.name = "MG" if which == "metal" else "ALT"
    alternate[0].occ = 0.9
    raw[0]["A"].add_residue(alternate)
    source = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert not observation.excluded_coordination_connections
    assert not diagnose_source_coordination_omissions(source)


def test_exact_duplicate_reversed_off_graph_records_do_not_multiply_evidence() -> None:
    raw = _raw_site()
    other = gemmi.Connection()
    other.name = raw.connections[0].name
    other.type = gemmi.ConnectionType.MetalC
    other.partner1 = raw.connections[0].partner2
    other.partner2 = raw.connections[0].partner1
    other.reported_distance = 2.13
    raw.connections.append(other)
    source = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert len(observation.excluded_coordination_connections) == 1
    assert len(diagnose_source_coordination_omissions(source)) == 1


def test_typed_nonmetal_pair_is_not_a_selected_metal_omission() -> None:
    raw = _raw_site(metal="C")
    source = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert not observation.excluded_coordination_connections
    assert not diagnose_source_coordination_omissions(source)


def test_unknown_metal_charge_remains_unknown_when_recording_source_loss() -> None:
    document = _raw_site().make_mmcif_document()
    document.sole_block().find_values("_atom_site.pdbx_formal_charge")[0] = "?"
    source = read_structure_string(document.as_string(), FileFormat.MMCIF)
    index = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "ZN"))
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert source.topology.formal_charge(index) is None
    assert observation.formal_charge(AtomRef(ResidueId("A", 1), "ZN")) is None
    assert len(diagnose_source_coordination_omissions(source)) == 1


def test_source_insertion_code_is_kept_in_excluded_partner_identity() -> None:
    raw = _raw_site()
    raw[0]["A"][1].seqid = gemmi.SeqId(2, "B")
    raw.connections[0].partner2 = gemmi.AtomAddress(
        "A", gemmi.SeqId(2, "B"), "HOH", "O"
    )
    source = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    observation = source.provenance.ingress.observation
    assert observation is not None
    evidence = observation.excluded_coordination_connections[0]
    assert evidence.endpoint_2.atom_ref.residue_id == ResidueId("A", 2, "B")
    assert len(diagnose_source_coordination_omissions(source)) == 1


def test_selected_donor_without_the_removed_metal_does_not_warn() -> None:
    raw = _raw_site(donor="LIG", donor_chain="B")
    source = read_structure_string(
        _text(raw, FileFormat.MMCIF),
        FileFormat.MMCIF,
        policy=StructureNormalizationPolicy(selected_chain_ids=("B",)),
    )
    observation = source.provenance.ingress.observation
    assert observation is not None
    assert not observation.excluded_coordination_connections
    assert not diagnose_source_coordination_omissions(source)


def test_source_record_order_preserves_retained_facets_and_warning_evidence() -> None:
    raw = _raw_site()
    reverse = gemmi.Connection()
    reverse.name = "site2"
    reverse.type = gemmi.ConnectionType.MetalC
    reverse.partner1 = raw.connections[0].partner2
    reverse.partner2 = raw.connections[0].partner1
    raw.connections.append(reverse)
    first = read_structure_string(_text(raw, FileFormat.MMCIF), FileFormat.MMCIF)
    reversed_raw = _raw_site()
    reversed_raw.connections.clear()
    reversed_raw.connections.append(reverse)
    reversed_raw.connections.append(raw.connections[0])
    second = read_structure_string(
        _text(reversed_raw, FileFormat.MMCIF), FileFormat.MMCIF
    )
    assert first.constitution == second.constitution
    assert first.geometry == second.geometry
    assert first.topology == second.topology
    assert set(diagnose_source_coordination_omissions(first)) == set(
        diagnose_source_coordination_omissions(second)
    )
