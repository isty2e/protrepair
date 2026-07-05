"""Focused tests for the gemmi-backed I/O boundary."""

import stat
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import ingress_options
from tests.support.structure_summary import summarize_structure

import protrepair.io.gemmi_ingress as gemmi_ingress
import protrepair.io.gemmi_writer as gemmi_writer
from protrepair.diagnostics.topology import detect_disulfide_topology
from protrepair.errors import StructureInputTooLargeError, StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure,
    read_structure_string,
    write_structure,
    write_structure_string,
)
from protrepair.io.ingress_policy import (
    LigandHandling,
    StructureNormalizationPolicy,
)
from protrepair.io.structure_ingress import apply_structure_normalization_policy
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
    is_covalent_like_relationship,
    is_source_provenance,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    MutationPolicy,
    OccupancyPolicy,
)


def test_read_structure_string_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """String ingress should bound hostile in-memory coordinate payloads."""

    monkeypatch.setattr(gemmi_ingress, "MAX_STRUCTURE_INPUT_BYTES", 4)

    with pytest.raises(StructureInputTooLargeError, match="exceeds 4 characters"):
        read_structure_string("HEADER", FileFormat.PDB, source_name="oversized.pdb")


def test_read_structure_rejects_oversized_file_before_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path ingress should stat oversized coordinate files before gemmi parsing."""

    pdb_path = tmp_path / "oversized.pdb"
    pdb_path.write_text("HEADER", encoding="utf-8")
    monkeypatch.setattr(gemmi_ingress, "MAX_STRUCTURE_INPUT_BYTES", 4)

    with pytest.raises(StructureInputTooLargeError, match="exceeds 4 bytes"):
        read_structure(pdb_path)


def test_read_structure_string_rejects_nonfinite_pdb_coordinate() -> None:
    """PDB ingress should reject non-finite coordinates before canonical geometry."""

    with pytest.raises(StructureNormalizationError, match="non-finite x coordinate"):
        read_structure_string(
            build_pdb_text(
                [
                    build_pdb_atom_line(
                        serial=1,
                        atom_name=" N  ",
                        residue_name="GLY",
                        chain_id="A",
                        residue_seq=1,
                        x=float("nan"),
                        element="N",
                    ),
                    "END",
                ]
            ),
            FileFormat.PDB,
        )


def test_read_structure_string_rejects_nonfinite_mmcif_coordinate() -> None:
    """mmCIF ingress should share the same finite-coordinate contract as PDB."""

    with pytest.raises(StructureNormalizationError, match="non-finite x coordinate"):
        read_structure_string(
            """data_numeric
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 ? nan 2.000 3.000 1.00 20.00 1 A 1
#
""",
            FileFormat.MMCIF,
        )


@pytest.mark.parametrize("occupancy", [-0.01, 1.01])
def test_read_structure_string_rejects_out_of_range_occupancy(
    occupancy: float,
) -> None:
    """Ingress occupancy must stay within the accepted PDB/mmCIF range."""

    with pytest.raises(StructureNormalizationError, match="occupancy"):
        read_structure_string(
            build_pdb_text(
                [
                    build_pdb_atom_line(
                        serial=1,
                        atom_name=" N  ",
                        residue_name="GLY",
                        chain_id="A",
                        residue_seq=1,
                        occupancy=occupancy,
                        element="N",
                    ),
                    "END",
                ]
            ),
            FileFormat.PDB,
        )


def test_read_structure_string_rejects_negative_b_factor() -> None:
    """Ingress B factors must be finite and non-negative."""

    with pytest.raises(StructureNormalizationError, match="B factor"):
        read_structure_string(
            build_pdb_text(
                [
                    build_pdb_atom_line(
                        serial=1,
                        atom_name=" N  ",
                        residue_name="GLY",
                        chain_id="A",
                        residue_seq=1,
                        b_factor=-0.01,
                        element="N",
                    ),
                    "END",
                ]
            ),
            FileFormat.PDB,
        )


def test_read_structure_string_rejects_nonfinite_occupancy_and_b_factor() -> None:
    """Ingress should reject non-finite scalar atom geometry values."""

    with pytest.raises(StructureNormalizationError, match="non-finite occupancy"):
        read_structure_string(
            build_pdb_text(
                [
                    build_pdb_atom_line(
                        serial=1,
                        atom_name=" N  ",
                        residue_name="GLY",
                        chain_id="A",
                        residue_seq=1,
                        occupancy=float("nan"),
                        element="N",
                    ),
                    "END",
                ]
            ),
            FileFormat.PDB,
        )

    with pytest.raises(StructureNormalizationError, match="non-finite B factor"):
        read_structure_string(
            build_pdb_text(
                [
                    build_pdb_atom_line(
                        serial=1,
                        atom_name=" N  ",
                        residue_name="GLY",
                        chain_id="A",
                        residue_seq=1,
                        b_factor=float("inf"),
                        element="N",
                    ),
                    "END",
                ]
            ),
            FileFormat.PDB,
        )


def test_read_structure_string_accepts_numeric_boundary_values() -> None:
    """Ingress validation should not reject valid occupancy and B-factor limits."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    occupancy=0.00,
                    b_factor=0.00,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=1.00,
                    b_factor=999.99,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId("A", 1)
    n_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "N"))
    )
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )

    assert (n_geometry.occupancy, n_geometry.b_factor) == (0.0, 0.0)
    assert ca_geometry.occupancy == 1.0
    assert ca_geometry.b_factor == pytest.approx(999.99)


def test_read_structure_string_uses_first_model_for_multimodel_pdb() -> None:
    """PDB ingress should materialize only the first source model."""

    structure = read_structure_string(
        build_pdb_text(
            [
                "MODEL        1",
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    element="N",
                ),
                "ENDMDL",
                "MODEL        2",
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=9.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=8.0,
                    element="C",
                ),
                "ENDMDL",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "N")
    )

    assert structure.geometry.atom_geometry(atom_index).position.x == 1.0
    assert structure.constitution.resolve_atom_site(
        AtomRef(ResidueId("A", 1), "CA")
    ) is None


def test_read_structure_string_uses_first_model_for_multimodel_mmcif() -> None:
    """mmCIF ingress should match the same first-model-only contract as PDB."""

    structure = read_structure_string(
        """data_multimodel
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . GLY A 1 1 ? 1.000 2.000 3.000 1.00 20.00 1 A 1
ATOM 2 N N . GLY A 1 1 ? 9.000 8.000 7.000 1.00 20.00 1 A 2
ATOM 3 C CA . GLY A 1 1 ? 8.000 7.000 6.000 1.00 20.00 1 A 2
#
""",
        FileFormat.MMCIF,
    )

    atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "N")
    )

    assert structure.geometry.atom_geometry(atom_index).position == Vec3(
        x=1.0,
        y=2.0,
        z=3.0,
    )
    assert structure.constitution.resolve_atom_site(
        AtomRef(ResidueId("A", 1), "CA")
    ) is None


def test_read_structure_string_empty_first_model_does_not_fall_through() -> None:
    """An empty first model should not be replaced by a populated later model."""

    structure = read_structure_string(
        build_pdb_text(
            [
                "MODEL        1",
                "ENDMDL",
                "MODEL        2",
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=9.0,
                    element="N",
                ),
                "ENDMDL",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    assert structure.constitution.chains == ()
    assert structure.geometry.atom_geometries == ()


def test_read_structure_string_resolves_atom_altloc_by_highest_occupancy() -> None:
    """Altloc sites should normalize to one atom by occupancy policy."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    atom_ref = AtomRef(residue_id=residue_id, atom_name="CA")
    atom_site = structure.constitution.resolve_atom_site(atom_ref)
    atom_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(atom_ref)
    )

    assert atom_site is not None
    assert atom_geometry.altloc == "B"
    assert atom_geometry.position == Vec3(x=4.0, y=5.0, z=6.0)


def test_read_structure_string_can_select_lowest_occupancy_atom_site() -> None:
    """Occupancy policy should affect atom-site normalization."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.30,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.70,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            occupancy_policy=OccupancyPolicy.LOWEST
        ).structure_normalization_policy(),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    atom_ref = AtomRef(residue_id=residue_id, atom_name="CA")
    atom_site = structure.constitution.resolve_atom_site(atom_ref)
    atom_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(atom_ref)
    )

    assert atom_site is not None
    assert atom_geometry.altloc == "A"
    assert atom_geometry.position == Vec3(x=1.0, y=2.0, z=3.0)


def test_read_structure_string_selects_coherent_residue_altloc_cohort() -> None:
    """Residue altloc normalization should not mix atoms from different cohorts."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.60,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.40,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.40,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.60,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=5,
                    atom_name=" CB ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=5.0,
                    occupancy=1.00,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    n_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "N"))
    )
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )
    cb_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CB"))
    )

    assert (n_geometry.altloc, n_geometry.position.x) == ("A", 1.0)
    assert (ca_geometry.altloc, ca_geometry.position.x) == ("A", 3.0)
    assert (cb_geometry.altloc, cb_geometry.position.x) == (None, 5.0)


def test_read_structure_string_lowest_policy_selects_residue_altloc_cohort() -> None:
    """LOWEST occupancy policy should select one residue altloc cohort coherently."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.20,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            occupancy_policy=OccupancyPolicy.LOWEST
        ).structure_normalization_policy(),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    n_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "N"))
    )
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )

    assert (n_geometry.altloc, n_geometry.position.x) == ("B", 3.0)
    assert (ca_geometry.altloc, ca_geometry.position.x) == ("B", 4.0)


def test_read_structure_string_altloc_tie_uses_first_seen_cohort() -> None:
    """Altloc cohort ties should be deterministic and not lexical-order based."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.50,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.50,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.50,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="SER",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.50,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    n_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "N"))
    )
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )

    assert (n_geometry.altloc, n_geometry.position.x) == ("B", 1.0)
    assert (ca_geometry.altloc, ca_geometry.position.x) == ("B", 2.0)


def test_read_structure_string_resolves_duplicates_inside_selected_altloc_cohort() -> (
    None
):
    """Duplicate atom names inside a selected cohort should still use occupancy."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.20,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" CB ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.10,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )

    assert (ca_geometry.altloc, ca_geometry.position.x) == ("A", 2.0)


def test_read_structure_string_selected_altloc_overrides_blank_duplicate_atom() -> None:
    """A cohort-specific atom should win over a blank duplicate for the same atom."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    occupancy=1.00,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.40,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.20,
                    element="N",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    ca_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(residue_id, "CA"))
    )

    assert (ca_geometry.altloc, ca_geometry.position.x) == ("A", 3.0)


def test_read_structure_string_can_select_residue_variant_by_mutation_policy() -> None:
    """Duplicate residue ids should normalize by residue occupancy score."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.5,
                    y=2.5,
                    z=3.5,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.5,
                    y=5.5,
                    z=6.5,
                    occupancy=0.20,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    lowest_structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    occupancy=0.80,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    altloc="A",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=1,
                    x=1.5,
                    y=2.5,
                    z=3.5,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    atom_name=" CA ",
                    altloc="B",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=4.5,
                    y=5.5,
                    z=6.5,
                    occupancy=0.20,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            mutation_policy=MutationPolicy.LOWEST_OCCUPANCY,
        ).structure_normalization_policy(),
    )

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure_residue = structure.constitution.residue_or_ligand(residue_id)
    lowest_structure_residue = lowest_structure.constitution.residue_or_ligand(
        residue_id
    )
    assert structure_residue is not None
    assert lowest_structure_residue is not None
    assert structure_residue.component_id == "ALA"
    assert lowest_structure_residue.component_id == "GLY"


def test_read_structure_string_selects_duplicate_ligand_residue_variant() -> None:
    """Ligand microheterogeneity should resolve before canonical duplicate ids."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.20,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" N1 ",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.20,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="NAD",
                    chain_id="L",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=4,
                    record_name="HETATM",
                    atom_name=" N1 ",
                    residue_name="NAD",
                    chain_id="L",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.80,
                    element="N",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    assert len(structure.constitution.ligands) == 1
    ligand = structure.constitution.ligands[0]
    ligand_geometry = structure.residue_geometry(
        structure.constitution.residue_index(ligand.residue_id)
    )

    assert ligand.component_id == "NAD"
    assert ligand_geometry.atom_geometry("C1").position.x == 3.0
    assert ligand_geometry.atom_geometry("N1").position.x == 4.0


def test_read_structure_string_selects_ligand_altloc_cohort_without_mixing() -> None:
    """Retained ligand altloc normalization should also use one coherent cohort."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    altloc="A",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=1.0,
                    occupancy=0.60,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    altloc="B",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    occupancy=0.40,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" N1 ",
                    altloc="A",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=3.0,
                    occupancy=0.40,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=4,
                    record_name="HETATM",
                    atom_name=" N1 ",
                    altloc="B",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=4.0,
                    occupancy=0.60,
                    element="N",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    ligand = structure.constitution.ligands[0]
    ligand_geometry = structure.residue_geometry(
        structure.constitution.residue_index(ligand.residue_id)
    )

    assert ligand.component_id == "FAD"
    assert (
        ligand_geometry.atom_geometry("C1").altloc,
        ligand_geometry.atom_geometry("C1").position.x,
    ) == ("A", 1.0)
    assert (
        ligand_geometry.atom_geometry("N1").altloc,
        ligand_geometry.atom_geometry("N1").position.x,
    ) == ("A", 3.0)


def test_read_structure_string_can_filter_chains_and_drop_ligands() -> None:
    """Boundary options should normalize chain and ligand selection at ingress."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="B",
                    residue_seq=2,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="B",
                    residue_seq=3,
                    x=7.0,
                    y=8.0,
                    z=9.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=StructureNormalizationPolicy(
            ligand_handling=LigandHandling.DROP,
            selected_chain_ids=("B",),
        ),
    )

    assert structure.chain_ids() == ("B",)
    assert structure.constitution.ligands == ()


def test_read_structure_string_keeps_non_water_ligands_only() -> None:
    """Ligand retention should exclude water from the ligand bucket."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O  ",
                    residue_name="HOH",
                    chain_id="A",
                    residue_seq=2,
                    x=4.0,
                    y=5.0,
                    z=6.0,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="A",
                    residue_seq=3,
                    x=7.0,
                    y=8.0,
                    z=9.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    assert structure.chain_ids() == ("A",)
    assert tuple(
        residue.component_id for residue in structure.constitution.ligands
    ) == ("FAD",)


def test_read_structure_string_surfaces_source_explicit_inter_residue_bonds() -> None:
    """PDB LINK records should become source-explicit topology bonds."""

    structure = read_structure_string(
        build_2q6f_linked_pdb_text(),
        FileFormat.PDB,
    )

    bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
        and bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_LINK
    )
    assert tuple(_bond_display_token(structure, bond) for bond in bonds) == (
        "D:4.C-D:5.N",
        "D:5.C-D:6.O",
    )
    assert tuple(bond.relationship_type for bond in bonds) == (
        BondRelationshipType.COVALENT,
        BondRelationshipType.COVALENT,
    )
    absent_pair = frozenset(
        (
            structure.constitution.resolve_atom_index(
                AtomRef(ResidueId("A", 143), "SG")
            ),
            structure.constitution.resolve_atom_index(
                AtomRef(ResidueId("D", 5), "C20")
            ),
        )
    )
    assert not any(
        frozenset((bond.atom_index_1, bond.atom_index_2)) == absent_pair
        for bond in bonds
    )


def test_read_structure_string_surfaces_pdb_conect_inter_residue_bonds() -> None:
    """PDB CONECT records should become source-explicit topology bonds."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    2",
                "CONECT    2    1",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    assert structure.topology.bonds == (
        TopologyBond(
            atom_index_1=structure.constitution.atom_index(
                AtomRef(ResidueId("A", 1), "C1")
            ),
            atom_index_2=structure.constitution.atom_index(
                AtomRef(ResidueId("L", 1), "O1")
            ),
            relationship_type=BondRelationshipType.UNKNOWN,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.PDB_CONECT,
                source_id="CONECT",
            ),
        ),
    )


def test_read_structure_string_keeps_conect_for_selected_altloc_atom() -> None:
    """CONECT endpoints should survive when their source altloc was selected."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    altloc="A",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" C1 ",
                    altloc="B",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    occupancy=0.20,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    3",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    assert tuple(
        bond.source_metadata.record_type
        for bond in structure.topology.bonds
        if bond.source_metadata is not None
    ) == (SourceBondRecordType.PDB_CONECT,)


def test_read_structure_string_drops_conect_from_discarded_altloc_atom() -> None:
    """CONECT from an unselected altloc atom must not bind to selected AtomRef."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    altloc="A",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" C1 ",
                    altloc="B",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    occupancy=0.20,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    2    3",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    assert not any(
        bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        for bond in structure.topology.bonds
    )


def test_read_structure_string_drops_conect_from_discarded_ligand_variant() -> None:
    """CONECT from discarded ligand microheterogeneity should not survive."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    occupancy=0.20,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="NAD",
                    chain_id="L",
                    residue_seq=1,
                    occupancy=0.80,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=3,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    element="N",
                ),
                "CONECT    1    3",
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    assert structure.constitution.ligands[0].component_id == "NAD"
    assert not any(
        bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        for bond in structure.topology.bonds
    )


def test_read_structure_string_drops_conect_with_reused_multimodel_serials() -> None:
    """CONECT with serials reused across models should be treated ambiguous."""

    structure = read_structure_string(
        build_pdb_text(
            [
                "MODEL        1",
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "ENDMDL",
                "MODEL        2",
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    x=9.0,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=8.0,
                    element="O",
                ),
                "ENDMDL",
                "CONECT    1    2",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    assert not any(
        bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        for bond in structure.topology.bonds
    )


def test_apply_structure_normalization_policy_drops_links_with_missing_endpoints() -> (
    None
):
    """Source links should be filtered when normalization removes their endpoints."""

    structure = read_structure_string(
        build_2q6f_linked_pdb_text(),
        FileFormat.PDB,
    )
    normalized = apply_structure_normalization_policy(
        structure,
        policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.DROP),
    )

    assert not any(
        bond.provenance is BondProvenance.SOURCE_EXPLICIT
        for bond in normalized.topology.bonds
    )


def test_read_structure_string_keeps_proximity_disulfide_out_of_topology() -> None:
    """Default ingress must not promote SG-SG proximity into topology truth."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" SG ",
                    residue_name="CYS",
                    chain_id="A",
                    residue_seq=1,
                    element="S",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" SG ",
                    residue_name="CYS",
                    chain_id="A",
                    residue_seq=2,
                    x=3.05,
                    element="S",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    likely_disulfides, ambiguous_disulfides = detect_disulfide_topology(structure)

    assert len(likely_disulfides) == 1
    assert ambiguous_disulfides == ()
    assert not any(
        bond.relationship_type is BondRelationshipType.DISULFIDE
        for bond in structure.topology.bonds
    )


def test_read_structure_string_seeds_topology_bonds_from_pdb_link() -> None:
    """PDB LINK records should seed TopologyBond entries with SOURCE_EXPLICIT."""

    structure = read_structure_string(
        build_2q6f_linked_pdb_text(),
        FileFormat.PDB,
    )

    bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    assert len(bonds) == 2
    assert all(bond.provenance is BondProvenance.SOURCE_EXPLICIT for bond in bonds)
    assert all(
        bond.relationship_type is BondRelationshipType.COVALENT for bond in bonds
    )
    assert all(
        bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_LINK
        for bond in bonds
    )
    assert all(is_covalent_like_relationship(bond) for bond in bonds)
    assert all(is_source_provenance(bond) for bond in bonds)


def test_read_structure_string_seeds_topology_bonds_from_pdb_conect() -> None:
    """PDB CONECT records should seed TopologyBond entries as UNKNOWN type."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C1 ",
                    residue_name="MOV",
                    chain_id="A",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    2",
                "CONECT    2    1",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    bonds = structure.topology.bonds
    assert len(bonds) == 1
    bond = bonds[0]
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.relationship_type is BondRelationshipType.UNKNOWN
    assert bond.source_metadata is not None
    assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
    assert bond.source_metadata.source_id == "CONECT"
    assert not is_covalent_like_relationship(bond)


def test_read_structure_string_seeds_template_resolved_present_atom_bonds() -> None:
    """Known component templates should seed bonds only for present atoms."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" CA ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    x=2.0,
                    element="C",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    bond_tokens = {
        frozenset(
            (
                structure.constitution.atom_ref_at(bond.atom_index_1).display_token(),
                structure.constitution.atom_ref_at(bond.atom_index_2).display_token(),
            )
        )
        for bond in structure.topology.bonds
    }

    assert bond_tokens == {frozenset(("A:1.N", "A:1.CA"))}
    assert structure.topology.bonds[0].provenance is BondProvenance.TEMPLATE_RESOLVED


def test_read_structure_string_seeds_retained_non_polymer_template_bonds() -> None:
    """Known retained non-polymer/cofactor templates should seed topology bonds."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" PA ",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    element="P",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1A",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    assert any(
        bond.provenance is BondProvenance.TEMPLATE_RESOLVED
        for bond in structure.topology.bonds
    )


def test_read_structure_string_seeds_sequence_inferred_peptide_bonds() -> None:
    """Adjacent polymer residues should seed sequence-inferred peptide bonds."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" C  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    atom_name=" N  ",
                    residue_name="ALA",
                    chain_id="A",
                    residue_seq=2,
                    x=2.0,
                    element="N",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    peptide_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SEQUENCE_INFERRED
    )

    assert len(peptide_bonds) == 1
    assert {
        structure.constitution.atom_ref_at(peptide_bonds[0].atom_index_1),
        structure.constitution.atom_ref_at(peptide_bonds[0].atom_index_2),
    } == {
        AtomRef(ResidueId("A", 1), "C"),
        AtomRef(ResidueId("A", 2), "N"),
    }


def test_read_structure_string_source_bonds_override_template_bonds() -> None:
    """Source-explicit bonds should win over template bonds on same endpoints."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" PA ",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    element="P",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1A",
                    residue_name="FAD",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    2",
                "END",
            ]
        ),
        FileFormat.PDB,
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    assert len(structure.topology.bonds) == 1
    assert structure.topology.bonds[0].provenance is BondProvenance.SOURCE_EXPLICIT


def test_read_structure_string_seeds_intra_residue_conect_bonds() -> None:
    """Intra-residue PDB CONECT bonds should enter topology.bonds."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="M",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    2",
                "CONECT    2    1",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    bonds = structure.topology.bonds
    assert len(bonds) == 1
    bond = bonds[0]
    assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert bond.relationship_type is BondRelationshipType.UNKNOWN
    assert bond.source_metadata is not None
    assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT


def test_read_structure_string_link_takes_precedence_over_conect() -> None:
    """When LINK and CONECT cover the same endpoint pair, LINK wins."""

    structure = read_structure_string(
        build_2q6f_linked_pdb_text(),
        FileFormat.PDB,
    )
    link_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    assert len(link_bonds) == 2

    pdb_text = build_2q6f_linked_pdb_text()
    atom_lines = [
        line for line in pdb_text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    serials_by_name: dict[tuple[str, str, int], int] = {}
    for line in atom_lines:
        serial = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        chain_id = line[21:22]
        seq_num = int(line[22:26].strip())
        serials_by_name[(chain_id, atom_name, seq_num)] = serial

    conect_lines: list[str] = []
    for bond in link_bonds:
        ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
        ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
        s1 = serials_by_name.get(
            (
                ref_1.residue_id.chain_id,
                ref_1.atom_name,
                ref_1.residue_id.seq_num,
            )
        )
        s2 = serials_by_name.get(
            (
                ref_2.residue_id.chain_id,
                ref_2.atom_name,
                ref_2.residue_id.seq_num,
            )
        )
        if s1 is not None and s2 is not None:
            conect_lines.append(f"CONECT{s1:>5}{s2:>5}")

    augmented_text = pdb_text.replace("END", "\n".join(conect_lines) + "\nEND", 1)
    structure_with_conect = read_structure_string(augmented_text, FileFormat.PDB)

    source_bonds_with_conect = tuple(
        bond
        for bond in structure_with_conect.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    assert len(source_bonds_with_conect) == len(link_bonds)
    for bond in source_bonds_with_conect:
        assert bond.source_metadata is not None
        assert bond.source_metadata.record_type is SourceBondRecordType.PDB_LINK


def test_write_pdb_emits_conect_from_source_explicit_topology_bonds() -> None:
    """PDB egress should serialize source-explicit topology bonds as CONECT."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="M",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "CONECT    1    2",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    pdb_text = write_structure_string(structure, FileFormat.PDB)
    conect_lines = tuple(
        line for line in pdb_text.splitlines() if line.startswith("CONECT")
    )
    roundtripped = read_structure_string(pdb_text, FileFormat.PDB)

    assert conect_lines == ("CONECT    1    2", "CONECT    2    1")
    assert len(roundtripped.topology.bonds) == 1
    bond = roundtripped.topology.bonds[0]
    assert bond.source_metadata is not None
    assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT


def test_write_pdb_preserves_link_metadata_via_topology_connections() -> None:
    """PDB LINK source topology should roundtrip as LINK metadata."""

    structure = read_structure_string(build_2q6f_linked_pdb_text(), FileFormat.PDB)
    pdb_text = write_structure_string(structure, FileFormat.PDB)
    roundtripped = read_structure_string(pdb_text, FileFormat.PDB)

    assert "LINK" in pdb_text
    source_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    roundtripped_source_bonds = tuple(
        bond
        for bond in roundtripped.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
        and bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.PDB_LINK
    )
    assert len(roundtripped_source_bonds) == len(source_bonds)
    for bond in roundtripped_source_bonds:
        assert bond.source_metadata is not None
        assert bond.source_metadata.record_type is SourceBondRecordType.PDB_LINK


def test_write_pdb_does_not_emit_non_covalent_source_bonds_as_conect() -> None:
    """Non-covalent source topology should not become PDB CONECT bonds."""

    source = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" ZN ",
                    residue_name="ZN",
                    chain_id="L",
                    residue_seq=1,
                    element="ZN",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="M",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=source.constitution.atom_index(
                        AtomRef(ResidueId("L", 1), "ZN")
                    ),
                    atom_index_2=source.constitution.atom_index(
                        AtomRef(ResidueId("M", 1), "O1")
                    ),
                    relationship_type=BondRelationshipType.METAL_COORDINATION,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_LINK,
                        source_id="metal",
                    ),
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )

    pdb_text = write_structure_string(structure, FileFormat.PDB)

    assert "LINK" in pdb_text
    assert not any(line.startswith("CONECT") for line in pdb_text.splitlines())


def test_write_mmcif_emits_struct_conn_from_source_explicit_topology_bonds() -> None:
    """mmCIF egress should serialize source-explicit topology as struct_conn."""

    structure = read_structure_string(build_2q6f_linked_pdb_text(), FileFormat.PDB)
    mmcif_text = write_structure_string(structure, FileFormat.MMCIF)
    roundtripped = read_structure_string(mmcif_text, FileFormat.MMCIF)

    assert "_struct_conn.id" in mmcif_text
    source_bonds = tuple(
        bond
        for bond in structure.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    roundtripped_source_bonds = tuple(
        bond
        for bond in roundtripped.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )
    assert {
        bond.endpoint_pair() for bond in source_bonds
    } <= {bond.endpoint_pair() for bond in roundtripped_source_bonds}
    for bond in roundtripped_source_bonds:
        assert bond.source_metadata is not None
        assert (
            bond.source_metadata.record_type is SourceBondRecordType.MMCIF_STRUCT_CONN
        )


def test_write_mmcif_emits_struct_conn_from_pdb_conect_topology_bonds() -> None:
    """PDB CONECT-origin topology should remain source-explicit in mmCIF."""

    structure = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="M",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                build_pdb_atom_line(
                    serial=3,
                    record_name="HETATM",
                    atom_name=" N1 ",
                    residue_name="OTH",
                    chain_id="N",
                    residue_seq=1,
                    x=4.0,
                    element="N",
                ),
                "CONECT    1    2",
                "CONECT    2    3",
                "END",
            ]
        ),
        FileFormat.PDB,
    )

    mmcif_text = write_structure_string(structure, FileFormat.MMCIF)
    roundtripped = read_structure_string(mmcif_text, FileFormat.MMCIF)

    source_bonds = tuple(
        bond
        for bond in roundtripped.topology.bonds
        if bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )

    assert "_struct_conn.id" in mmcif_text
    assert "CONECT" not in {
        line.split(maxsplit=1)[0]
        for line in mmcif_text.splitlines()
        if line.startswith("CONECT")
    }
    assert len(source_bonds) == 2
    assert {
        bond.source_metadata.source_id
        for bond in source_bonds
        if bond.source_metadata is not None
    } == {"protrepair_0_1", "protrepair_1_2"}
    assert all(
        bond.relationship_type is BondRelationshipType.UNKNOWN for bond in source_bonds
    )
    assert all(
        bond.source_metadata is not None
        and bond.source_metadata.record_type is SourceBondRecordType.MMCIF_STRUCT_CONN
        for bond in source_bonds
    )


@pytest.mark.parametrize(
    "provenance",
    [
        BondProvenance.TEMPLATE_RESOLVED,
        BondProvenance.SEQUENCE_INFERRED,
        BondProvenance.EVIDENCE_RESOLVED,
        BondProvenance.REPAIR_INFERRED,
    ],
)
def test_write_pdb_emits_conect_from_model_resolved_covalent_bonds(
    provenance: BondProvenance,
) -> None:
    """PDB egress should serialize covalent-like repaired topology as CONECT."""

    source = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    atom_ref_1 = AtomRef(ResidueId("L", 1), "C1")
    atom_ref_2 = AtomRef(ResidueId("L", 1), "O1")
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=source.constitution.atom_index(atom_ref_1),
                    atom_index_2=source.constitution.atom_index(atom_ref_2),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=provenance,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )

    assert gemmi_writer.source_explicit_topology_bonds_for_egress(structure) == ()
    assert gemmi_writer.pdb_conect_topology_bonds_for_egress(structure) == (
        structure.topology.bonds[0],
    )

    pdb_text = write_structure_string(structure, FileFormat.PDB)

    assert "CONECT    1    2" in pdb_text


def test_write_pdb_does_not_emit_model_resolved_non_covalent_as_conect() -> None:
    """Only covalent-like model-resolved topology may become PDB CONECT."""

    source = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" ZN ",
                    residue_name="ZN",
                    chain_id="L",
                    residue_seq=1,
                    element="ZN",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="OBS",
                    chain_id="M",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=source.constitution.atom_index(
                        AtomRef(ResidueId("L", 1), "ZN")
                    ),
                    atom_index_2=source.constitution.atom_index(
                        AtomRef(ResidueId("M", 1), "O1")
                    ),
                    relationship_type=BondRelationshipType.METAL_COORDINATION,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )

    assert gemmi_writer.pdb_conect_topology_bonds_for_egress(structure) == ()

    pdb_text = write_structure_string(structure, FileFormat.PDB)

    assert not any(line.startswith("CONECT") for line in pdb_text.splitlines())


def test_write_mmcif_emits_struct_conn_from_model_resolved_covalent_bonds() -> None:
    """mmCIF egress should serialize covalent-like repaired topology."""

    source = read_structure_string(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    record_name="HETATM",
                    atom_name=" C1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    element="C",
                ),
                build_pdb_atom_line(
                    serial=2,
                    record_name="HETATM",
                    atom_name=" O1 ",
                    residue_name="LIG",
                    chain_id="L",
                    residue_seq=1,
                    x=2.0,
                    element="O",
                ),
                "END",
            ]
        ),
        FileFormat.PDB,
    )
    atom_ref_1 = AtomRef(ResidueId("L", 1), "C1")
    atom_ref_2 = AtomRef(ResidueId("L", 1), "O1")
    structure = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=source.constitution.atom_index(atom_ref_1),
                    atom_index_2=source.constitution.atom_index(atom_ref_2),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )

    mmcif_text = write_structure_string(structure, FileFormat.MMCIF)

    assert "_struct_conn.id" in mmcif_text
    assert "'protrepair_0_1' covale" in mmcif_text
    assert " C1 " in mmcif_text
    assert " O1 " in mmcif_text


def test_apply_structure_normalization_policy_drops_bonds_with_missing_endpoints() -> (
    None
):
    """Topology bonds should be filtered when normalization removes endpoints."""

    structure = read_structure_string(
        build_2q6f_linked_pdb_text(),
        FileFormat.PDB,
    )
    assert len(structure.topology.bonds) > 0

    normalized = apply_structure_normalization_policy(
        structure,
        policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.DROP),
    )

    assert not any(
        bond.provenance is BondProvenance.SOURCE_EXPLICIT
        for bond in normalized.topology.bonds
    )


def test_write_structure_string_roundtrips_pdb_and_mmcif() -> None:
    """PDB and mmCIF serialization should roundtrip to the same semantics."""

    structure = build_canonical_structure()

    pdb_roundtrip = read_structure_string(
        write_structure_string(structure, FileFormat.PDB),
        FileFormat.PDB,
    )
    mmcif_roundtrip = read_structure_string(
        write_structure_string(structure, FileFormat.MMCIF),
        FileFormat.MMCIF,
    )

    assert summarize_structure(pdb_roundtrip) == summarize_structure(structure)
    assert summarize_structure(mmcif_roundtrip) == summarize_structure(structure)


def test_write_structure_writes_content_via_atomic_path_boundary(
    tmp_path: Path,
) -> None:
    """Path writer should preserve serialized content and remove temp files."""

    structure = build_canonical_structure()
    output_path = tmp_path / "fixture.pdb"

    write_structure(structure, output_path)

    assert output_path.read_text(encoding="utf-8") == write_structure_string(
        structure,
        FileFormat.PDB,
    )
    assert summarize_structure(read_structure(output_path)) == summarize_structure(
        structure
    )
    assert tuple(tmp_path.glob(".fixture.pdb.*.tmp")) == ()


def test_write_structure_preserves_existing_output_permissions(
    tmp_path: Path,
) -> None:
    """Atomic replacement should not relax or tighten existing file mode bits."""

    structure = build_canonical_structure()
    output_path = tmp_path / "fixture.pdb"
    output_path.write_text("original", encoding="utf-8")
    output_path.chmod(0o600)

    write_structure(structure, output_path)

    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert summarize_structure(read_structure(output_path)) == summarize_structure(
        structure
    )


def test_write_structure_reports_missing_output_directory(tmp_path: Path) -> None:
    """Path writer should report the requested target when parent is absent."""

    structure = build_canonical_structure()
    output_path = tmp_path / "missing" / "fixture.pdb"

    with pytest.raises(FileNotFoundError) as error:
        write_structure(structure, output_path)

    error_message = str(error.value)
    assert str(output_path) in error_message
    assert "output directory does not exist" in error_message


def test_write_structure_does_not_touch_output_when_serialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialization failure should not create or replace boundary files."""

    output_path = tmp_path / "fixture.pdb"
    output_path.write_text("original", encoding="utf-8")

    def _fail_serialization(
        _structure: ProteinStructure,
        _file_format: FileFormat,
    ) -> str:
        raise ValueError("serialization blocked")

    monkeypatch.setattr(gemmi_writer, "write_structure_string", _fail_serialization)

    with pytest.raises(ValueError, match="serialization blocked"):
        write_structure(build_canonical_structure(), output_path)

    assert output_path.read_text(encoding="utf-8") == "original"
    assert tuple(tmp_path.glob(".fixture.pdb.*.tmp")) == ()


def test_write_structure_removes_temp_file_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace failure should preserve the old target and clean temp output."""

    structure = build_canonical_structure()
    output_path = tmp_path / "fixture.pdb"
    output_path.write_text("original", encoding="utf-8")
    original_replace = Path.replace

    def _fail_replace(self: Path, target: Path) -> Path:
        if target == output_path:
            raise OSError("replace blocked")

        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replace blocked"):
        write_structure(structure, output_path)

    assert output_path.read_text(encoding="utf-8") == "original"
    assert tuple(tmp_path.glob(".fixture.pdb.*.tmp")) == ()


def test_write_structure_string_preserves_formal_charge_fields() -> None:
    """PDB and mmCIF writing should preserve polymer and hetero charge fields."""

    polymer_residue = residue_payload(
        component_id="ASP",
        residue_id=ResidueId("A", 1),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
            atom_payload("CG", "C", Vec3(2.0, 1.0, 0.0)),
            atom_payload(
                "OD1",
                "O",
                Vec3(3.0, 1.0, 0.0),
                formal_charge=-1,
            ),
            atom_payload(
                "OD2",
                "O",
                Vec3(2.0, 2.0, 0.0),
                formal_charge=-1,
            ),
        ),
    )
    ligand_residue = residue_payload(
        component_id="LIG",
        residue_id=ResidueId("A", 101),
        atoms=(
            atom_payload(
                "O1",
                "O",
                Vec3(5.0, 0.0, 0.0),
                formal_charge=-1,
            ),
        ),
        is_hetero=True,
    )
    structure = build_structure(
        chains=(chain_payload("A", (polymer_residue,)),),
        ligands=(ligand_residue,),
        source_format=FileFormat.PDB,
    )

    pdb_text = write_structure_string(structure, FileFormat.PDB)

    assert "OD1 ASP A   1" in pdb_text
    assert "OD2 ASP A   1" in pdb_text
    assert "OD1 ASP A   1" in pdb_text and "O1-" in next(
        line for line in pdb_text.splitlines() if " OD1 ASP A   1" in line
    )
    assert "OD2 ASP A   1" in pdb_text and "O1-" in next(
        line for line in pdb_text.splitlines() if " OD2 ASP A   1" in line
    )
    assert "O1  LIG A 101" in pdb_text and "O1-" in next(
        line for line in pdb_text.splitlines() if " O1  LIG A 101" in line
    )

    mmcif_text = write_structure_string(structure, FileFormat.MMCIF)
    assert "-1" in mmcif_text


def test_read_structure_infers_format_from_path_suffix(tmp_path: Path) -> None:
    """Path-based reading should infer PDB and mmCIF formats from suffixes."""

    structure = build_canonical_structure()
    pdb_path = tmp_path / "fixture.pdb"
    cif_path = tmp_path / "fixture.cif"
    pdb_path.write_text(
        write_structure_string(structure, FileFormat.PDB),
        encoding="utf-8",
    )
    cif_path.write_text(
        write_structure_string(structure, FileFormat.MMCIF),
        encoding="utf-8",
    )

    pdb_structure = read_structure(pdb_path)
    cif_structure = read_structure(cif_path)

    assert summarize_structure(pdb_structure) == summarize_structure(structure)
    assert summarize_structure(cif_structure) == summarize_structure(structure)


def test_read_structure_pdb_path_matches_string_ingress_with_conect(
    tmp_path: Path,
) -> None:
    """PDB path ingress should match string ingress while preserving CONECT bonds."""

    pdb_contents = build_pdb_text(
        [
            build_pdb_atom_line(
                serial=1,
                atom_name=" CA ",
                altloc="A",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=1.0,
                y=2.0,
                z=3.0,
                occupancy=0.20,
                element="C",
            ),
            build_pdb_atom_line(
                serial=2,
                atom_name=" CA ",
                altloc="B",
                residue_name="ALA",
                chain_id="A",
                residue_seq=1,
                x=4.0,
                y=5.0,
                z=6.0,
                occupancy=0.80,
                element="C",
            ),
            build_pdb_atom_line(
                serial=3,
                record_name="HETATM",
                atom_name=" C1 ",
                residue_name="LIG",
                chain_id="L",
                residue_seq=1,
                x=7.0,
                y=8.0,
                z=9.0,
                element="C",
            ),
            build_pdb_atom_line(
                serial=4,
                record_name="HETATM",
                atom_name=" O1 ",
                residue_name="LIG",
                chain_id="L",
                residue_seq=1,
                x=8.0,
                y=8.0,
                z=9.0,
                element="O",
            ),
            "CONECT    3    4",
            "END",
        ]
    )
    pdb_path = tmp_path / "single_read_fixture.pdb"
    pdb_path.write_text(pdb_contents, encoding="utf-8")

    from_path = read_structure(pdb_path)
    from_string = read_structure_string(
        pdb_contents,
        FileFormat.PDB,
        source_name=pdb_path.name,
    )

    assert summarize_structure(from_path) == summarize_structure(from_string)
    assert from_path.provenance.ingress.source_name == pdb_path.name
    assert len(from_path.topology.bonds) == 1
    assert from_path.topology.bonds == from_string.topology.bonds


def test_read_structure_pdb_path_reads_text_once_and_skips_file_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PDB path ingress should share one text read across gemmi and CONECT."""

    pdb_path = tmp_path / "single_read_counter.pdb"
    pdb_path.write_text(
        build_pdb_text(
            [
                build_pdb_atom_line(
                    serial=1,
                    atom_name=" N  ",
                    residue_name="GLY",
                    chain_id="A",
                    residue_seq=1,
                    element="N",
                ),
                "END",
            ]
        ),
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    read_text_calls: list[Path] = []

    def _counting_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self == pdb_path:
            read_text_calls.append(self)

        return original_read_text(self, encoding=encoding, errors=errors)

    def _fail_file_parser(_path: Path, _file_format: FileFormat) -> object:
        raise AssertionError("PDB path ingress should not call read_raw_structure")

    monkeypatch.setattr(Path, "read_text", _counting_read_text)
    monkeypatch.setattr(gemmi_ingress, "read_raw_structure", _fail_file_parser)

    structure = read_structure(pdb_path)

    assert structure.geometry.atom_count() == 1
    assert read_text_calls == [pdb_path]


def test_read_structure_mmcif_path_still_uses_file_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mmCIF path ingress should keep its existing gemmi file-parser path."""

    structure = build_canonical_structure()
    cif_path = tmp_path / "fixture.cif"
    cif_path.write_text(
        write_structure_string(structure, FileFormat.MMCIF),
        encoding="utf-8",
    )
    original_read_raw_structure = gemmi_ingress.read_raw_structure
    read_raw_calls: list[FileFormat] = []

    def _spy_read_raw_structure(path: Path, file_format: FileFormat) -> object:
        read_raw_calls.append(file_format)
        return original_read_raw_structure(path, file_format)

    monkeypatch.setattr(gemmi_ingress, "read_raw_structure", _spy_read_raw_structure)

    roundtripped = read_structure(cif_path)

    assert summarize_structure(roundtripped) == summarize_structure(structure)
    assert read_raw_calls == [FileFormat.MMCIF]


def build_canonical_structure() -> ProteinStructure:
    """Build a small canonical structure for I/O roundtrip tests."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("SER", "A", 2, ("N", "CA", "C", "O", "CB")),
                ),
            ),
            chain_payload(
                "B",
                (build_residue("TYR", "B", 10, ("N", "CA", "C", "O", "CB", "CG")),),
            ),
        ),
        ligands=(build_residue("FAD", "B", 99, ("C1", "N1", "O1"), is_hetero=True),),
        source_format=FileFormat.PDB,
        source_name="fixture",
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Build a canonical residue for roundtrip tests."""

    atoms = tuple(
        build_atom(atom_name, atom_index)
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_atom(atom_name: str, atom_index: int) -> CanonicalAtomPayload:
    """Build a canonical atom with deterministic coordinates."""

    return atom_payload(
        name=atom_name,
        element=infer_element(atom_name),
        position=Vec3(
            x=float(atom_index),
            y=float(atom_index) + 0.5,
            z=float(atom_index) + 1.0,
        ),
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple test element from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]


def _bond_display_token(structure: ProteinStructure, bond: TopologyBond) -> str:
    """Return one source-bond display token in atom-ref space."""

    atom_ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
    atom_ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
    return f"{atom_ref_1.display_token()}-{atom_ref_2.display_token()}"


def build_pdb_text(lines: list[str]) -> str:
    """Join fixed-width PDB records into a text payload."""

    return "\n".join(lines) + "\n"


def build_2q6f_linked_pdb_text() -> str:
    """Return the 2Q6F local fixture with source-explicit annotated links."""

    fixture_path = Path("tests/fixtures/pdb/refinement/2q6f_cys143_pje_local.pdb")
    pdb_text = fixture_path.read_text(encoding="utf-8")
    link_records = "\n".join(
        (
            "LINK         C   LEU D   4                 N   PJE D   5     "
            "1555   1555  1.35  ",
            "LINK         C   PJE D   5                 O   010 D   6     "
            "1555   1555  1.44  ",
        )
    )
    return pdb_text.replace("CRYST1", f"{link_records}\nCRYST1", 1)


def build_pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_seq: int,
    record_name: str = "ATOM",
    altloc: str = " ",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    occupancy: float = 1.0,
    b_factor: float = 20.0,
    element: str = "",
) -> str:
    """Build one fixed-width PDB atom record for gemmi reader tests."""

    return (
        f"{record_name:<6}{serial:>5} {atom_name}{altloc}{residue_name:>3} "
        f"{chain_id}{residue_seq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}"
        f"          {element:>2}  "
    )
