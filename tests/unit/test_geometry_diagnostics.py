"""Heavy-atom geometry diagnostics over canonical repaired structures."""

from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
    RestraintLibrary,
    UnknownElementRadiusError,
    build_default_component_library,
)
from protrepair.chemistry.nonstandard.ingestion import (
    ingest_component_library,
    ingest_restraint_library,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    ValidationIssueKind,
    detect_heavy_geometry,
)
from protrepair.diagnostics.geometry import (
    bond_angle_degrees,
    severe_intrinsic_geometry_residues,
)
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.heavy import repair_heavy_atoms


def test_detect_heavy_geometry_accepts_repaired_standard_sidechain() -> None:
    """Standard heavy-atom repair should not trip broad geometry plausibility checks."""

    residue = build_residue(
        component_id="SER",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(2.155, 1.058, 0.0)),
        ),
    )
    structure = repair_heavy_atoms(build_structure((residue,))).structure
    residue_id = ResidueId(chain_id="A", seq_num=1)

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
        residue_ids=(residue_id,),
    )

    assert report.is_empty()


def test_detect_heavy_geometry_reports_bond_length_outlier() -> None:
    """Implausible bonded distances should be reported explicitly."""

    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_payload("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_payload("CB", "C", Vec3(5.50, 0.0, 0.0)),
                    atom_payload("OG", "O", Vec3(6.90, 0.0, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert len(report.bond_length_outliers) >= 1
    assert any(
        {
            outlier.atom_name_1,
            outlier.atom_name_2,
        }
        == {"CA", "CB"}
        for outlier in report.bond_length_outliers
    )


def test_detect_heavy_geometry_uses_standard_restraint_bond_targets() -> None:
    """Standard residues should use tighter bond targets than the coarse fallback."""

    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 1.730, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert any(
        {outlier.atom_name_1, outlier.atom_name_2} == {"CA", "CB"}
        and outlier.observed_distance_angstrom < 1.8
        and outlier.tolerance_angstrom < 0.25
        for outlier in report.bond_length_outliers
    )


def test_detect_heavy_geometry_reports_bond_angle_outlier() -> None:
    """Implausible heavy-atom angles should be surfaced explicitly."""

    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 0.1, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 1.4, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert len(report.bond_angle_outliers) >= 1
    assert any(
        outlier.center_atom_name == "CA" for outlier in report.bond_angle_outliers
    )


def test_detect_heavy_geometry_skips_undefined_bond_angle_triplets() -> None:
    """Coincident atoms should not be converted into fake angle outliers."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=residue_id,
                atoms=(
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(-1.165, 0.978, 0.0)),
                    atom_payload("OG", "O", Vec3(-1.500, 2.250, 0.0)),
                ),
            ),
        )
    )
    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert (
        bond_angle_degrees(
            residue_geometry,
            atom_name_1="N",
            center_atom_name="CA",
            atom_name_2="C",
        )
        is None
    )
    assert not any(
        outlier.center_atom_name == "CA"
        and {outlier.atom_name_1, outlier.atom_name_2} == {"N", "C"}
        for outlier in report.bond_angle_outliers
    )


def test_detect_heavy_geometry_uses_standard_restraint_angle_targets() -> None:
    """Standard residues should use explicit angle targets over the broad window."""

    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(-1.165, 0.978, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert any(
        outlier.center_atom_name == "CA"
        and {outlier.atom_name_1, outlier.atom_name_2} == {"CB", "C"}
        and outlier.maximum_angle_degrees < 140.0
        for outlier in report.bond_angle_outliers
    )


def test_detect_heavy_geometry_uses_bundled_nonstandard_restraints_by_default() -> None:
    """Bundled nonstandard residues should use packaged restraint targets by default."""

    structure = build_structure(
        (
            build_residue(
                component_id="MSE",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(-1.165, 0.978, 0.0)),
                    atom_payload("CG", "C", Vec3(-2.250, 0.050, 0.0)),
                    atom_payload("SE", "SE", Vec3(-3.950, 0.350, 0.0)),
                    atom_payload("CE", "C", Vec3(-5.000, 1.700, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_default_component_library(),
    )

    assert any(
        {outlier.atom_name_1, outlier.atom_name_2} == {"CG", "SE"}
        and outlier.component_id == "MSE"
        and outlier.tolerance_angstrom < 0.25
        for outlier in report.bond_length_outliers
    )


def test_detect_heavy_geometry_does_not_false_flag_mse_c_se_fallback() -> None:
    """MSE C-Se fallback lengths should use selenium radii, not carbon defaults."""

    structure = build_structure(
        (
            build_residue(
                component_id="MSE",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(-1.165, 0.978, 0.0)),
                    atom_payload("CG", "C", Vec3(-2.250, 0.050, 0.0)),
                    atom_payload("SE", "SE", Vec3(-4.183, 0.375, 0.0)),
                    atom_payload("CE", "C", Vec3(-5.100, 2.104, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_default_component_library(),
        restraint_library=RestraintLibrary(),
    )

    assert not any(
        outlier.component_id == "MSE"
        and {outlier.atom_name_1, outlier.atom_name_2}
        in ({"CG", "SE"}, {"SE", "CE"})
        for outlier in report.bond_length_outliers
    )


def test_detect_heavy_geometry_does_not_false_flag_fe_n_fallback() -> None:
    """Template-declared Fe-N bonds should not use the default carbon radius."""

    component_library = ComponentLibrary(
        templates={
            "FEX": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="FEX",
                    atom_names=("FE", "N1"),
                    bonds=(BondDefinition("FE", "N1"),),
                )
            )
        }
    )
    structure = build_structure(
        (
            build_residue(
                component_id="FEX",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("FE", "FE", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("N1", "N", Vec3(2.03, 0.0, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=component_library,
        restraint_library=RestraintLibrary(),
    )

    assert report.is_empty()


def test_detect_heavy_geometry_reports_unknown_fallback_bond_radii_once() -> None:
    """Fallback bond-length checks should aggregate unknown covalent radii."""

    component_library = ComponentLibrary(
        templates={
            "UNX": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="UNX",
                    atom_names=("X1", "Y1"),
                    bonds=(
                        BondDefinition("X1", "Y1"),
                        BondDefinition("X1", "Y1"),
                    ),
                )
            )
        }
    )
    structure = build_structure(
        (
            build_residue(
                component_id="UNX",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("X1", "XX", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("Y1", "C1", Vec3(1.4, 0.0, 0.0)),
                ),
            ),
        )
    )

    with pytest.raises(UnknownElementRadiusError) as error_info:
        detect_heavy_geometry(
            structure,
            component_library=component_library,
            restraint_library=RestraintLibrary(),
        )

    message = str(error_info.value)
    assert "heavy geometry fallback bonds for A:1" in message
    assert message.count("XX") == 1
    assert "C1" in message


def test_detect_heavy_geometry_accepts_external_ingested_restraints(
    tmp_path: Path,
) -> None:
    """Externally ingested CCD assets should provide audit targets when supplied."""

    cif_path = tmp_path / "custom-components.cif"
    cif_path.write_text(
        "\n".join(
            (
                "data_NSX",
                "_chem_comp.id NSX",
                "_chem_comp.type 'L-PEPTIDE LINKING'",
                "_chem_comp.name CUSTOM",
                "_chem_comp.mon_nstd_parent_comp_id SER",
                "loop_",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "_chem_comp_atom.pdbx_stereo_config",
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
                "N N 0 N -0.531 1.358 0.0",
                "CA C 0 S 0.0 0.0 0.0",
                "C C 0 N 1.525 0.0 0.0",
                "O O 0 N 2.100 1.200 0.0",
                "CB C 0 N -1.165 0.978 0.0",
                "SG S 0 N -2.950 1.100 0.0",
                "loop_",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
                "N CA SING N",
                "CA C SING N",
                "C O DOUB N",
                "CA CB SING N",
                "CB SG SING N",
            )
        ),
        encoding="utf-8",
    )
    component_library = ingest_component_library(cif_path)
    restraint_library = ingest_restraint_library(cif_path)
    structure = build_structure(
        (
            build_residue(
                component_id="NSX",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-0.531, 1.358, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.525, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.100, 1.200, 0.0)),
                    atom_payload("CB", "C", Vec3(-1.165, 0.978, 0.0)),
                    atom_payload("SG", "S", Vec3(-2.480, 1.070, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=component_library,
        restraint_library=restraint_library,
    )

    assert any(
        {outlier.atom_name_1, outlier.atom_name_2} == {"CB", "SG"}
        and outlier.component_id == "NSX"
        and outlier.tolerance_angstrom < 0.25
        for outlier in report.bond_length_outliers
    )


def test_detect_heavy_geometry_can_filter_to_repaired_residues() -> None:
    """Residue filters should keep the audit focused on repaired residues."""

    residue_1 = build_residue(
        component_id="SER",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(1.0, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(2.0, 0.0, 0.0)),
            atom_payload("CB", "C", Vec3(0.0, 0.1, 0.0)),
            atom_payload("OG", "O", Vec3(0.0, 1.4, 0.0)),
        ),
    )
    residue_2 = build_residue(
        component_id="GLY",
        residue_id=ResidueId(chain_id="A", seq_num=2),
        atoms=(
            atom_payload("N", "N", Vec3(2.919, 1.358, 0.0)),
            atom_payload("CA", "C", Vec3(3.450, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(4.975, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(5.605, 1.058, 0.0)),
        ),
    )
    structure = build_structure((residue_1, residue_2))

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
        residue_ids=(ResidueId(chain_id="A", seq_num=2),),
    )

    assert report.is_empty()


def test_heavy_geometry_report_projects_invalid_geometry_issues() -> None:
    """Heavy-atom geometry findings should project into validation issues."""

    structure = build_structure(
        (
            build_residue(
                component_id="SER",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 0.1, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 1.4, 0.0)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )
    issues = report.to_issues()

    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.INVALID_GEOMETRY


def test_severe_intrinsic_geometry_residues_flags_pathological_leucine() -> None:
    """Auto-repair severity should recognize strong restraint-backed residue burden."""

    structure = build_structure(
        (
            build_residue(
                component_id="LEU",
                residue_id=ResidueId(chain_id="A", seq_num=32),
                atoms=(
                    atom_payload("N", "N", Vec3(-4.300, 6.200, 18.600)),
                    atom_payload("CA", "C", Vec3(-3.600, 5.000, 19.100)),
                    atom_payload("C", "C", Vec3(-2.400, 5.300, 20.000)),
                    atom_payload("O", "O", Vec3(-1.400, 4.700, 19.800)),
                    atom_payload("CB", "C", Vec3(-3.474, 4.042, 20.144)),
                    atom_payload("CG", "C", Vec3(-4.009, 2.598, 20.202)),
                    atom_payload("CD1", "C", Vec3(-5.389, 2.549, 20.596)),
                    atom_payload("CD2", "C", Vec3(-4.341, 2.566, 18.974)),
                ),
            ),
        )
    )

    report = detect_heavy_geometry(
        structure,
        component_library=build_standard_component_library(),
    )
    findings = severe_intrinsic_geometry_residues(report)

    assert len(findings) == 1
    assert findings[0].residue_id == ResidueId("A", 32)
    assert findings[0].restraint_backed_outlier_count >= 3
    assert findings[0].total_outlier_count >= 5


def build_residue(
    *,
    component_id: str,
    residue_id: ResidueId,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CanonicalResiduePayload:
    """Build one canonical residue payload for geometry diagnostics."""

    return residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        atoms=atoms,
    )


def build_structure(residues: tuple[CanonicalResiduePayload, ...]) -> ProteinStructure:
    """Build one canonical test structure from one chain of residues."""

    return build_canonical_structure(
        chains=(chain_payload("A", residues),),
        source_format=FileFormat.PDB,
        source_name="geometry-diagnostic-fixture",
    )
