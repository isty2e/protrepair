from pathlib import Path

from protrepair.chemistry import (
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
)
from protrepair.chemistry.nonstandard.ingestion import (
    angle_between,
    first_non_degenerate_plane_normal,
    ingest_component_library,
    ingest_component_template,
    ingest_restraint_template,
)
from protrepair.io.gemmi_normalization import gemmi


def test_ingestion_angle_returns_none_for_coincident_tuple_vector() -> None:
    """Boundary tuple geometry should retain its observation-only None result."""

    assert (
        angle_between(
            center_position=(0.0, 0.0, 0.0),
            first_position=(0.0, 0.0, 0.0),
            second_position=(1.0, 0.0, 0.0),
        )
        is None
    )


def test_ingestion_plane_normal_returns_none_for_collinear_positions() -> None:
    """CCD tuple geometry should not inherit placement exception semantics."""

    assert (
        first_non_degenerate_plane_normal(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            )
        )
        is None
    )


def test_ingest_component_library_reads_custom_monomer_cif(tmp_path: Path) -> None:
    """Custom monomer-CIF assets should ingest into the canonical library."""

    cif_path = tmp_path / "custom-components.cif"
    cif_path.write_text(
        "\n".join(
            (
                "data_MSE",
                "_chem_comp.id MSE",
                "_chem_comp.type 'L-PEPTIDE LINKING'",
                "_chem_comp.name SELENOMETHIONINE",
                "_chem_comp.mon_nstd_parent_comp_id MET",
                "_chem_comp.one_letter_code M",
                "loop_",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "_chem_comp_atom.pdbx_stereo_config",
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
                "N N 0 N 1.869 0.121 -1.981",
                "CA C 0 S 0.520 -0.459 -1.960",
                "C C 0 N -0.232 -0.028 -3.192",
                "O O 0 N 0.011 1.037 -3.706",
                "CB C 0 N -0.222 0.023 -0.713",
                "CG C 0 N 0.541 -0.414 0.538",
                "SE SE 0 N -0.410 0.204 2.135",
                "CE C 0 N 1.106 0.721 2.198",
                "H H 0 N 2.103 1.087 -1.548",
                "loop_",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
                "N CA SING N",
                "CA C SING N",
                "C O DOUB N",
                "CA CB SING N",
                "CB CG SING N",
                "CG SE SING N",
                "SE CE SING N",
                "N H SING N",
            )
        ),
        encoding="utf-8",
    )

    library = ingest_component_library(cif_path)
    template = library.require("MSE")

    assert template.component_id == "MSE"
    assert template.definition.formal_charges == {}
    assert template.ordered_atom_names() == (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SE",
        "CE",
    )
    assert template.definition.bonded_atom_names("SE") == frozenset({"CG", "CE"})

    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    ingested_template = ingest_component_template(block)
    restraint_template = ingest_restraint_template(block)

    assert ingested_template is not None
    assert restraint_template is not None
    heavy_atom_semantics = ingested_template.heavy_atom_semantics
    hydrogen_semantics = ingested_template.hydrogen_semantics
    assert isinstance(heavy_atom_semantics, IdealGeometryHeavyAtomSemantics)
    assert isinstance(hydrogen_semantics, IdealGeometryHydrogenSemantics)
    atom_by_name = {
        atom.atom_name: atom for atom in heavy_atom_semantics.component.atoms
    }
    assert atom_by_name["CA"].stereo_config == "S"
    assert atom_by_name["SE"].ideal_position is not None
    assert atom_by_name["H"].ideal_position is not None
    assert restraint_template.bond_targets
    assert restraint_template.angle_targets
    assert restraint_template.chirality_targets
    assert restraint_template.bond_target("CG", "SE") is not None


def test_ingest_component_library_tolerates_missing_parent_mapping(
    tmp_path: Path,
) -> None:
    """External assets without parent residue metadata should still ingest."""

    cif_path = tmp_path / "custom-components.cif"
    cif_path.write_text(
        "\n".join(
            (
                "data_NSP",
                "_chem_comp.id NSP",
                "_chem_comp.type 'L-PEPTIDE LINKING'",
                "_chem_comp.name NONSTANDARD",
                "loop_",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "N N 0",
                "CA C 0",
                "C C 0",
                "O O 0",
                "CB C 0",
                "loop_",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
                "N CA SING N",
                "CA C SING N",
                "C O DOUB N",
                "CA CB SING N",
            )
        ),
        encoding="utf-8",
    )

    library = ingest_component_library(cif_path)

    assert library.require("NSP").component_id == "NSP"


def test_ingest_component_record_derives_planarity_targets_from_ideal_geometry(
    tmp_path: Path,
) -> None:
    """Conjugated ideal coordinates should yield plane-restraint targets."""

    cif_path = tmp_path / "custom-planar.cif"
    cif_path.write_text(
        "\n".join(
            (
                "data_PTRX",
                "_chem_comp.id PTRX",
                "_chem_comp.type 'L-PEPTIDE LINKING'",
                "_chem_comp.name PLANAR",
                "_chem_comp.mon_nstd_parent_comp_id TYR",
                "loop_",
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "_chem_comp_atom.pdbx_stereo_config",
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
                "N N 0 N 0.0 0.0 1.0",
                "CA C 0 S 1.4 0.0 1.0",
                "C C 0 N 2.6 0.0 1.0",
                "O O 0 N 3.6 0.0 1.0",
                "CB C 0 N 1.4 1.5 1.0",
                "CG C 0 N 2.2 2.5 1.0",
                "CD1 C 0 N 3.5 2.3 1.0",
                "CE1 C 0 N 4.1 3.4 1.0",
                "CZ C 0 N 3.4 4.6 1.0",
                "CE2 C 0 N 2.1 4.8 1.0",
                "CD2 C 0 N 1.5 3.7 1.0",
                "loop_",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
                "N CA SING N",
                "CA C SING N",
                "C O DOUB N",
                "CA CB SING N",
                "CB CG SING N",
                "CG CD1 AROM Y",
                "CD1 CE1 AROM Y",
                "CE1 CZ AROM Y",
                "CZ CE2 AROM Y",
                "CE2 CD2 AROM Y",
                "CD2 CG AROM Y",
            )
        ),
        encoding="utf-8",
    )

    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    restraint_template = ingest_restraint_template(block)

    assert restraint_template is not None
    assert restraint_template.plane_targets
    assert any(
        {"CG", "CD1", "CE1", "CZ"} <= set(target.atom_names)
        for target in restraint_template.plane_targets
    )
