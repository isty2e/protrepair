from pathlib import Path

from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_registry,
    build_bundled_nonstandard_restraint_library,
    bundled_nonstandard_asset_path,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
    bundled_retained_non_polymer_asset_path,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.heavy import repair_heavy_atoms


def test_bundled_nonstandard_registry_exposes_curated_record() -> None:
    """Bundled nonstandard records should retain metadata and heavy filtering."""

    registry = build_bundled_nonstandard_registry()
    record = registry.get("MSE")

    assert record is not None
    assert record.parent_standard_id == "MET"
    assert record.one_letter_code == "M"
    assert any(atom.is_hydrogen() for atom in record.atoms)
    assert all(not atom.is_hydrogen() for atom in record.heavy_atoms())
    atom_by_name = {atom.atom_name: atom for atom in record.atoms}
    assert atom_by_name["CA"].stereo_config == "S"
    assert atom_by_name["SE"].ideal_position is not None
    assert atom_by_name["H"].ideal_position is not None
    assert record.to_template().component_id == "MSE"
    assert "SE" in record.to_template().ordered_atom_names()
    assert "H" not in record.to_template().ordered_atom_names()
    restraint_template = record.to_restraint_template()
    assert restraint_template.component_id == "MSE"
    assert restraint_template.bond_target("CG", "SE") is not None
    assert restraint_template.angle_targets_for_center("CA")
    assert restraint_template.chirality_targets


def test_bundled_nonstandard_restraint_library_exposes_planarity_targets() -> None:
    """Bundled nonstandard restraint library should surface derived plane targets."""

    restraint_library = build_bundled_nonstandard_restraint_library()
    ptr_restraints = restraint_library.require("PTR")

    assert ptr_restraints.bond_target("P", "O1P") is not None
    assert ptr_restraints.angle_targets_for_center("CG")
    assert ptr_restraints.plane_targets
    assert any(
        {"CG", "CD1", "CE1", "CZ"} <= set(target.atom_names)
        for target in ptr_restraints.plane_targets
    )


def test_default_component_library_includes_bundled_nonstandard_templates() -> None:
    """The internal default library should include bundled nonstandard templates."""

    standard_library = build_standard_component_library()
    default_library = build_default_component_library()

    assert standard_library.get("MSE") is None
    assert default_library.require("MSE").component_id == "MSE"
    assert default_library.require("SEP").component_id == "SEP"
    assert default_library.normalize_component_id("HSE") == "HIS"


def test_default_component_library_includes_bundled_retained_non_polymer_templates(
) -> None:
    """The internal default library should include bundled cofactor templates."""

    default_library = build_default_component_library()

    hem_template = default_library.require("HEM")
    fad_template = default_library.require("FAD")
    nag_template = default_library.require("NAG")

    assert hem_template.component_id == "HEM"
    assert hem_template.hydrogen_semantics is not None
    assert fad_template.component_id == "FAD"
    assert fad_template.hydrogen_semantics is not None
    assert nag_template.component_id == "NAG"
    assert nag_template.hydrogen_semantics is not None


def test_complete_bundled_nonstandard_residue_is_not_treated_as_unknown() -> None:
    """Complete bundled nonstandard residues should survive heavy repair cleanly."""

    default_library = build_default_component_library()
    record = build_bundled_nonstandard_registry().get("MSE")
    assert record is not None
    template = default_library.require("MSE")
    atom_by_name = {atom.atom_name: atom for atom in record.heavy_atoms()}
    residue = residue_payload(
        component_id="MSE",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=tuple(
            atom_payload(
                atom_name,
                atom_by_name[atom_name].element,
                Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(template.ordered_atom_names())
        ),
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        ligands=(),
        source_name="mse-complete",
        source_format=FileFormat.PDB,
    )

    result = repair_heavy_atoms(structure)

    assert not result.has_warnings()
    assert result.structure.constitution.chain("A").residues[0].component_id == "MSE"


def test_bundled_nonstandard_asset_is_packaged_in_source_tree() -> None:
    """The packaged gz asset should exist where the runtime loader expects it."""

    asset_path = Path(bundled_nonstandard_asset_path())
    assert asset_path.name == "nonstandard_components.json.gz"
    assert asset_path.exists()


def test_bundled_retained_non_polymer_registry_exposes_curated_cofactor() -> None:
    """Bundled retained non-polymers should surface known cofactor records."""

    registry = build_bundled_retained_non_polymer_registry()
    record = registry.get("HEM")

    assert record is not None
    assert record.component_id == "HEM"
    assert any(atom.is_hydrogen() for atom in record.atoms)
    assert record.to_restraint_template().bond_targets


def test_bundled_retained_non_polymer_asset_is_packaged_in_source_tree() -> None:
    """The packaged retained non-polymer asset should exist in source tree."""

    asset_path = Path(bundled_retained_non_polymer_asset_path())
    assert asset_path.name == "retained_non_polymer_components.json.gz"
    assert asset_path.exists()
