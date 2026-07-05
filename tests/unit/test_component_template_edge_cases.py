"""Adversarial edge cases for component-template semantics."""

from dataclasses import replace

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
    ChemicalComponentDefinition,
    ComponentLibrary,
    ForceFieldAtomParams,
    HeavyAtomSemantics,
    HydrogenSemantics,
    ResidueTemplate,
    RotatableHydrogenKind,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.geometry import Vec3
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
)
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.terminal import augment_c_terminal_oxt
from protrepair.workflow.contracts import OrphanFragmentPolicy


def make_template(
    component_id: str,
    *,
    aliases: tuple[str, ...] = (),
) -> ResidueTemplate:
    """Build a minimal residue template for edge-case tests."""

    return ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id=component_id,
            atom_names=("N", "CA", "C", "O"),
            aliases=aliases,
        )
    )


def atom_entry(
    name: str,
    element: str,
    position: Vec3,
    *,
    formal_charge: int | None = None,
) -> CanonicalAtomPayload:
    """Build one canonical atom payload for tests."""

    return atom_payload(
        name,
        element,
        position,
        formal_charge=formal_charge,
    )


def residue_entry(
    *,
    component_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
    chain_id: str = "A",
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Build one canonical residue payload for tests."""

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_test_structure(
    *,
    residues: tuple[CanonicalResiduePayload, ...] = (),
    ligands: tuple[CanonicalResiduePayload, ...] = (),
    source_name: str,
    source_format: FileFormat = FileFormat.PDB,
    chain_id: str = "A",
) -> ProteinStructure:
    """Build one canonical test structure from payload tuples."""

    return build_canonical_structure(
        chains=(chain_payload(chain_id, residues),) if residues else (),
        ligands=ligands,
        source_format=source_format,
        source_name=source_name,
    )


def residue_component_id(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> str:
    """Return one canonical residue component id."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return residue_site.component_id


def residue_atom_names(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> tuple[str, ...]:
    """Return atom-site names for one canonical residue."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return residue_site.atom_site_names()


def ligand_atom_names(
    structure: ProteinStructure,
    index: int = 0,
) -> tuple[str, ...]:
    """Return atom-site names for one canonical ligand payload."""

    return structure.constitution.ligands[index].atom_site_names()


def has_atom(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name: str,
) -> bool:
    """Return whether one canonical atom site exists."""

    return (
        structure.constitution.resolve_atom_site(
            AtomRef(residue_id=residue_id, atom_name=atom_name)
        )
        is not None
    )


def topology_bond_between(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> TopologyBond | None:
    """Return a canonical residue-local topology bond when present."""

    atom_index_1 = structure.constitution.atom_index(AtomRef(residue_id, atom_name_1))
    atom_index_2 = structure.constitution.atom_index(AtomRef(residue_id, atom_name_2))
    for bond in structure.topology.bonds:
        if {bond.atom_index_1, bond.atom_index_2} == {atom_index_1, atom_index_2}:
            return bond

    return None


def has_template_topology_bond(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> bool:
    """Return whether canonical topology has one template-resolved covalent bond."""

    bond = topology_bond_between(structure, residue_id, atom_name_1, atom_name_2)
    return (
        bond is not None
        and bond.relationship_type is BondRelationshipType.COVALENT
        and bond.provenance is BondProvenance.TEMPLATE_RESOLVED
    )


def atom_position(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name: str,
) -> Vec3:
    """Return one canonical atom position."""

    atom_ref = AtomRef(residue_id=residue_id, atom_name=atom_name)
    atom_geometry = structure.geometry.atom_geometry(
        structure.constitution.atom_index(atom_ref)
    )

    return atom_geometry.position


def atom_distance(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
) -> float:
    """Return the distance between two canonical atoms in one residue."""

    return atom_position(
        structure,
        residue_id,
        atom_name_1,
    ).distance_to(atom_position(structure, residue_id, atom_name_2))


def test_component_library_rejects_alias_collision_between_templates() -> None:
    """Two templates must not silently share the same alias."""

    with pytest.raises(ValueError, match="ambiguous alias"):
        ComponentLibrary(
            templates={
                "HIS": make_template("HIS", aliases=("HSE",)),
                "MSE": make_template("MSE", aliases=("HSE",)),
            }
        )


def test_component_library_rejects_alias_collision_with_component_id() -> None:
    """A template alias must not shadow another template's canonical id."""

    with pytest.raises(ValueError, match="ambiguous alias"):
        ComponentLibrary(
            templates={
                "HIS": make_template("HIS", aliases=("MSE",)),
                "MSE": make_template("MSE"),
            }
        )


def test_heavy_atom_semantics_rejects_duplicate_atom_names() -> None:
    """Heavy-atom semantics should reject duplicate atom-order entries."""

    with pytest.raises(ValueError, match="unique"):
        HeavyAtomSemantics(
            program=InternalCoordinateProgram.backbone_only(),
            atom_order=("N", "CA", "CA"),
        )


def test_hydrogen_semantics_rejects_rotatable_and_static_configuration_mix() -> None:
    """Hydrogen semantics must not mix rotatable and static planning modes."""

    with pytest.raises(ValueError, match="either"):
        HydrogenSemantics(
            plan_with_backbone=((("HA",), "class3", ("CB", "N", "CA")),),
            rotatable_kind=RotatableHydrogenKind.SER,
        )


def test_hydrogen_semantics_rejects_noop_configuration() -> None:
    """Hydrogen semantics should reject configurations with no executable plan."""

    with pytest.raises(ValueError, match="require"):
        HydrogenSemantics()


def test_hydrogen_semantics_rejects_without_backbone_only_plan() -> None:
    """Backbone-conditional plans require a primary with-backbone plan."""

    with pytest.raises(ValueError, match="with_backbone"):
        HydrogenSemantics(
            plan_without_backbone=((("HA",), "class3", ("CB", "N", "CA")),),
        )


def test_repair_heavy_atoms_preserves_ligand_hydrogens() -> None:
    """Direct heavy-atom repair should not strip ligand hydrogens."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="GLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                ),
            ),
        ),
        ligands=(
            residue_entry(
                component_id="LIG",
                seq_num=2,
                atoms=(
                    atom_entry("C1", "C", Vec3(7.0, 7.0, 7.0)),
                    atom_entry("H1", "H", Vec3(7.8, 7.0, 7.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_name="ligand-hydrogen-edge",
    )

    result = repair_heavy_atoms(structure)

    assert ligand_atom_names(result.structure) == ("C1", "H1")


def test_custom_alias_template_normalizes_and_repairs_heavy_atoms() -> None:
    """A custom alias should resolve to the canonical template during repair."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    aliased_template = replace(
        ala_template,
        definition=replace(ala_template.definition, aliases=("DAL",)),
    )
    custom_library = library.with_template(aliased_template)
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="DAL",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                ),
            ),
        ),
        source_name="alias-heavy-repair",
    )

    result = repair_heavy_atoms(structure, component_library=custom_library)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert residue_component_id(result.structure, residue_id) == "ALA"
    assert has_atom(result.structure, residue_id, "CB")
    assert has_template_topology_bond(result.structure, residue_id, "CA", "CB")
    assert any(
        event.kind is RepairEventKind.COMPONENT_NORMALIZED for event in result.repairs
    )
    assert any(
        event.kind is RepairEventKind.HEAVY_ATOMS_ADDED for event in result.repairs
    )


def test_heavy_repair_adds_topology_for_sidechain_branch_atoms() -> None:
    """Repaired side-chain heavy atoms should carry template topology bonds."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.5, 0.2)),
                ),
            ),
        ),
        source_name="ser-sidechain-topology-repair",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "OG")
    assert has_template_topology_bond(result.structure, residue_id, "CB", "OG")
    assert any(
        event.kind is RepairEventKind.HEAVY_ATOMS_ADDED
        and event.atom_names == ("OG",)
        for event in result.repairs
    )


def test_heavy_repair_preserves_source_explicit_topology_bonds() -> None:
    """Source-explicit topology bonds should survive heavy repair remapping."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="ALA",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                ),
            ),
        ),
        source_name="source-link-heavy-repair",
    )
    structure_with_source_bond = ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(residue_id, "N")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(residue_id, "CA")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_LINK,
                        source_id="toy-source-link",
                    ),
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )

    result = repair_heavy_atoms(structure_with_source_bond)

    source_bond = topology_bond_between(result.structure, residue_id, "N", "CA")
    assert source_bond is not None
    assert source_bond.provenance is BondProvenance.SOURCE_EXPLICIT
    assert source_bond.source_metadata is not None
    assert source_bond.source_metadata.source_id == "toy-source-link"
    assert has_template_topology_bond(result.structure, residue_id, "CA", "CB")


def test_heavy_repair_adds_topology_for_terminal_oxt() -> None:
    """Terminal heavy-atom augmentation should add the C-OXT topology bond."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="ALA",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.5, 0.2)),
                ),
            ),
        ),
        source_name="terminal-oxt-topology-repair",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "OXT")
    assert has_template_topology_bond(result.structure, residue_id, "C", "OXT")


def test_integer_hydrogen_plan_arguments_are_accepted_end_to_end() -> None:
    """Integer-valued static hydrogen plans should be executable."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    custom_template = replace(
        ala_template,
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=(
                (("HX",), "calcCoordinate", ("C", "CA", "CB", 1, 180, 109)),
            ),
        ),
    )
    custom_library = library.with_template(custom_template)
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="ALA",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.6, 0.0)),
                ),
            ),
        ),
        source_name="integer-hydrogen-plan",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "HX")


def test_rotatable_hydrogen_tolerates_missing_neighbor_forcefield_params() -> None:
    """Rotatable-hydrogen refinement should tolerate sparse neighbor parameters."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    custom_library = library.with_template(
        replace(gly_template, forcefield_parameters={})
    )
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.6, 0.0)),
                    atom_entry("OG", "O", Vec3(2.8, 3.5, -0.5)),
                ),
            ),
            residue_entry(
                component_id="GLY",
                seq_num=2,
                atoms=(
                    atom_entry("N", "N", Vec3(4.0, 0.8, 1.7)),
                    atom_entry("CA", "C", Vec3(5.0, 1.2, 2.1)),
                    atom_entry("C", "C", Vec3(5.9, 0.4, 3.0)),
                    atom_entry("O", "O", Vec3(6.9, 0.8, 3.4)),
                ),
            ),
        ),
        source_name="missing-forcefield-neighbor",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "HG")


def test_explicit_alias_map_resolves_lowercase_alias_during_heavy_repair() -> None:
    """Explicit alias maps should normalize lowercase aliases into templates."""

    template = replace(
        build_standard_component_library().require("GLY"),
        definition=replace(
            build_standard_component_library().require("GLY").definition,
            aliases=(),
        ),
    )
    library = ComponentLibrary(
        templates={"GLY": template},
        alias_to_component_id={"dgly": "gly"},
    )
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="DGLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                ),
            ),
        ),
        source_name="explicit-alias-map",
    )

    result = repair_heavy_atoms(structure, component_library=library)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert residue_component_id(result.structure, residue_id) == "GLY"
    assert has_atom(result.structure, residue_id, "O")


def test_lowercase_forcefield_parameter_keys_are_normalized() -> None:
    """Template force-field parameter keys should normalize to canonical atom names."""

    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="SER",
            atom_names=("N", "CA", "C", "O", "CB", "OG"),
        ),
        forcefield_parameters={
            "og": ForceFieldAtomParams(0.1, 0.2, 0.3),
            "cb": ForceFieldAtomParams(0.1, 0.2, 0.3),
        },
    )

    assert template.has_forcefield_params("OG")
    assert template.has_forcefield_params("CB")


def test_repair_heavy_atoms_ligand_only_input_preserves_ligand_hydrogens() -> None:
    """Ligand-only heavy repair should preserve ligand hydrogens unchanged."""

    structure = build_test_structure(
        ligands=(
            residue_entry(
                component_id="LIG",
                seq_num=1,
                atoms=(
                    atom_entry("C1", "C", Vec3(7.0, 7.0, 7.0)),
                    atom_entry("H1", "H", Vec3(7.8, 7.0, 7.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_name="ligand-only-heavy-repair",
    )

    result = repair_heavy_atoms(structure)

    assert result.structure.constitution.chains == ()
    assert ligand_atom_names(result.structure) == ("C1", "H1")


def test_repair_heavy_atoms_strips_polymer_hydrogens_but_keeps_ligands() -> None:
    """Direct heavy repair should strip polymer hydrogens without touching ligands."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="GLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("H1", "H", Vec3(0.5, 1.3, 1.2)),
                ),
            ),
        ),
        ligands=(
            residue_entry(
                component_id="LIG",
                seq_num=2,
                atoms=(
                    atom_entry("C1", "C", Vec3(7.0, 7.0, 7.0)),
                    atom_entry("H1", "H", Vec3(7.8, 7.0, 7.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_name="polymer-vs-ligand-hydrogens",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert "H1" not in residue_atom_names(result.structure, residue_id)
    assert ligand_atom_names(result.structure) == ("C1", "H1")


def test_repair_heavy_atoms_does_not_duplicate_existing_oxt() -> None:
    """Terminal OXT repair should not duplicate an already-present atom."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="GLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("OXT", "O", Vec3(3.6, 0.2, 1.0)),
                ),
            ),
        ),
        source_name="existing-oxt",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert residue_atom_names(result.structure, residue_id).count("OXT") == 1
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_terminal_oxt_augmentation_skips_degenerate_terminal_frame() -> None:
    """Undefined terminal geometry should be a no-op, not a false repair event."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="GLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_entry("O", "O", Vec3(2.0, 0.0, 0.0)),
                ),
            ),
        ),
        source_name="degenerate-terminal-oxt-frame",
    )

    result = augment_c_terminal_oxt(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert not has_atom(result.structure, residue_id, "OXT")
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_repair_heavy_atoms_skips_degenerate_terminal_oxt_frame() -> None:
    """Direct heavy repair should not report OXT when terminal placement no-ops."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="GLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_entry("O", "O", Vec3(2.0, 0.0, 0.0)),
                ),
            ),
        ),
        source_name="degenerate-direct-heavy-terminal-oxt-frame",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert not has_atom(result.structure, residue_id, "OXT")
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_repair_heavy_atoms_keeps_backbone_o_when_sidechain_placement_fails() -> None:
    """A later side-chain placement failure should not discard a valid O repair."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("CB", "C", Vec3(1.45, 0.0, 0.0)),
                ),
            ),
        ),
        source_name="partial-heavy-placement-after-degenerate-sidechain",
    )

    result = repair_heavy_atoms(structure, augment_c_terminal_oxt=False)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "O")
    assert not has_atom(result.structure, residue_id, "OG")
    assert any(
        event.kind is RepairEventKind.HEAVY_ATOMS_ADDED
        and event.atom_names == ("O",)
        for event in result.repairs
    )


def test_repair_heavy_atoms_rebuilds_orphan_sidechain_fragment() -> None:
    """Disconnected sidechain atoms should be rebuilt from the anchored fragment."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("OG", "O", Vec3(3.00, -2.00, 0.0)),
                ),
            ),
        ),
        source_name="orphan-serine-fragment",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "CB")
    assert has_atom(result.structure, residue_id, "OG")
    assert atom_position(result.structure, residue_id, "OG") != Vec3(3.0, -2.0, 0.0)
    assert atom_distance(result.structure, residue_id, "CB", "OG") < 1.6


def test_repair_heavy_atoms_can_preserve_orphan_fragment_by_policy() -> None:
    """Preserve policy should keep disconnected orphan atoms in place."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("OG", "O", Vec3(3.00, -2.00, 0.0)),
                ),
            ),
        ),
        source_name="preserved-orphan-serine-fragment",
    )

    result = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "CB")
    assert has_atom(result.structure, residue_id, "OG")
    assert atom_position(result.structure, residue_id, "OG") == Vec3(3.0, -2.0, 0.0)


def test_reference_guidance_replaces_orphan_fragment_atoms_after_pruning() -> None:
    """Reference-guided repair should replace orphan atoms, not preserve them."""

    input_structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("OG", "O", Vec3(3.00, -2.00, 0.0)),
                ),
            ),
        ),
        source_name="orphan-serine-input",
    )
    reference_structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.88, -1.20)),
                    atom_entry("OG", "O", Vec3(1.95, -1.55, -2.35)),
                ),
            ),
        ),
        source_name="orphan-serine-reference",
    )

    result = repair_heavy_atoms(
        input_structure,
        reference_structure=reference_structure,
    )
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert atom_position(result.structure, residue_id, "CB") == atom_position(
        reference_structure,
        residue_id,
        "CB",
    )
    assert atom_position(result.structure, residue_id, "OG") == atom_position(
        reference_structure,
        residue_id,
        "OG",
    )


def test_repair_heavy_atoms_rebuilds_multistep_orphan_sidechain_fragment() -> None:
    """Orphaned multi-atom sidechain subtrees should be rebuilt from the seed."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="ASP",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("OD1", "O", Vec3(4.00, -2.20, 0.10)),
                    atom_entry("OD2", "O", Vec3(4.30, -1.40, -0.80)),
                ),
            ),
        ),
        source_name="orphan-aspartate-fragment",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "CG")
    assert has_atom(result.structure, residue_id, "OD1")
    assert has_atom(result.structure, residue_id, "OD2")
    assert atom_position(result.structure, residue_id, "OD1") != Vec3(4.0, -2.2, 0.1)
    assert atom_position(result.structure, residue_id, "OD2") != Vec3(4.3, -1.4, -0.8)


def test_repair_heavy_atoms_can_salvage_safe_orphan_fragment_geometry() -> None:
    """Safe salvage should preserve connected orphan geometry better than rebuild."""

    seed_structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="PHE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("CD1", "C", Vec3(4.00, -2.20, 0.10)),
                    atom_entry("CE1", "C", Vec3(5.00, -2.80, 0.20)),
                    atom_entry("CZ", "C", Vec3(6.00, -3.10, 0.30)),
                ),
            ),
        ),
        source_name="seed-phenylalanine-fragment",
    )
    rebuilt_seed_structure = repair_heavy_atoms(
        seed_structure,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    ).structure
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="PHE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry(
                        "CD1",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CD1",
                        ).with_offset(0.00, 0.00, 0.00),
                    ),
                    atom_entry(
                        "CE1",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CE1",
                        ).with_offset(0.12, -0.03, 0.02),
                    ),
                    atom_entry(
                        "CZ",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CZ",
                        ).with_offset(0.08, 0.05, -0.01),
                    ),
                ),
            ),
        ),
        source_name="salvageable-phenylalanine-fragment",
    )

    rebuilt = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    ).structure
    salvaged = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.SALVAGE_WHEN_SAFE,
    ).structure
    residue_id = ResidueId(chain_id="A", seq_num=1)

    original_distance = atom_distance(structure, residue_id, "CD1", "CE1")
    rebuilt_distance = atom_distance(rebuilt, residue_id, "CD1", "CE1")
    salvaged_distance = atom_distance(salvaged, residue_id, "CD1", "CE1")

    assert has_atom(salvaged, residue_id, "CG")
    assert has_atom(salvaged, residue_id, "CD2")
    assert has_atom(salvaged, residue_id, "CE2")
    assert abs(salvaged_distance - original_distance) < abs(
        rebuilt_distance - original_distance
    )


def test_repair_heavy_atoms_falls_back_when_orphan_salvage_is_not_safe() -> None:
    """Unsafe orphan salvage should fall back to rebuilt coordinates."""

    seed_structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="PHE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("CD1", "C", Vec3(4.00, -2.20, 0.10)),
                    atom_entry("CE1", "C", Vec3(5.00, -2.80, 0.20)),
                    atom_entry("CZ", "C", Vec3(6.00, -3.10, 0.30)),
                ),
            ),
        ),
        source_name="unsafe-phenylalanine-seed",
    )
    rebuilt_seed_structure = repair_heavy_atoms(
        seed_structure,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    ).structure
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="PHE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry(
                        "CD1",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CD1",
                        ),
                    ),
                    atom_entry(
                        "CE1",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CE1",
                        ).with_offset(3.50, -2.70, -3.20),
                    ),
                    atom_entry(
                        "CZ",
                        "C",
                        atom_position(
                            rebuilt_seed_structure,
                            ResidueId(chain_id="A", seq_num=1),
                            "CZ",
                        ).with_offset(6.20, -5.10, -6.40),
                    ),
                ),
            ),
        ),
        source_name="unsafe-phenylalanine-fragment",
    )

    rebuilt = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    ).structure
    salvaged = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.SALVAGE_WHEN_SAFE,
    ).structure
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert atom_position(salvaged, residue_id, "CD1") == atom_position(
        rebuilt,
        residue_id,
        "CD1",
    )
    assert atom_position(salvaged, residue_id, "CE1") == atom_position(
        rebuilt,
        residue_id,
        "CE1",
    )
    assert atom_position(salvaged, residue_id, "CZ") == atom_position(
        rebuilt,
        residue_id,
        "CZ",
    )


def test_repair_heavy_atoms_rebuilds_aromatic_orphan_fragment() -> None:
    """Orphaned aromatic-ring fragments should be rebuilt from the first anchor."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="PHE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("CD1", "C", Vec3(4.00, -2.20, 0.10)),
                    atom_entry("CE1", "C", Vec3(5.00, -2.80, 0.20)),
                    atom_entry("CZ", "C", Vec3(6.00, -3.10, 0.30)),
                ),
            ),
        ),
        source_name="orphan-phenylalanine-fragment",
    )

    result = repair_heavy_atoms(structure)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "CG")
    assert has_atom(result.structure, residue_id, "CD1")
    assert has_atom(result.structure, residue_id, "CD2")
    assert has_atom(result.structure, residue_id, "CE1")
    assert has_atom(result.structure, residue_id, "CE2")
    assert has_atom(result.structure, residue_id, "CZ")
    assert atom_position(result.structure, residue_id, "CD1") != Vec3(4.0, -2.2, 0.1)


def test_repair_heavy_atoms_reports_invalid_backbone_without_terminal_oxt_crash() -> (
    None
):
    """Invalid terminal backbones should report an error instead of crashing."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("OG", "O", Vec3(2.80, -1.40, -1.90)),
                ),
            ),
        ),
        source_name="invalid-backbone-terminal-gap",
    )

    result = repair_heavy_atoms(structure)

    assert not has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "OXT")
    assert any(
        issue.kind is ValidationIssueKind.INVALID_BACKBONE for issue in result.issues
    )
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_repair_heavy_atoms_skips_terminal_oxt_for_unsupported_component() -> None:
    """Unsupported terminal components should be left unchanged at chain end."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="UNK",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                ),
            ),
        ),
        source_name="unsupported-terminal-component",
    )

    result = repair_heavy_atoms(structure)

    assert not has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "OXT")
    assert any(
        issue.kind is ValidationIssueKind.MISSING_COMPONENT_DEFINITION
        for issue in result.issues
    )
    assert not any(
        event.kind is RepairEventKind.C_TERMINAL_OXT_ADDED for event in result.repairs
    )


def test_repair_heavy_atoms_reports_known_nonstandard_template_repair_gap() -> None:
    """Known bundled nonstandard residues should report missing heavy-repair support."""

    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="MSE",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_entry("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_entry("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_entry("O", "O", Vec3(2.10, 2.35, 0.0)),
                    atom_entry("CB", "C", Vec3(1.70, -0.80, -1.20)),
                    atom_entry("CG", "C", Vec3(3.10, -1.20, -1.60)),
                    atom_entry("SE", "SE", Vec3(4.60, -0.50, -1.10)),
                ),
            ),
        ),
        source_name="mse-template-repair-gap",
    )

    result = repair_heavy_atoms(structure)

    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        and "parent MET" in issue.message
        for issue in result.issues
    )


def test_add_hydrogens_with_custom_alias_library_preserves_ligand_hydrogens() -> None:
    """Hydrogenation should preserve ligand hydrogens while using custom aliases."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    aliased_template = replace(
        gly_template,
        definition=replace(gly_template.definition, aliases=("DGLY",)),
    )
    custom_library = library.with_template(aliased_template)
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="DGLY",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                ),
            ),
        ),
        ligands=(
            residue_entry(
                component_id="LIG",
                seq_num=2,
                atoms=(
                    atom_entry("C1", "C", Vec3(7.0, 7.0, 7.0)),
                    atom_entry("H1", "H", Vec3(7.8, 7.0, 7.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_name="custom-alias-hydrogenation",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert (
        residue_component_id(
            result.structure,
            ResidueId(chain_id="A", seq_num=1),
        )
        == "GLY"
    )
    assert ligand_atom_names(result.structure) == ("C1", "H1")


def test_rotatable_hydrogen_tolerates_missing_current_forcefield_params() -> None:
    """Rotatable hydrogen placement should tolerate missing current-residue params."""

    library = build_standard_component_library()
    ser_template = library.require("SER")
    custom_library = library.with_template(
        replace(ser_template, forcefield_parameters={})
    )
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="SER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.6, 0.0)),
                    atom_entry("OG", "O", Vec3(2.8, 3.5, -0.5)),
                ),
            ),
            residue_entry(
                component_id="GLY",
                seq_num=2,
                atoms=(
                    atom_entry("N", "N", Vec3(4.0, 0.8, 1.7)),
                    atom_entry("CA", "C", Vec3(5.0, 1.2, 2.1)),
                    atom_entry("C", "C", Vec3(5.9, 0.4, 3.0)),
                    atom_entry("O", "O", Vec3(6.9, 0.8, 3.4)),
                ),
            ),
        ),
        source_name="missing-current-forcefield",
    )

    result = add_hydrogens(structure, component_library=custom_library)

    assert has_atom(result.structure, ResidueId(chain_id="A", seq_num=1), "HG")


def test_component_library_with_template_preserves_existing_aliases() -> None:
    """Adding one template should not break previously registered aliases."""

    library = build_standard_component_library()
    gly_template = library.require("GLY")
    extended_library = library.with_template(
        replace(
            gly_template,
            definition=replace(gly_template.definition, aliases=("DGLY",)),
        )
    )

    assert extended_library.normalize_component_id("HSE") == "HIS"
    assert extended_library.normalize_component_id("dgly") == "GLY"


def test_hydrogen_semantics_static_plan_prefers_without_backbone_variant() -> None:
    """Backbone-conditional static plans should select the alternate variant."""

    semantics = HydrogenSemantics(
        plan_with_backbone=((("HB",), "class3", ("CB", "N", "CA")),),
        plan_without_backbone=((("HX",), "class3", ("CB", "N", "CA")),),
    )

    assert semantics.static_plan(include_backbone_hydrogen=True) == (
        (("HB",), "class3", ("CB", "N", "CA")),
    )
    assert semantics.static_plan(include_backbone_hydrogen=False) == (
        (("HX",), "class3", ("CB", "N", "CA")),
    )


def test_hydrogen_semantics_static_plan_falls_back_to_primary_plan() -> None:
    """Static plans without an alternate should reuse the primary variant."""

    semantics = HydrogenSemantics(
        plan_with_backbone=((("HB",), "class3", ("CB", "N", "CA")),),
    )

    assert semantics.static_plan(include_backbone_hydrogen=False) == (
        (("HB",), "class3", ("CB", "N", "CA")),
    )


def test_residue_template_missing_atom_names_normalizes_inputs() -> None:
    """Missing-atom checks should normalize present and excluded atom names."""

    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="SER",
            atom_names=("N", "CA", "C", "O", "CB", "OG", "OXT"),
        )
    )

    assert template.missing_atom_names(
        ("n", "ca", "c", "o"),
        exclude_atom_names=("oxt",),
    ) == ("CB", "OG")


def test_component_library_with_template_rejects_new_alias_collision() -> None:
    """Adding a template should still reject alias conflicts against the library."""

    library = ComponentLibrary(
        templates={
            "HIS": make_template("HIS", aliases=("HSE",)),
            "GLY": make_template("GLY"),
        }
    )

    with pytest.raises(ValueError, match="ambiguous alias"):
        library.with_template(make_template("MSE", aliases=("HSE",)))


def test_hydrogen_operation_rejects_unknown_method_name() -> None:
    """Hydrogen geometry DSL should reject unknown operation names."""

    with pytest.raises(ValueError, match="unsupported hydrogen geometry method"):
        HydrogenSemantics.evaluate_operation(
            "class999",
            ("CB", "N", "CA"),
            atom_coordinates={
                "CB": Vec3(1.0, 0.0, 0.0),
                "N": Vec3(0.0, 0.0, 0.0),
                "CA": Vec3(0.0, 1.0, 0.0),
            },
        )


def test_hydrogen_operation_argument_type_guards_hold() -> None:
    """Hydrogen geometry DSL should reject swapped numeric and atom arguments."""

    with pytest.raises(TypeError, match="coordinate arguments"):
        HydrogenSemantics.resolve_coordinate_argument(
            1.0,
            {"CA": Vec3(0.0, 0.0, 0.0)},
        )

    with pytest.raises(TypeError, match="numeric hydrogen-plan arguments"):
        HydrogenSemantics.resolve_numeric_argument("CA")


def test_custom_alias_heavy_repair_emits_single_normalization_event() -> None:
    """Alias-driven heavy repair should normalize exactly once per residue."""

    library = build_standard_component_library()
    ser_template = library.require("SER")
    custom_library = library.with_template(
        replace(
            ser_template,
            definition=replace(ser_template.definition, aliases=("DSER",)),
        )
    )
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="DSER",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.6, 0.0)),
                ),
            ),
        ),
        source_name="single-normalization-event",
    )

    result = repair_heavy_atoms(structure, component_library=custom_library)
    normalization_events = [
        event
        for event in result.repairs
        if event.kind is RepairEventKind.COMPONENT_NORMALIZED
    ]

    assert len(normalization_events) == 1
    assert normalization_events[0].component_id == "SER"


def test_integer_plan_single_residue_does_not_add_propagated_backbone_hydrogen() -> (
    None
):
    """Single-residue custom hydrogen plans should not manufacture propagated H."""

    library = build_standard_component_library()
    ala_template = library.require("ALA")
    custom_library = library.with_template(
        replace(
            ala_template,
            hydrogen_semantics=HydrogenSemantics(
                plan_with_backbone=(
                    (("HX",), "calcCoordinate", ("C", "CA", "CB", 1, 180, 109)),
                ),
            ),
        )
    )
    structure = build_test_structure(
        residues=(
            residue_entry(
                component_id="ALA",
                seq_num=1,
                atoms=(
                    atom_entry("N", "N", Vec3(1.0, 1.0, 1.0)),
                    atom_entry("CA", "C", Vec3(2.0, 1.5, 1.0)),
                    atom_entry("C", "C", Vec3(3.0, 1.0, 1.5)),
                    atom_entry("O", "O", Vec3(3.8, 1.2, 2.3)),
                    atom_entry("CB", "C", Vec3(2.0, 2.6, 0.0)),
                ),
            ),
        ),
        source_name="single-residue-custom-plan",
    )

    result = add_hydrogens(structure, component_library=custom_library)
    residue_id = ResidueId(chain_id="A", seq_num=1)

    assert has_atom(result.structure, residue_id, "HX")
    assert "H" not in residue_atom_names(result.structure, residue_id)


def test_component_library_accepts_matching_explicit_alias_mapping() -> None:
    """Explicit alias mappings may repeat template aliases when they agree."""

    template = make_template("GLY", aliases=("DGLY",))
    library = ComponentLibrary(
        templates={"GLY": template},
        alias_to_component_id={"dgly": "GLY"},
    )

    assert library.normalize_component_id("DGLY") == "GLY"


def test_class2_operation_accepts_integer_bond_length_argument() -> None:
    """Tetrahedral-pair DSL operations should accept integer bond lengths."""

    coordinates = HydrogenSemantics.evaluate_operation(
        "class2",
        ("CA", "CG", "CB", 1),
        atom_coordinates={
            "CA": Vec3(1.0, 0.0, 0.0),
            "CG": Vec3(0.0, 1.0, 0.0),
            "CB": Vec3(0.0, 0.0, 0.0),
        },
    )

    assert len(coordinates) == 2


def test_class5_operation_accepts_integer_bond_length_argument() -> None:
    """Planar-single DSL operations should accept integer bond lengths."""

    coordinates = HydrogenSemantics.evaluate_operation(
        "class5",
        ("CE1", "CD1", "CG", 1),
        atom_coordinates={
            "CE1": Vec3(1.0, 0.0, 0.0),
            "CD1": Vec3(0.0, 0.0, 0.0),
            "CG": Vec3(0.0, 1.0, 0.0),
        },
    )

    assert len(coordinates) == 1


def test_class3_operation_returns_one_coordinate() -> None:
    """Single-coordinate tetrahedral DSL operations should stay shape-stable."""

    coordinates = HydrogenSemantics.evaluate_operation(
        "class3",
        ("CB", "N", "CA"),
        atom_coordinates={
            "CB": Vec3(1.0, 0.0, 0.0),
            "N": Vec3(0.0, 0.0, 0.0),
            "CA": Vec3(0.0, 1.0, 0.0),
        },
    )

    assert len(coordinates) == 1
