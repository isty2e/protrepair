"""Unit tests for generic side-chain packing request and result entities."""

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

from protrepair.diagnostics import (
    IssueSeverity,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.errors import ResidueNotFoundError
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.packing import (
    PackingCapabilities,
    PackingMode,
    PackingPlan,
    PackingResult,
    PackingScope,
    PackingSpec,
)


def test_packing_spec_normalizes_backend_name_and_residue_lists() -> None:
    """Packing specs should normalize backend names and deduplicate residue ids."""

    residue_id = ResidueId(chain_id="A", seq_num=10)
    spec = PackingSpec(
        backend_name=" FASPR ",
        mode=PackingMode.PACK,
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(residue_id, residue_id),
    )

    assert spec.backend_name == "faspr"
    assert spec.mutable_residue_ids == (residue_id,)
    assert spec.referenced_residue_ids() == (residue_id,)
    assert spec.is_local()


def test_packing_spec_rejects_invalid_local_or_overlapping_ids() -> None:
    """Packing specs should reject invalid local and overlapping residue sets."""

    residue_id = ResidueId(chain_id="A", seq_num=10)

    with pytest.raises(ValueError, match="local side-chain packing"):
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
        )

    with pytest.raises(ValueError, match="must not overlap"):
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(residue_id,),
            frozen_residue_ids=(residue_id,),
        )


def test_packing_capabilities_enforce_supported_surface() -> None:
    """Capability declarations should reject unsupported specs."""

    capabilities = PackingCapabilities(
        supports_full_structure_packing=False,
        supports_local_packing=True,
        supports_partial_sequence=False,
        supports_refinement=False,
        supports_noncanonical_components=False,
        deterministic_given_same_inputs=True,
    )
    supported_spec = PackingSpec(
        backend_name="hpacker",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId(chain_id="A", seq_num=10),),
    )
    unsupported_spec = PackingSpec(
        backend_name="hpacker",
        mode=PackingMode.REFINE,
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId(chain_id="A", seq_num=10),),
    )

    assert capabilities.supports_spec(supported_spec)
    assert not capabilities.supports_spec(unsupported_spec)

    with pytest.raises(ValueError, match="refinement mode"):
        capabilities.require_support_for(unsupported_spec)


def test_packing_plan_validates_referenced_residues_and_sequence_length() -> None:
    """Packing plans should reject residue ids or sequences that do not match."""

    structure = build_structure()
    mutable_residue_ids = (
        ResidueId(chain_id="A", seq_num=1),
        ResidueId(chain_id="A", seq_num=2),
    )

    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=mutable_residue_ids,
            target_sequence="AG",
        ),
    )

    assert plan.spec.referenced_residue_ids() == mutable_residue_ids
    assert plan.selected_residue_count() == 2

    with pytest.raises(ValueError, match="target_sequence length"):
        PackingPlan.from_inputs(
            structure,
            PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=mutable_residue_ids,
                target_sequence="A",
            ),
        )

    with pytest.raises(ResidueNotFoundError):
        PackingPlan.from_inputs(
            structure,
            PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=(ResidueId(chain_id="A", seq_num=9),),
            ),
        )


def test_packing_plan_resolves_selection_and_fixed_residues() -> None:
    """Packing plans should own selection and fixed-residue semantics."""

    structure = build_structure()
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(ResidueId(chain_id="A", seq_num=2),),
            frozen_residue_ids=(ResidueId(chain_id="A", seq_num=1),),
            target_sequence="V",
        ),
    )

    assert plan.selected_residue_ids() == (ResidueId(chain_id="A", seq_num=2),)
    assert plan.fixed_residue_ids() == (
        ResidueId(chain_id="A", seq_num=1),
    )
    assert plan.selected_residue_count() == 1
    assert plan.selected_residue_sites() == (
        structure.constitution.chain("A").residue(ResidueId(chain_id="A", seq_num=2)),
    )


def test_packing_plan_detects_changed_residues_in_packed_output() -> None:
    """Packing plans should compare packed structures at the domain layer."""

    structure = build_structure()
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )
    mutated_residue = structure.constitution.residue_or_ligand(
        ResidueId(chain_id="A", seq_num=2)
    )
    assert mutated_residue is not None
    mutated_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(
            ResidueId(chain_id="A", seq_num=2)
        ),
    )
    changed_structure = structure.with_updated_residue_facets(
        mutated_residue,
        residue_geometry=mutated_geometry.with_atom_geometry(
            "CB",
            atom_payload("CB", "C", Vec3(99.0, 98.0, 97.0), b_factor=20.0)[1],
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(
                ResidueId(chain_id="A", seq_num=2)
            ),
        ),
    )

    assert plan.changed_residue_ids_after(changed_structure) == (
        ResidueId(chain_id="A", seq_num=2),
    )


def test_packing_plan_ignores_hydrogen_regeneration_and_coordinate_noise() -> None:
    """Packing change reporting should not confuse backend noise with packing."""

    structure = build_hydrogenated_structure()
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
    )
    residue_id = ResidueId(chain_id="A", seq_num=2)
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    noisy_geometry = residue_geometry.with_atom_geometry(
        "CB",
        residue_geometry.atom_geometry("CB").with_position(
            residue_geometry.position("CB").with_offset(0.01, 0.0, 0.0)
        ),
    )
    noisy_structure = structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=noisy_geometry,
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )

    assert plan.changed_residue_ids_after(noisy_structure.without_hydrogens()) == ()


def test_packing_plan_reports_only_meaningful_local_heavy_atom_changes() -> None:
    """Local packing change reporting should not inflate fixed-residue scope."""

    structure = build_structure()
    plan = PackingPlan.from_inputs(
        structure,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(ResidueId(chain_id="A", seq_num=2),),
        ),
    )
    noisy_fixed_residue_id = ResidueId(chain_id="A", seq_num=1)
    moved_mutable_residue_id = ResidueId(chain_id="A", seq_num=2)
    changed_structure = move_residue_atom(
        move_residue_atom(
            structure,
            residue_id=noisy_fixed_residue_id,
            atom_name="CA",
            dx=0.01,
        ),
        residue_id=moved_mutable_residue_id,
        atom_name="CB",
        dx=1.0,
    )

    assert plan.changed_residue_ids_after(changed_structure) == (
        moved_mutable_residue_id,
    )


def test_packing_plan_rejects_empty_polymer_region() -> None:
    """Packing plans require at least one polymer residue."""

    with pytest.raises(ValueError, match="at least one polymer residue"):
        PackingPlan(
            structure=build_canonical_structure(
                chains=(),
                source_format=FileFormat.PDB,
                source_name="empty-polymer",
            ),
            spec=PackingSpec(backend_name="faspr", scope=PackingScope.FULL),
            polymer_residue_ids=(),
        )


def test_packing_result_normalizes_backend_metadata_and_validates_changes() -> None:
    """Packing results should normalize backend metadata and changed residues."""

    structure = build_structure()
    residue_id = ResidueId(chain_id="A", seq_num=1)
    result = PackingResult(
        packed_structure=structure,
        changed_residue_ids=(residue_id, residue_id),
        issues=(
            ValidationIssue.for_residue(
                kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                severity=IssueSeverity.WARNING,
                message="backend skipped one residue",
                residue_id=residue_id,
            ),
        ),
        backend_name=" FASPR ",
        backend_version=" 1.0.0 ",
    )

    assert result.backend_name == "faspr"
    assert result.backend_version == "1.0.0"
    assert result.changed_residue_ids == (residue_id,)
    assert result.changed_residue(residue_id)
    assert result.changed_residue_count() == 1
    assert result.has_issues()


def build_structure() -> ProteinStructure:
    """Build a minimal canonical structure for packing tests."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="packing-fixture",
    )


def build_hydrogenated_structure() -> ProteinStructure:
    """Build a minimal packing fixture with polymer hydrogens."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue("GLY", "A", 1, ("N", "CA", "C", "O", "H")),
                    build_residue("ALA", "A", 2, ("N", "CA", "C", "O", "CB", "H")),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="packing-hydrogen-fixture",
    )


def move_residue_atom(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    atom_name: str,
    dx: float,
) -> ProteinStructure:
    """Return a copy with one residue atom translated along x."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    moved_geometry = residue_geometry.with_atom_geometry(
        atom_name,
        residue_geometry.atom_geometry(atom_name).with_position(
            residue_geometry.position(atom_name).with_offset(dx, 0.0, 0.0)
        ),
    )
    return structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=moved_geometry,
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
) -> CanonicalResiduePayload:
    """Build a canonical residue for a unit fixture."""

    atoms = tuple(
        build_atom(atom_name, atom_index)
        for atom_index, atom_name in enumerate(atom_names, start=1)
    )
    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
    )


def build_atom(atom_name: str, atom_index: int) -> CanonicalAtomPayload:
    """Build one deterministic canonical atom for a unit fixture."""

    position = Vec3(float(atom_index), float(atom_index + 1), float(atom_index + 2))
    return atom_payload(
        name=atom_name,
        element=infer_element(atom_name),
        position=position,
        b_factor=20.0,
    )


def infer_element(atom_name: str) -> str:
    """Infer a simple element token from an atom name."""

    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        raise ValueError(f"atom_name must contain at least one letter: {atom_name}")

    return letters[0]
