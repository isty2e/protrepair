"""Canonical reconstruction-plan tests for ingested nonstandard residues."""

from tests.support.canonical_builders import (
    atom_payload,
    completion_payload,
)

from protrepair.chemistry import IdealGeometryHeavyAtomSemantics
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.fragment_matching import FragmentBoundary
from protrepair.transformer.completion.heavy.reconstruction import (
    build_component_reconstruction_plan,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.workflow.contracts import OrphanFragmentPolicy


def test_build_component_reconstruction_plan_extends_missing_path_through_orphan() -> (
    None
):
    """Missing bridge atoms should pull downstream orphan atoms into rebuild order."""

    record = require_record("MSE")
    parent_template = build_standard_component_library().require("MET")
    residue = build_residue_from_record(
        record,
        present_atom_names=("N", "CA", "C", "O", "CB", "CG", "CE"),
    )

    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=parent_template,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    )

    assert plan.parent_mapping is not None
    assert plan.parent_mapping.parent_standard_id == "MET"
    assert plan.parent_mapping.shared_atom_names == (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CE",
    )
    assert plan.parent_mapping.parent_only_atom_names == ("SD",)
    assert plan.parent_mapping.component_only_atom_names == ("SE",)
    assert plan.retained_fragment_atom_names == ("N", "CA", "C", "O", "CB", "CG")
    assert plan.structural_anchor_atom_names == ("N", "CA", "C", "O", "CB")
    assert plan.orphan_atom_names == ("CE",)
    assert plan.reconstruction_atom_names == ("SE", "CE")
    assert plan.frontier_bonds == (
        FragmentBoundary(
            present_atom_name="CG",
            missing_atom_name="SE",
        ),
    )
    assert tuple(layer.atom_names for layer in plan.reconstruction_layers) == (
        ("SE",),
        ("CE",),
    )
    assert plan.placement_atom_names() == ("SE", "CE")
    assert plan.unreachable_atom_names == ()


def test_build_component_reconstruction_plan_rebuilds_detached_phosphate_subtree() -> (
    None
):
    """Missing phosphate bridges should reconstruct the whole detached subtree."""

    record = require_record("SEP")
    parent_template = build_standard_component_library().require("SER")
    residue = build_residue_from_record(
        record,
        present_atom_names=("N", "CA", "CB", "OG", "C", "O", "O1P", "O2P", "O3P"),
    )

    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=parent_template,
        orphan_fragment_policy=OrphanFragmentPolicy.REBUILD,
    )

    assert plan.parent_mapping is not None
    assert plan.parent_mapping.parent_standard_id == "SER"
    assert plan.parent_mapping.shared_atom_names == ("N", "CA", "CB", "OG", "C", "O")
    assert plan.parent_mapping.parent_only_atom_names == ()
    assert plan.parent_mapping.component_only_atom_names == (
        "P",
        "O1P",
        "O2P",
        "O3P",
    )
    assert plan.retained_fragment_atom_names == ("N", "CA", "CB", "OG", "C", "O")
    assert plan.orphan_atom_names == ("O1P", "O2P", "O3P")
    assert plan.reconstruction_atom_names == ("P", "O1P", "O2P", "O3P")
    assert plan.frontier_bonds == (
        FragmentBoundary(
            present_atom_name="OG",
            missing_atom_name="P",
        ),
    )
    assert tuple(layer.atom_names for layer in plan.reconstruction_layers) == (
        ("P",),
        ("O1P", "O2P", "O3P"),
    )
    assert plan.unreachable_atom_names == ()


def test_build_component_reconstruction_plan_respects_preserve_orphan_policy() -> None:
    """Preserve policy should not force detached observed atoms into rebuild."""

    record = require_record("SEP")
    parent_template = build_standard_component_library().require("SER")
    residue = build_residue_from_record(
        record,
        present_atom_names=("N", "CA", "CB", "OG", "C", "O", "O1P", "O2P", "O3P"),
    )

    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=parent_template,
        orphan_fragment_policy=OrphanFragmentPolicy.PRESERVE,
    )

    assert plan.orphan_atom_names == ()
    assert plan.reconstruction_atom_names == ("P",)
    assert tuple(layer.atom_names for layer in plan.reconstruction_layers) == (("P",),)
    assert plan.placement_atom_names() == ("P",)


def require_record(component_id: str):
    """Return one bundled nonstandard record required by the test fixture."""

    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    return record


def canonical_template_for(record):
    """Return the canonical template projected from one bundled record."""

    template = record.to_template()
    assert isinstance(template.heavy_atom_semantics, IdealGeometryHeavyAtomSemantics)
    return template


def build_residue_from_record(
    record,
    *,
    present_atom_names: tuple[str, ...],
) -> CompletionResiduePayload:
    """Build one partial residue from a bundled nonstandard component record."""

    atom_by_name = {atom.atom_name: atom for atom in record.heavy_atoms()}
    return completion_payload(
        component_id=record.component_id,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=tuple(
            atom_payload(
                name=atom_name,
                element=atom_by_name[atom_name].element,
                position=Vec3(float(index), 0.0, 0.0),
            )
            for index, atom_name in enumerate(present_atom_names, start=1)
        ),
    )
