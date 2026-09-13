"""Public hydrogen/workflow paths consume the same coupled chemical targets."""

from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.hydrogen_expectation import (
    derive_structure_hydrogen_expectation_model,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondProvenance
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.protonation import (
    PrasRatioHistidineProtonationRequest,
)
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.completion.terminal.augmentation import (
    augment_c_terminal_oxt,
)
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
    requested_process_goal,
)


@pytest.fixture(scope="module")
def source() -> ProteinStructure:
    text = Path("tests/fixtures/corpus/pdb1afc.ent").read_text()
    return read_structure_string(
        "\n".join(
            line
            for line in text.splitlines()
            if line.startswith("ATOM") and line[21] == "A"
        ),
        FileFormat.PDB,
    )


@pytest.fixture(scope="module")
def prepared(source):
    result = add_hydrogens(source)
    assert not result.issues
    return result.structure


def _targets(structure):
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(structure), build_default_component_library()
    )
    return tuple(
        target
        for chain in structure.constitution.chains
        for residue in chain.residues
        for target in preparation.targets_for(residue.residue_id)
    )


def test_direct_hydrogenation_realizes_every_standard_site(prepared):
    targets = _targets(prepared)
    assert len(targets) == 290
    assert all(target.is_realized() for target in targets)
    expectations = derive_structure_hydrogen_expectation_model(
        prepared, component_library=build_default_component_library()
    )
    for (
        residue_id,
        names,
    ) in expectations.expected_hydrogen_atom_names_by_residue.items():
        residue = prepared.constitution.residue_or_ligand(residue_id)
        assert residue is not None
        assert set(names) <= set(residue.atom_site_names())


def test_repeated_direct_preparation_preserves_graph_and_coordinates(prepared):
    repeated = add_hydrogens(prepared)
    assert not repeated.issues
    assert repeated.structure.constitution == prepared.constitution
    assert repeated.structure.topology == prepared.topology
    assert repeated.structure.geometry == prepared.geometry


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_public_preparation_egress_preserves_site_graphs(prepared, file_format):
    reread = read_structure_string(
        write_structure_string(prepared, file_format), file_format
    )
    for target in _targets(prepared):
        graph = target.resolution.graph
        assert graph is not None
        for atom in graph.atoms:
            index = reread.constitution.atom_index(
                AtomRef(target.residue_id, atom.name)
            )
            assert (reread.topology.formal_charge(index) or 0) == atom.charge
        for bond in graph.bonds:
            first = reread.constitution.atom_index(
                AtomRef(target.residue_id, bond.atom_name_1)
            )
            second = reread.constitution.atom_index(
                AtomRef(target.residue_id, bond.atom_name_2)
            )
            actual = reread.topology.bond_between(first, second)
            assert actual is not None and actual.order == bond.order
        for hydrogen, parent in target.hydrogen_atom_sites():
            first = reread.constitution.atom_index(
                AtomRef(target.residue_id, hydrogen.name)
            )
            second = reread.constitution.atom_index(AtomRef(target.residue_id, parent))
            actual = reread.topology.bond_between(first, second)
            assert actual is not None
            assert actual.order in (
                (None, 1) if file_format is FileFormat.PDB else (1,)
            )

    # Single CONECT expresses connectivity, not an explicit order. Re-preparation
    # resolves its observed H identities without inventing an additional proton.
    repeated = add_hydrogens(reread)
    assert not repeated.issues
    assert repeated.structure.constitution == reread.constitution
    assert all(target.is_realized() for target in _targets(repeated.structure))


def test_terminal_prerequisite_adds_oxygen_and_its_canonical_bond(prepared):
    residue_id = prepared.constitution.chains[0].residues[-1].residue_id
    oxygen = AtomRef(residue_id, "OXT")
    source = prepared.without_atom_refs((oxygen,))
    hydrogen_only = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue_id,))
    )
    assert hydrogen_only.structure.constitution.resolve_atom_index(oxygen) is None
    assert hydrogen_only.issues

    terminal = augment_c_terminal_oxt(source)
    assert not terminal.issues
    c = terminal.structure.constitution.atom_index(AtomRef(residue_id, "C"))
    oxt = terminal.structure.constitution.atom_index(oxygen)
    bond = terminal.structure.topology.bond_between(c, oxt)
    assert bond is not None
    assert bond.order == 1
    assert bond.provenance is BondProvenance.TEMPLATE_RESOLVED
    for offset in range(len(source.constitution.atom_slots)):
        source_index = AtomIndex(offset)
        ref = source.constitution.atom_ref_at(source_index)
        index = terminal.structure.constitution.atom_index(ref)
        assert terminal.structure.geometry.atom_geometry(
            index
        ) == source.geometry.atom_geometry(source_index)

    completed = materialize_hydrogens_core(
        terminal.structure, target_residue_ids=frozenset((residue_id,))
    )
    assert not completed.issues
    assert all(target.is_realized() for target in _targets(completed.structure))


@pytest.mark.parametrize("workflow", (False, True))
def test_changed_histidine_request_replaces_generated_choices(prepared, workflow):
    def prepare(structure, request=None):
        if not workflow:
            return add_hydrogens(structure, histidine_protonation=request).structure
        return process_structure(
            structure,
            requested_goals=RequestedGoalSet(
                (
                    requested_process_goal(
                        scope=WholeStructureScope(),
                        value=HydrogenCoverageState.COMPLETE,
                    ),
                )
            ),
            transform_requests=WorkflowTransformRequests(histidine_protonation=request),
        ).structure

    cation = prepare(prepared, PrasRatioHistidineProtonationRequest(1.0))
    his = [
        target
        for target in _targets(cation)
        if target.site.template.component_id == "HIS"
        and target.site.kind is PolymerChemicalSite.SIDECHAIN
    ]
    assert len(his) == 5
    assert all(
        target.is_realized()
        and target.resolution.graph is not None
        and target.resolution.graph.protonation_key()[0] == 1
        for target in his
    )

    unchanged = prepare(cation)
    assert unchanged.topology == cation.topology
    reset = prepare(cation, PrasRatioHistidineProtonationRequest(0.0))
    assert not reset.provenance.microstate_overrides
    assert all(target.is_realized() for target in _targets(reset))
    for target in _targets(reset):
        if (
            target.site.template.component_id == "HIS"
            and target.site.kind is PolymerChemicalSite.SIDECHAIN
        ):
            assert target.resolution.graph is not None
            assert target.resolution.graph.protonation_key()[0] == 0


def test_unresolved_histidine_does_not_evaluate_fixed_template_ring_hydrogens(source):
    residue = next(
        r for r in source.constitution.chains[0].residues if r.component_id == "HIS"
    )
    broken = source.without_atom_refs((AtomRef(residue.residue_id, "ND1"),))
    result = materialize_hydrogens_core(
        broken, target_residue_ids=frozenset((residue.residue_id,))
    )
    completed = result.structure.constitution.residue_or_ligand(residue.residue_id)
    assert completed is not None
    assert {"HA", "HB1", "HB2"} <= set(completed.atom_site_names())
    assert not {"HD1", "HD2", "HE1", "HE2"}.intersection(completed.atom_site_names())
    assert any("sidechain" in issue.message for issue in result.issues)


def test_workflow_repairs_charge_even_when_h_inventory_is_complete(prepared):
    target = next(
        t
        for t in _targets(prepared)
        if t.site.template.component_id == "LYS"
        and t.site.kind is PolymerChemicalSite.SIDECHAIN
    )
    index = prepared.constitution.residue_index(target.residue_id)
    current = prepared.with_updated_residue_facets(
        prepared.constitution.residue_site_at(index),
        residue_geometry=prepared.residue_geometry(index),
        formal_charge_by_atom_name=tuple(
            (name, 0 if name == "NZ" else charge)
            for name, charge in prepared.residue_formal_charge_by_atom_name(index)
        ),
    )
    assert not next(
        t
        for t in _targets(current)
        if t.residue_id == target.residue_id
        and t.site.kind is PolymerChemicalSite.SIDECHAIN
    ).is_realized()
    result = process_structure(
        current,
        requested_goals=RequestedGoalSet(
            (
                requested_process_goal(
                    scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
                ),
            )
        ),
    )
    repaired = next(
        t
        for t in _targets(result.structure)
        if t.residue_id == target.residue_id
        and t.site.kind is PolymerChemicalSite.SIDECHAIN
    )
    assert repaired.is_realized(), [issue.message for issue in result.issues]
    assert result.structure.geometry == current.geometry
