"""Unit tests for continuous local bond-planning support resolution."""

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

import protrepair.state.retained_non_polymer_chemistry as retained_non_polymer_chemistry
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference import retained_non_polymer_fallback
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RdkitUnavailableError, RefinementError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.scope import AtomSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.continuous.bonds import plan_continuous_region_bonds
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    LocalBondPlanningSupportResolution,
    resolve_local_bond_planning_support,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required dependency import guard
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_resolve_local_bond_planning_support_prefers_component_templates() -> None:
    """Known template residues should resolve to template-backed support."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
            source_name="template-backed-ser-support",
        )
    )
    residue_id = ResidueId("A", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.TEMPLATE
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_allows_single_center_passive_context() -> (
    None
):
    """Template-less passive single-center context needs no bonded fallback."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
            source_name="single-center-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_strict_allows_single_center_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict fallback policy must not block nonbonding single-center context."""

    def fail_resolution(*args, **kwargs):
        raise AssertionError("single-center passive context resolved chemistry")

    monkeypatch.setattr(
        "protrepair.transformer.continuous.support."
        "resolve_retained_non_polymer_chemistry",
        fail_resolution,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
            source_name="strict-single-center-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_returns_passive_fallback_bonds() -> None:
    """Connected passive retained residues should expose fallback heavy bonds."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
                atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
                atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
                atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
            ),
            source_name="connected-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
    assert tuple(
        sorted(
            tuple(sorted((bond.atom_name_1, bond.atom_name_2)))
            for bond in resolution.fallback_bond_definitions
        )
    ) == (("C1", "O1"),)


def test_resolve_local_bond_planning_support_strict_policy_blocks_passive_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict passive context support must not consult RDKit fallback chemistry."""

    def fail_resolution(*args, **kwargs):
        raise AssertionError("strict passive support resolved chemistry")

    monkeypatch.setattr(
        "protrepair.transformer.continuous.support."
        "resolve_retained_non_polymer_chemistry",
        fail_resolution,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="strict-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
        allow_retained_non_polymer_rdkit_fallback=False,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_prefers_evidence_over_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit retained-ligand evidence should provide passive heavy topology."""

    def fail_fallback(*args, **kwargs):
        raise AssertionError("evidence-backed passive support called fallback")

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    ligand_residue_id = ResidueId("L", 1)
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="evidence-passive-context-support",
        )
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=ligand_residue_id,
        allow_retained_non_polymer_rdkit_fallback=False,
        retained_non_polymer_chemistry_evidence=evidence,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
    assert tuple(
        sorted(
            tuple(sorted((bond.atom_name_1, bond.atom_name_2)))
            for bond in resolution.fallback_bond_definitions
        )
    ) == (("C1", "O1"),)


def test_resolve_local_bond_planning_support_bad_evidence_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid explicit evidence should block rather than silently falling back."""

    def fail_fallback(*args, **kwargs):
        raise AssertionError("invalid evidence path called fallback")

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    ligand_residue_id = ResidueId("L", 1)
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="bad-evidence-passive-context-support",
        )
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="not_a_smiles",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=ligand_residue_id,
        retained_non_polymer_chemistry_evidence=evidence,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert "explicit chemistry evidence" in resolution.blocker_message


def test_resolve_local_bond_planning_support_rejects_disconnected_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnected evidence must not be accepted as passive ligand topology."""

    def fail_fallback(*args, **kwargs):
        raise AssertionError("disconnected evidence path called fallback")

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        fail_fallback,
    )
    ligand_residue_id = ResidueId("L", 1)
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="disconnected-evidence-passive-context-support",
        )
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="C.O",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=ligand_residue_id,
        retained_non_polymer_chemistry_evidence=evidence,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert "explicit chemistry evidence" in resolution.blocker_message


def test_resolve_local_bond_planning_support_rejects_ambiguous_fallback() -> None:
    """Passive fallback support should reject chemistry unsafe for hydrogenation."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.91, 1.4, 0.0)),
            ),
            source_name="ambiguous-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert resolution.fallback_bond_definitions == ()


def test_continuous_readiness_reuses_retained_ligand_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One readiness pass should not resolve retained RDKit fallback twice."""

    original_infer_fallback = (
        retained_non_polymer_chemistry.infer_retained_non_polymer_rdkit_fallback
    )
    fallback_call_count = 0

    def counting_infer_fallback(*args, **kwargs):
        nonlocal fallback_call_count
        fallback_call_count += 1
        return original_infer_fallback(*args, **kwargs)

    monkeypatch.setattr(
        retained_non_polymer_chemistry,
        "infer_retained_non_polymer_rdkit_fallback",
        counting_infer_fallback,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="continuous-readiness-resolution-reuse",
        )
    )

    derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        AtomSetScope((AtomRef(ResidueId("A", 1), "CA"),)),
        component_library=build_default_component_library(),
        context_radius_angstrom=5.0,
    )

    assert fallback_call_count == 1


def test_plan_continuous_region_bonds_preserves_source_h_anchor() -> None:
    """Template-less passive planning must not infer over source H topology."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("O1", "O", Vec3(3.1, 1.4, 0.0)),
            atom_payload("HSRC", "H", Vec3(3.1, 1.4, 0.9)),
        ),
        source_name="source-h-anchor-passive-context-planning",
    )
    structure = type(structure).from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "C1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "HSRC")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    selected_scope = AtomSetScope(
        atom_refs=(AtomRef(ResidueId("A", 1), "OG"),),
    )
    atom_input = AtomInput(
        atom_indices=(
            structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "OG")),
        ),
        basis=AtomInputBasis.ATOMWISE,
        selected_scope=selected_scope,
    )
    region = ContinuousRelaxationRegion.from_inputs(
        snapshot,
        atom_input,
        context_radius_angstrom=3.0,
    )
    ligand_residue_index = structure.constitution.residue_index(ligand_residue_id)

    planned_bond_names = {
        frozenset(
            {
                structure.constitution.atom_site_at(bond.atom_index_1).name,
                structure.constitution.atom_site_at(bond.atom_index_2).name,
            }
        )
        for bond in plan_continuous_region_bonds(
            region,
            build_default_component_library(),
        )
        if all(
            structure.constitution.residue_index_for_atom_index(atom_index)
            == ligand_residue_index
            for atom_index in (bond.atom_index_1, bond.atom_index_2)
        )
    }

    assert frozenset({"C1", "HSRC"}) in planned_bond_names
    assert frozenset({"O1", "HSRC"}) not in planned_bond_names


def test_continuous_readiness_requires_retained_ligand_h_topology() -> None:
    """Complete retained-ligand H coordinates do not substitute for H topology."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            atom_payload("H001", "H", Vec3(1.5, 0.7, 0.0)),
            atom_payload("H002", "H", Vec3(1.7, -0.7, 0.0)),
            atom_payload("H003", "H", Vec3(1.9, 0.7, 0.0)),
            atom_payload("H004", "H", Vec3(2.1, -0.7, 0.0)),
        ),
        source_name="retained-ligand-complete-h-without-h-topology",
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = AtomSetScope(
        atom_refs=(AtomRef(ResidueId("A", 1), "OG"),),
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_default_component_library(),
        context_radius_angstrom=3.0,
        retained_non_polymer_chemistry_evidence=evidence,
    )

    with pytest.raises(RefinementError, match="retained non-polymer hydrogen topology"):
        require_atom_scope_continuous_relaxation_execution(atom_scope_facts)


def test_continuous_readiness_accepts_present_retained_ligand_h_topology() -> None:
    """Retained-ligand H topology blocker should clear when expected bonds exist."""

    ligand_residue_id = ResidueId("L", 1)
    structure = build_ser_with_template_less_ligand_structure(
        ligand_atoms=(
            atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
            atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            atom_payload("H001", "H", Vec3(1.5, 0.7, 0.0)),
            atom_payload("H002", "H", Vec3(1.7, -0.7, 0.0)),
            atom_payload("H003", "H", Vec3(1.9, 0.7, 0.0)),
            atom_payload("H004", "H", Vec3(2.1, -0.7, 0.0)),
        ),
        source_name="retained-ligand-complete-h-with-h-topology",
    )
    structure = type(structure).from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "C1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "O1")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "C1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "H001")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "C1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "H002")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "C1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "H003")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "O1")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ligand_residue_id, "H004")
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.REPAIR_INFERRED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_scope = AtomSetScope(
        atom_refs=(AtomRef(ResidueId("A", 1), "OG"),),
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
        snapshot,
        atom_scope,
        component_library=build_default_component_library(),
        context_radius_angstrom=3.0,
        retained_non_polymer_chemistry_evidence=evidence,
    )

    require_atom_scope_continuous_relaxation_execution(atom_scope_facts)


def test_resolve_local_bond_planning_support_propagates_no_rdkit_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-RDKit passive context should expose the required capability failure."""

    monkeypatch.setattr(retained_non_polymer_fallback, "Chem", None)
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="no-rdkit-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    with pytest.raises(RdkitUnavailableError, match="operational RDKit installation"):
        support_resolution_for_residue(
            snapshot,
            residue_id=residue_id,
        )


def test_resolve_local_bond_planning_support_blocks_selected_template_less_ligand() -> (
    None
):
    """Template-less fallback support must not make selected ligands editable."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
                atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
                atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
                atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
            ),
            source_name="selected-template-less-support-blocker",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
        movable_atom_names=("C1",),
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_evidence_keeps_selected_ligand_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit evidence is passive-context support, not movable ligand support."""

    def fail_resolution(*args, **kwargs):
        raise AssertionError("selected evidence-backed ligand resolved chemistry")

    monkeypatch.setattr(
        "protrepair.transformer.continuous.support."
        "resolve_retained_non_polymer_chemistry",
        fail_resolution,
    )
    ligand_residue_id = ResidueId("L", 1)
    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
            ),
            source_name="selected-evidence-template-less-support-blocker",
        )
    )
    evidence = (
        RetainedNonPolymerChemistryEvidence(
            residue_id=ligand_residue_id,
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        ),
    )

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=ligand_residue_id,
        movable_atom_names=("C1",),
        retained_non_polymer_chemistry_evidence=evidence,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert resolution.fallback_bond_definitions == ()


def support_resolution_for_residue(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_id: ResidueId,
    movable_atom_names: tuple[str, ...] = (),
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
) -> LocalBondPlanningSupportResolution:
    """Resolve local bond-planning support for one residue in a test snapshot."""

    constitution = snapshot.structure.constitution
    residue_index = constitution.residue_index(residue_id)
    residue_site = constitution.residue_site_at(residue_index)
    movable_atom_indices = tuple(
        constitution.atom_index(AtomRef(residue_id, atom_name))
        for atom_name in movable_atom_names
    )
    return resolve_local_bond_planning_support(
        snapshot,
        residue_index,
        residue_site,
        movable_atom_indices=movable_atom_indices,
        component_library=build_default_component_library(),
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
        retained_non_polymer_chemistry_evidence=(
            retained_non_polymer_chemistry_evidence
        ),
    )


def build_ser_with_template_less_ligand_structure(
    *,
    ligand_atoms: tuple[CanonicalAtomPayload, ...],
    source_name: str,
) -> ProteinStructure:
    """Build one SER fixture with one template-less retained non-polymer ligand."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=ligand_atoms,
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )
