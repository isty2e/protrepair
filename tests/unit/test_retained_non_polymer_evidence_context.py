"""Only explicitly admitted binding chemistry can extend residue-local evidence."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_external_bonds,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.scope import WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)


def _evidence():
    return RetainedNonPolymerChemistryEvidence(ResidueId("L", 1), "CO", ("C1", "O1"))


def _source(relationship=None, provenance=BondProvenance.SOURCE_EXPLICIT, *, extra=()):
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="XXX",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("C1", "C", Vec3(-1.43, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(0.0, 0.0, 0.0)),
                ),
            ),
            residue_payload(
                component_id="YYY",
                residue_id=ResidueId("L", 2),
                is_hetero=True,
                atoms=(atom_payload("C2", "C", Vec3(1.43, 0.0, 0.0)),),
            ),
            *extra,
        ),
        source_format=FileFormat.MMCIF,
    )
    if relationship is None:
        return source
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    source.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1")),
                    source.constitution.atom_index(AtomRef(ResidueId("L", 2), "C2")),
                    relationship_type=relationship,
                    provenance=provenance,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.UNKNOWN,
    ),
)
@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_local_override_cannot_admit_an_external_covalent_graph(
    relationship, provenance
):
    source = _source(relationship, provenance)
    evidence = _evidence()
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
        evidence=evidence,
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "external" in resolution.failure_reason
    assert not resolution.expected_hydrogen_atom_names

    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any("external" in issue.message for issue in result.issues)


@pytest.mark.parametrize("topology_only", (False, True))
def test_completed_local_override_is_refused_atomically_with_external_link(
    topology_only,
):
    isolated = _source()
    evidence = _evidence()
    prepared = add_retained_non_polymer_hydrogens(
        isolated,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert not prepared.issues
    current = prepared.structure
    external = TopologyBond(
        current.constitution.atom_index(AtomRef(evidence.residue_id, "O1")),
        current.constitution.atom_index(AtomRef(ResidueId("L", 2), "C2")),
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    linked = ProteinStructure.from_payload(
        constitution=current.constitution,
        geometry=current.geometry,
        topology=StructureTopology(
            constitution=current.constitution,
            atom_topologies=current.topology.atom_topologies,
            bonds=(*current.topology.bonds, external),
        ),
        polymer_blueprint=current.polymer_blueprint,
        provenance=current.provenance,
    )
    transformer = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = transformer(
        linked,
        build_default_component_library(),
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert result.structure is linked and result.issues and not result.repairs


@pytest.mark.parametrize(
    "relationship",
    (None, BondRelationshipType.HYDROGEN_BOND),
)
def test_local_override_keeps_isolated_and_typed_noncovalent_contexts(relationship):
    source = _source(relationship)
    evidence = _evidence()
    assert retained_non_polymer_evidence_external_bonds(evidence, source) == ()
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
        evidence=evidence,
    )
    assert (
        resolution.source is RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert not result.issues
    ligand = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in ligand.atom_sites) == 4
    if relationship is not None:
        bond = result.structure.topology.bond_between(
            result.structure.constitution.atom_index(
                AtomRef(evidence.residue_id, "O1")
            ),
            result.structure.constitution.atom_index(AtomRef(ResidueId("L", 2), "C2")),
        )
        assert bond is not None and bond.relationship_type is relationship


def test_unsupported_typed_coordination_preserves_source_without_organic_success():
    source = _source(BondRelationshipType.METAL_COORDINATION)
    evidence = _evidence()
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
        evidence=evidence,
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "unsupported donor/metal motif" in resolution.failure_reason
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert result.structure is source
    assert result.issues
    assert not result.repairs


def test_local_override_accepts_intra_residue_covalent_topology():
    source = _source()
    evidence = _evidence()
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert not result.issues
    assert (
        retained_non_polymer_evidence_external_bonds(evidence, result.structure) == ()
    )
    repeated = resolve_retained_non_polymer_topology(
        result.structure,
        build_default_component_library(),
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset((evidence.residue_id,)),
    )
    assert repeated.structure is result.structure and not repeated.issues


@pytest.mark.parametrize("strict", (False, True))
def test_public_workflow_does_not_hydrogenate_a_linked_isolated_override(strict):
    source = _source(BondRelationshipType.COVALENT)
    evidence = _evidence()
    result = process_structure(
        source,
        ingress=StructureIngressOptions(
            ligand_policy=LigandPolicy.KEEP,
            retained_non_polymer_chemistry_overrides=(
                RetainedNonPolymerChemistryOverride(
                    evidence.residue_id, evidence.smiles, evidence.heavy_atom_names
                ),
            ),
        ),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
        transform_requests=WorkflowTransformRequests(
            allow_retained_non_polymer_rdkit_fallback=not strict
        ),
    )
    residue = result.structure.constitution.residue_or_ligand(evidence.residue_id)
    assert residue is not None
    assert not any(atom.is_hydrogen() for atom in residue.atom_sites)
    assert result.requested_goals_fully_satisfied() is False


def test_unsupported_linked_override_does_not_block_independent_evidence():
    isolated = residue_payload(
        component_id="ZZZ",
        residue_id=ResidueId("L", 3),
        is_hetero=True,
        atoms=(
            atom_payload("C3", "C", Vec3(20.0, 0.0, 0.0)),
            atom_payload("O3", "O", Vec3(21.43, 0.0, 0.0)),
        ),
    )
    source = _source(BondRelationshipType.COVALENT, extra=(isolated,))
    evidence = (
        _evidence(),
        RetainedNonPolymerChemistryEvidence(ResidueId("L", 3), "CO", ("C3", "O3")),
    )
    targets = frozenset(item.residue_id for item in evidence)
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=evidence,
        target_residue_ids=targets,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure.constitution.ligands[0] == source.constitution.ligands[0]
    assert result.issues and all("external" in issue.message for issue in result.issues)
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[2].atom_sites
        )
        == 4
    )
    resolved = resolve_retained_non_polymer_topology(
        result.structure,
        build_default_component_library(),
        chemistry_evidence=evidence,
        target_residue_ids=targets,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert resolved.structure is result.structure and resolved.issues


@pytest.mark.parametrize("smiles", ("C=O", "[C-]#[O+]"))
def test_explicit_override_cannot_drop_excess_or_zero_inventory_source_hydrogens(
    smiles,
):
    evidence = _evidence()
    prepared = add_retained_non_polymer_hydrogens(
        _source(),
        chemistry_evidence=(evidence,),
        target_residue_ids=frozenset({evidence.residue_id}),
    )
    assert not prepared.issues
    source = prepared.structure
    override = RetainedNonPolymerChemistryEvidence(
        evidence.residue_id, smiles, evidence.heavy_atom_names
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        chemistry_evidence=(override,),
        target_residue_ids=frozenset({override.residue_id}),
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any(
        issue.kind
        in {
            ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
            ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        }
        for issue in result.issues
    )
