"""Committed side-chain packing workflow action proposal adapters."""

from protrepair.diagnostics.parser_readability import (
    rdkit_no_conect_extra_proximity_bond_clusters,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_partner_residue_ids_by_residue,
)
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import ResidueId
from protrepair.transformer.packing.spec import PackingMode, PackingScope, PackingSpec
from protrepair.workflow.actions.local_refinement import LocalRefinementTransformer
from protrepair.workflow.actions.packing import CommittedPackingTransformer
from protrepair.workflow.planning.action.domain import WorkflowActionDomain
from protrepair.workflow.planning.action.proposals import WorkflowActionProposal
from protrepair.workflow.planning.default_action_registry.capabilities import (
    COMMITTED_PACKING_CAPABILITY,
)

__all__ = [
    "committed_packing_is_admissible",
    "committed_packing_proposals",
]


def committed_packing_is_admissible(domain: WorkflowActionDomain) -> bool:
    """Return whether committed packing is admissible in the active domain."""

    if (
        CommittedPackingTransformer.planned_candidate(
            domain.structure,
            transform_requests=domain.transform_requests,
            component_library=domain.component_library,
        )
        is not None
    ):
        return True

    return bool(_automatic_committed_packing_specs(domain))


def committed_packing_proposals(
    domain: WorkflowActionDomain,
) -> tuple[WorkflowActionProposal, ...]:
    """Return explicit or automatic committed-packing proposals."""

    committed_packing_transformer = CommittedPackingTransformer.planned_candidate(
        domain.structure,
        transform_requests=domain.transform_requests,
        component_library=domain.component_library,
    )
    if committed_packing_transformer is not None:
        return (
            WorkflowActionProposal(
                transformer=committed_packing_transformer,
                capability=COMMITTED_PACKING_CAPABILITY,
                explicitly_requested=True,
            ),
        )

    return tuple(
        WorkflowActionProposal(
            transformer=CommittedPackingTransformer.from_planned_spec(
                packing_spec
            ),
            capability=COMMITTED_PACKING_CAPABILITY,
            explicitly_requested=False,
        )
        for packing_spec in _automatic_committed_packing_specs(domain)
    )


def _automatic_committed_packing_specs(
    domain: WorkflowActionDomain,
) -> tuple[PackingSpec, ...]:
    """Return automatic side-chain packing proposals after local FF stalls."""

    if domain.requests_explicit_repair_refinement():
        return ()

    parser_compatibility_facts = domain.parser_compatibility_facts
    if (
        parser_compatibility_facts is None
        or not parser_compatibility_facts.has_parser_visible_proximity_burden()
    ):
        return ()
    if (
        domain.chemistry_readiness_facts.hydrogen_coverage_state
        is not HydrogenCoverageState.COMPLETE
    ):
        return ()
    if not domain.has_adopted_action_family(LocalRefinementTransformer):
        return ()
    if domain.has_adopted_action_family(CommittedPackingTransformer):
        return ()

    residue_ids = _automatic_packing_parser_burden_residue_ids(domain)
    if not residue_ids:
        return ()

    return (
        PackingSpec(
            backend_name="faspr",
            mode=PackingMode.PACK,
            scope=PackingScope.LOCAL,
            mutable_residue_ids=residue_ids,
        ),
    )


def _automatic_packing_parser_burden_residue_ids(
    domain: WorkflowActionDomain,
) -> tuple[ResidueId, ...]:
    """Return parser-burden residues for one automatic packing proposal."""

    clusters = rdkit_no_conect_extra_proximity_bond_clusters(
        domain.structure,
        component_library=domain.component_library,
    )
    if not clusters:
        return ()

    residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for residue_id in clusters[0].residue_ids:
        residue_site = domain.structure.constitution.residue_or_ligand(residue_id)
        if residue_site is None or residue_site.is_hetero:
            continue

        residue_ids.append(residue_id)
        seen_residue_ids.add(residue_id)

    ambiguous_partners = ambiguous_disulfide_partner_residue_ids_by_residue(
        domain.structure
    )
    for residue_id in tuple(residue_ids):
        for partner_residue_id in sorted(
            ambiguous_partners.get(residue_id, frozenset()),
            key=lambda candidate: candidate.display_token(),
        ):
            if partner_residue_id in seen_residue_ids:
                continue

            partner_site = domain.structure.constitution.residue_or_ligand(
                partner_residue_id
            )
            if partner_site is None or partner_site.is_hetero:
                continue

            residue_ids.append(partner_residue_id)
            seen_residue_ids.add(partner_residue_id)

    return tuple(residue_ids)
