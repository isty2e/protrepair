"""Advisory source coordination omissions in the current structure."""

import gemmi

from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.structure.aggregate import ProteinStructure


def diagnose_source_coordination_omissions(
    structure: ProteinStructure,
) -> tuple[ValidationIssue, ...]:
    """Report source-declared partners absent beside a retained metal.

    Parameters
    ----------
    structure : ProteinStructure
        Current structure with its immutable original source observation.

    Returns
    -------
    tuple[ValidationIssue, ...]
        Non-blocking warnings. Empty output does not certify site completeness,
        oxidation state, solvent treatment or force-field accuracy.
    """
    observation = structure.provenance.ingress.observation
    if observation is None:
        return ()

    issues: list[ValidationIssue] = []
    originally_excluded = frozenset(observation.excluded_coordination_connections)
    for connection in observation.unrepresented_coordination_connections(
        structure.constitution, structure.geometry
    ):
        for retained, omitted in (
            connection.endpoint_pair(),
            tuple(reversed(connection.endpoint_pair())),
        ):
            index = retained.resolve_index(structure.constitution, structure.geometry)
            if (
                index is None
                or not gemmi.Element(
                    structure.constitution.atom_site_at(index).element
                ).is_metal
            ):
                continue

            metadata = connection.source_metadata
            source = metadata.record_type.value
            if metadata.source_id is not None:
                source += f" {metadata.source_id}"
            if metadata.reported_distance_angstrom is not None:
                source += (
                    f", reported distance {metadata.reported_distance_angstrom:g} A"
                )
            cause = (
                "was excluded during source normalization and remains absent"
                if connection in originally_excluded
                else "is no longer represented with its original source identity"
            )
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.SOURCE_COORDINATION_PARTNER_OMITTED,
                    severity=IssueSeverity.WARNING,
                    scope=EventScope.for_residue_set((retained.atom_ref.residue_id,)),
                    message=(
                        f"{retained.component_id} {retained.atom_ref.display_token()} "
                        f"has a source-declared coordination partner "
                        f"{omitted.component_id} {omitted.atom_ref.display_token()} "
                        f"(altloc {omitted.altloc or '.'}; {source}) that {cause} "
                        "in the current structure; retained topology was preserved, "
                        "and no solvent or metal microstate was inferred"
                    ),
                )
            )
            break
    return tuple(issues)
