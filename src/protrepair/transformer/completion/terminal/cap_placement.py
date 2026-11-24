"""Admit one two-sided peptide-cap product on the exact current carrier."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.water_contacts import stationary_water_refusal
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import BoundPeptidePorts
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.sequence_boundary import PeptideSequenceBoundary
from protrepair.transformer.completion.hydrogen.linked_caps import (
    stage_complete_cap_topology,
)
from protrepair.transformer.completion.terminal.ace_selection import (
    GeneratedAceSelection,
)
from protrepair.transformer.completion.terminal.cap_contacts import PeptideCapContacts
from protrepair.transformer.completion.terminal.cap_product import (
    StagedPeptideCap,
    frame_for,
    neighbors,
    occupied_cap,
)


class PortCondition(str, Enum):
    """Current chemical state, independently of any attempt on another branch."""

    OPEN = "open"
    RECONCILE = "check_existing_cap"
    COMPLETE = "already_capped"
    OCCUPIED = "occupied"
    OMITTED = "omitted"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class CapPortAssessment:
    """Observed condition plus the existing or reserved cap address, when known."""

    condition: PortCondition
    reason: str
    cap_id: ResidueId | None = None
    current_partner: AtomRef | None = None

    def requires_action(self) -> bool:
        return self.condition in (PortCondition.OPEN, PortCondition.RECONCILE)

    @classmethod
    def observe(
        cls, binding: BoundPeptidePorts, parent: ProteinStructure, port: AtomRef
    ) -> "CapPortAssessment":
        """Reconcile input evidence with the current output; never infer a gap here."""
        source = binding.source
        ingress = source.provenance.ingress
        current = parent.provenance.ingress
        observation = ingress.observation
        c = parent.constitution
        old = source.constitution.residue_or_ligand(port.residue_id)
        site = c.residue_or_ligand(port.residue_id)
        if old is None:
            return cls(PortCondition.REFUSED, "absent_at_input")
        if (
            source.constitution.role_at(
                source.constitution.residue_index(port.residue_id)
            )
            is not ResidueRole.POLYMER
        ):
            return cls(PortCondition.REFUSED, "not_polymer")
        if (
            site is None
            or site.component_id != old.component_id
            or c.role_at(c.residue_index(port.residue_id)) is not ResidueRole.POLYMER
            or (
                current is not ingress
                and (observation is None or current.observation is not observation)
            )
        ):
            return cls(PortCondition.REFUSED, "changed_identity")
        if observation is not None and port in observation.unresolved_conect_ports:
            return cls(PortCondition.REFUSED, "unresolved_conect")
        if (
            ingress.uncertain_peptide_link_ports is not None
            and port in ingress.uncertain_peptide_link_ports
        ):
            return cls(PortCondition.REFUSED, "uncertain_source_connection")
        for omitted in parent.provenance.omitted_peptide_caps:
            if omitted.port == port and omitted.identifies_port_in(c):
                return cls(PortCondition.OMITTED, "omitted_cap_partner", omitted.cap_id)
        if observation is None or ingress.uncertain_peptide_link_ports is None:
            return cls(PortCondition.REFUSED, "uninspected_source")
        if port in observation.unresolved_polymer_endpoints:
            return cls(PortCondition.REFUSED, "unresolved_or_omitted_partner")
        try:
            existing = occupied_cap(parent, port)
        except ValueError as error:
            return cls(PortCondition.REFUSED, str(error), binding.attached.get(port))
        if existing is not None:
            why, found = existing
            if port in binding.attached and binding.attached[port] != found:
                return cls(PortCondition.REFUSED, "changed_cap_partner")
            if any(
                other.residue_id not in (found, port.residue_id)
                for other, _ in neighbors(observation, port)
            ):
                return cls(PortCondition.REFUSED, "competing_source_partner")
            if why == "already_capped":
                return cls(PortCondition.COMPLETE, why, found)
            if binding.attached.get(port) == found:
                return cls(PortCondition.RECONCILE, why, found)
            return cls(PortCondition.REFUSED, why, found)
        if port in binding.attached:
            return cls(
                PortCondition.REFUSED, "lost_cap_partner", binding.attached[port]
            )

        for snapshot in (parent, observation):
            partners = tuple(
                other
                for other, _ in neighbors(snapshot, port)
                if other.residue_id != port.residue_id
            )
            if partners:
                actual = (
                    partners[0] if snapshot is parent and len(partners) == 1 else None
                )
                return cls(
                    PortCondition.OCCUPIED, "occupied_port", current_partner=actual
                )
        fact = binding.facts.get(port)
        if fact is None:
            return cls(PortCondition.REFUSED, "no_gap_evidence")
        if fact.kind is PeptideSequenceBoundary.CONSTRUCT_END:
            return cls(PortCondition.REFUSED, "construct_end")
        if port not in binding.qualified_gap_ports:
            return cls(PortCondition.REFUSED, "unknown_neighbor")
        cap_id = binding.cap_ids.get(port)
        if cap_id is None or c.residue_or_ligand(cap_id) is not None:
            return cls(PortCondition.REFUSED, "unavailable_address")
        return cls(PortCondition.OPEN, "missing_continuation", cap_id)


@dataclass(frozen=True, slots=True)
class PeptideCapPlacement:
    """An all-or-nothing result for one port, including exhausted pose evidence."""

    structure: ProteinStructure
    reason: str
    attempted: int = 0
    refusals: tuple[tuple[str, int], ...] = ()
    detail: str = ""

    @classmethod
    def execute(
        cls,
        binding: BoundPeptidePorts,
        parent: ProteinStructure,
        port: AtomRef,
        cap_id: ResidueId,
        library: ComponentLibrary,
    ) -> "PeptideCapPlacement":
        """Stage all chemistry and geometry before publishing either side of the link."""
        state = CapPortAssessment.observe(binding, parent, port)
        if not state.requires_action() or state.cap_id != cap_id:
            return cls(parent, state.reason)
        if state.condition is PortCondition.RECONCILE:
            try:
                staged = stage_complete_cap_topology(parent, port, cap_id)
                if staged is not parent and occupied_cap(staged, port) == (
                    "already_capped",
                    cap_id,
                ):
                    return cls(staged, "reconciled_existing")
            except ValueError as error:
                return cls(parent, state.reason, detail=str(error))
            return cls(parent, state.reason)

        rejected: Counter[str] = Counter()
        attempted = 0
        detail = ""
        try:
            frame = frame_for(parent, port)
            before = {
                pair: (depth, uncertain)
                for pair, depth, uncertain in PeptideCapContacts(
                    parent, port, cap_id, library
                ).clashes(parent)
            }

            def admitted() -> Iterator[ProteinStructure]:
                nonlocal attempted, detail
                staged = None
                contacts = None
                for pose in frame.poses():
                    attempted += 1
                    try:
                        frame.verify(pose)
                    except ValueError as error:
                        rejected[str(error)] += 1
                        continue
                    if staged is None:
                        staged = StagedPeptideCap.build(
                            parent, port, cap_id, frame, pose
                        )
                        contacts = PeptideCapContacts(
                            staged.structure, port, cap_id, library
                        )
                        candidate = staged.structure
                    else:
                        candidate = staged.at(pose)
                    assert contacts is not None
                    for pair, depth, uncertain in contacts.clashes(candidate):
                        if all(ref.residue_id == cap_id for ref in pair):
                            continue
                        previous = before.get(pair)
                        if (
                            previous is None
                            or depth > previous[0] + 1.0e-6
                            or (uncertain and not previous[1])
                        ):
                            rejected[
                                "unresolved_h_contact"
                                if uncertain
                                else "steric_contact"
                            ] += 1
                            detail = " / ".join(ref.display_token() for ref in pair)
                            break
                    else:
                        refusal = stationary_water_refusal(
                            parent,
                            candidate,
                            run_input=binding.source,
                            component_library=library,
                        )
                        if refusal is not None:
                            rejected["stationary_water"] += 1
                            detail = refusal.message
                            continue
                        if occupied_cap(candidate, port) != ("already_capped", cap_id):
                            raise ValueError("unverified_two_sided_product")
                        yield candidate

            poses = admitted()
            reason = "no_acceptable_pose"
            if port.atom_name == "N" and frame.n_substituent is None:
                choice = GeneratedAceSelection.from_verified_poses(
                    parent, poses, flank_id=port.residue_id, cap_id=cap_id
                )
                reasons = tuple(r.value for r in choice.rejected)
                rejected.update(reasons)
                if reasons:
                    reason = "backbone_withheld"
                    detail = ", ".join(dict.fromkeys(reasons))
                selected = choice.structure if choice.accepted is not None else None
            else:
                selected = next(poses, None)
            if selected is not None:
                return cls(
                    selected, "placed", attempted, tuple(sorted(rejected.items()))
                )
        except ValueError as error:
            reason = str(error)
        return cls(parent, reason, attempted, tuple(sorted(rejected.items())), detail)
