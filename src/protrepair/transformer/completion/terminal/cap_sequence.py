"""Chemical qualification of the deposited neighbor at a missing peptide port."""

from bisect import bisect_left
from collections import defaultdict
from functools import cache

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.sequence_boundary import PeptideSequenceBoundary


@cache
def linking_component(component: str, required_end: str) -> bool:
    """Whether the local catalog verifies this component's directed peptide linkage."""
    template = build_default_component_library().get(component)
    if template is None or template.component_id != component:
        return False
    elements = None
    if component not in STANDARD_COMPONENT_ATOMS:
        entry = build_bundled_nonstandard_registry().get(component)
        if entry is None or entry.chem_comp_type not in {
            "L-PEPTIDE LINKING",
            "D-PEPTIDE LINKING",
            "PEPTIDE LINKING",
        }:
            return False
        elements = {a.atom_name: a.element for a in entry.atoms}
        if any(
            elements.get(name) != elem
            for name, elem in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))
        ):
            return False
    bonds = {
        (frozenset((b.atom_name_1, b.atom_name_2)), b.order)
        for b in template.definition.bonds
        if not b.aromatic
    }
    if not all(
        (frozenset((a, b)), order) in bonds
        for a, b, order in (("N", "CA", 1), ("CA", "C", 1), ("C", "O", 2))
    ):
        return False
    end_neighbors = template.definition.bonded_atom_names(required_end)
    if required_end == "C":
        return end_neighbors == {"CA", "O", "OXT"}
    # Admit a secondary amine only when its other substituent is saturated
    # carbon. In particular an N-acyl group's carbonyl already uses this port.
    for substituent in end_neighbors - {"CA"}:
        if elements is not None and elements.get(substituent) != "C":
            return False
        if any(
            (b.order != 1 or b.aromatic)
            for b in template.definition.bonds
            if substituent in (b.atom_name_1, b.atom_name_2)
        ):
            return False
    return (
        1 <= len(end_neighbors) <= 2
        and all((frozenset(("N", n)), 1) in bonds for n in end_neighbors)
        and "N" in template.polymer_hydrogen_parent_aliases().values()
    )


def qualified_peptide_neighbors(source: ProteinStructure) -> frozenset[AtomRef]:
    """Return gaps whose every feasible image has a chemically linking neighbor."""
    ingress = source.provenance.ingress
    observation = ingress.observation
    if observation is None:
        return frozenset()
    paths = {p.instance_id: p for p in ingress.deposited_peptide_sequences}
    bad: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for instance, path in paths.items():
        seq = path.components
        for pos, name in enumerate(seq):
            for side, adj, need in (("C", pos + 1, "N"), ("N", pos - 1, "C")):
                if not 0 <= adj < len(seq) or not linking_component(seq[adj], need):
                    bad[instance, name, side].append(pos + 1)

    qualified: set[AtomRef] = set()
    for fact in observation.deposited_peptide_boundaries:
        if fact.kind is PeptideSequenceBoundary.MISSING_CONTINUATION:
            forbidden = bad[fact.instance_id, fact.component_id, fact.port.atom_name]
            i = bisect_left(forbidden, fact.first_position)
            if fact.instance_id in paths and (
                i == len(forbidden) or forbidden[i] > fact.last_position
            ):
                qualified.add(fact.port)
    return frozenset(qualified)
