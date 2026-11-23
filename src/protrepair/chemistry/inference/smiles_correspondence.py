"""Prove a caller SMILES correspondence before preparing a retained site."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import combinations
from math import isfinite
from typing import TYPE_CHECKING

from protrepair.chemistry.inference.ccd_graph import component_heavy_molecule
from protrepair.chemistry.inference.rdkit_graph import (
    kekulized_molecule,
    require_compatible_named_pose_stereochemistry,
)
from protrepair.chemistry.inference.rdkit_hydrogen_geometry import (
    named_pose_hydrogen_bond_lengths,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    HEAVY_BOND_GEOMETRY_TOLERANCE,
    selected_isolated_hydrogen_parents,
)
from protrepair.chemistry.inference.smiles_model import (
    SMILES_CIP_LIMIT,
    CorrespondenceRefusal,
    NamedProjection,
    strict_smiles_model,
)
from protrepair.chemistry.radii import RadiusKind, prepare_radius_lookup
from protrepair.chemistry.retained_non_polymer.definition import MatchedSmiles
from protrepair.chemistry.retained_non_polymer.evidence import (
    CorrespondenceFailure,
    RefusedSmiles,
)
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.chemistry.retained_non_polymer.isolated_graph import (
    require_source_isolation,
)
from protrepair.errors import BondOrderConstraintError, PoseStereoConflictError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueRole, ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import BondRelationshipType

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


@dataclass(frozen=True, slots=True)
class CorrespondenceLimits:
    """Internal per-site work limits; reaching a limit cannot certify a match."""

    max_bytes: int = 4096
    max_atoms: int = 128
    max_extensions: int = 100_000
    max_candidates: int = 1024


_DEFAULT_LIMITS = CorrespondenceLimits()


@dataclass(frozen=True, slots=True)
class _SiteEvidence:
    residue: ResidueSite
    geometry: ResidueGeometry
    elements: dict[str, str]
    charges: dict[str, int]
    required: set[tuple[str, str]]
    orders: dict[tuple[str, str], int]
    anchors: dict[str, str]

    @classmethod
    def read(
        cls,
        observation: StructureObservation,
        rid: ResidueId,
        limits: CorrespondenceLimits,
    ) -> "_SiteEvidence":
        c = observation.constitution
        site = c.residue_or_ligand(rid)
        if site is None:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.INCOMPATIBLE,
                "site absent in the original observation",
            )
        if len(site.atom_sites) > 4 * limits.max_atoms:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.SEARCH_LIMIT, "site atom limit"
            )
        ri = c.residue_index(rid)
        pose = observation.geometry.residue_geometry(constitution=c, residue_index=ri)
        elements = {a.name: a.element for a in site.atom_sites if not a.is_hydrogen()}
        if len(elements) > limits.max_atoms:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.SEARCH_LIMIT, "site heavy atom limit"
            )
        if any(
            not all(isfinite(x) for x in pose.position(a.name)) for a in site.atom_sites
        ):
            raise CorrespondenceRefusal(
                CorrespondenceFailure.INDETERMINATE, "nonfinite site coordinates"
            )
        charges = {}
        for a in site.atom_sites:
            q = observation.topology.formal_charge(c.atom_index(AtomRef(rid, a.name)))
            if q is not None:
                if a.is_hydrogen() and q != 0:
                    raise CorrespondenceRefusal(
                        CorrespondenceFailure.INCOMPATIBLE, f"charged H/D/T at {a.name}"
                    )
                charges[a.name] = q
        required = set()
        orders = {}
        anchors = {}
        for b in observation.bonds_for_residue(rid):
            if b.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            p, r = (c.atom_ref_at(i) for i in b.endpoint_pair())
            if p.residue_id != r.residue_id or b.relationship_type not in (
                BondRelationshipType.UNKNOWN,
                BondRelationshipType.COVALENT,
            ):
                raise CorrespondenceRefusal(
                    CorrespondenceFailure.INCOMPATIBLE,
                    f"nonisolated or incompatible attachment at {p.display_token()}--{r.display_token()}",
                )
            a, z = sorted((p.atom_name, r.atom_name))
            if a in elements and z in elements:
                required.add((a, z))
                if b.order is not None:
                    orders[a, z] = b.order
            else:
                h, pa = (z, a) if a in elements else (a, z)
                if (
                    pa not in elements
                    or b.order not in (None, 1)
                    or anchors.setdefault(h, pa) != pa
                ):
                    raise CorrespondenceRefusal(
                        CorrespondenceFailure.INCOMPATIBLE,
                        f"conflicting source H/D/T attachment at {h}",
                    )
        return cls(site, pose, elements, charges, required, orders, anchors)

    def mappings(
        self, mol: "Mol", limits: CorrespondenceLimits
    ) -> Iterator[tuple[str, ...]]:
        elements = tuple(a.GetSymbol().upper() for a in mol.GetAtoms())
        if Counter(elements) != Counter(self.elements.values()):
            return
        radii = prepare_radius_lookup(elements, RadiusKind.COVALENT)
        radii.require_complete("SMILES correspondence")
        possible = {n: set() for n in self.elements}
        for a, b in combinations(sorted(self.elements), 2):
            distance = self.geometry.position(a).distance_to(self.geometry.position(b))
            expected = radii.radius_angstrom(self.elements[a]) + radii.radius_angstrom(
                self.elements[b]
            )
            if abs(distance - expected) <= HEAVY_BOND_GEOMETRY_TOLERANCE:
                possible[a].add(b)
                possible[b].add(a)
        required = {n: set() for n in self.elements}
        for a, b in self.required:
            required[a].add(b)
            required[b].add(a)
        neigh = tuple({n.GetIdx() for n in a.GetNeighbors()} for a in mol.GetAtoms())
        anchors = Counter(self.anchors.values())
        domains = tuple(
            tuple(
                sorted(
                    n
                    for n, el in self.elements.items()
                    if el == elements[i]
                    and self.charges.get(n, a.GetFormalCharge()) == a.GetFormalCharge()
                    and len(required[n]) <= len(neigh[i]) <= len(possible[n])
                    and anchors[n] <= a.GetTotalNumHs()
                    and not (
                        Counter(elements[j] for j in neigh[i])
                        - Counter(self.elements[j] for j in possible[n])
                    )
                )
            )
            for i, a in enumerate(mol.GetAtoms())
        )
        if not all(domains):
            return
        order = []
        remaining = set(range(len(elements)))
        while remaining:
            i = min(
                remaining,
                key=lambda j: (
                    -sum(k in neigh[j] for k in order),
                    len(domains[j]),
                    -len(neigh[j]),
                    j,
                ),
            )
            order.append(i)
            remaining.remove(i)
        current: dict[int, str] = {}
        used: set[str] = set()
        extensions = 0

        def extend(depth: int) -> Iterator[tuple[str, ...]]:
            nonlocal extensions
            if depth == len(order):
                yield tuple(current[i] for i in range(len(order)))
                return
            i = order[depth]
            for n in domains[i]:
                extensions += 1
                if extensions > limits.max_extensions:
                    raise CorrespondenceRefusal(
                        CorrespondenceFailure.SEARCH_LIMIT,
                        "partial-assignment search limit",
                    )
                if n in used or any(
                    (j in neigh[i] and m not in possible[n])
                    or (m in required[n] and j not in neigh[i])
                    for j, m in current.items()
                ):
                    continue
                current[i] = n
                used.add(n)
                yield from extend(depth + 1)
                used.remove(n)
                del current[i]

        yield from extend(0)

    def admits(self, mol: "Mol", names: tuple[str, ...]) -> bool:
        if self.orders:
            idx = {n: i for i, n in enumerate(names)}
            try:
                kekulized_molecule(
                    mol,
                    bond_order_constraints={
                        (idx[a], idx[b]): o for (a, b), o in self.orders.items()
                    },
                )
            except BondOrderConstraintError:
                return False
        if self.anchors:
            lengths = named_pose_hydrogen_bond_lengths(mol, names, self.geometry)
            for h, parent in self.anchors.items():
                d = self.geometry.position(h).distance_to(
                    self.geometry.position(parent)
                )
                if not any(
                    p == parent and source_hydrogen_bond_length_is_compatible(d, length)
                    for p, length in lengths
                ):
                    return False
        try:
            require_compatible_named_pose_stereochemistry(
                mol,
                names,
                self.geometry,
                hydrogen_attachments=tuple(sorted(self.anchors.items())),
            )
        except PoseStereoConflictError:
            return False
        return True


def bind_smiles_assignment(
    structure: ProteinStructure,
    residue_id: ResidueId,
    smiles: str,
    *,
    limits: CorrespondenceLimits = _DEFAULT_LIMITS,
) -> MatchedSmiles | RefusedSmiles:
    """Resolve one selected site using original declarations before the live carrier.

    Parameters
    ----------
    structure : ProteinStructure
        Normalized workflow input.
    residue_id : ResidueId
        Address of one retained ligand.
    smiles : str
        Caller's molecular graph.
    limits : CorrespondenceLimits
        Work budget for this occurrence.

    Returns
    -------
    MatchedSmiles or RefusedSmiles
        Named chemistry or a site-local refusal which takes precedence over fallback.

    Raises
    ------
    ValueError
        The SMILES syntax is invalid or the site is absent.
    RdkitUnavailableError
        The required native backend is unavailable.
    """
    site = structure.constitution.residue_or_ligand(residue_id)
    if (
        site is None
        or structure.constitution.role_at(
            structure.constitution.residue_index(residue_id)
        )
        is not ResidueRole.LIGAND
    ):
        raise ValueError(
            f"SMILES assignment must target a kept ligand at {residue_id!r}"
        )
    need_h = None
    try:
        mol = strict_smiles_model(
            smiles, max_bytes=limits.max_bytes, max_atoms=limits.max_atoms
        )
        need_h = any(a.GetTotalNumHs() for a in mol.GetAtoms())
        if len(site.atom_sites) > 4 * limits.max_atoms:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.SEARCH_LIMIT, "current site atom limit"
            )
        try:
            require_source_isolation(structure, residue_id, "caller SMILES")
        except ValueError as error:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.INCOMPATIBLE, str(error)
            ) from error
        observation = structure.provenance.ingress.observation
        from_supplied_model = observation is None
        if observation is None:
            observation = StructureObservation.from_source_facets(
                constitution=structure.constitution,
                geometry=structure.geometry,
                topology=structure.topology,
            )
            # A hand-built carrier can still have template-derived bonds. Its
            # reported-only view also owns the independent check of deposited H.
            structure = structure.with_provenance(
                replace(
                    structure.provenance,
                    ingress=replace(
                        structure.provenance.ingress, observation=observation
                    ),
                )
            )
        source = _SiteEvidence.read(observation, residue_id, limits)
        if from_supplied_model:
            source = replace(
                source,
                orders=dict(
                    structure.topology.residue_source_heavy_bond_orders(
                        constitution=structure.constitution,
                        residue_index=structure.constitution.residue_index(residue_id),
                    )
                ),
            )
        match = _resolve_class(structure, site, source, mol, limits)
        return replace(match, from_supplied_model=from_supplied_model)
    except CorrespondenceRefusal as error:
        return RefusedSmiles(residue_id, error.cause, f"caller SMILES: {error}", need_h)


def _resolve_class(
    structure: ProteinStructure,
    site: ResidueSite,
    source: _SiteEvidence,
    mol: "Mol",
    limits: CorrespondenceLimits,
) -> MatchedSmiles:
    reference = None
    best = None
    best_key = None
    index = structure.constitution.residue_index(site.residue_id)
    charges = {
        n: q
        for n, q in structure.residue_formal_charge_by_atom_name(index)
        if q is not None
    }
    orders = {
        tuple(sorted(pair)): k
        for pair, k in structure.topology.residue_source_heavy_bond_orders(
            constitution=structure.constitution, residue_index=index
        ).items()
    }
    for count, names in enumerate(source.mappings(mol, limits), 1):
        if count > limits.max_candidates:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.SEARCH_LIMIT,
                "complete-candidate assessment limit",
            )
        try:
            if not source.admits(mol, names):
                continue
            literal = NamedProjection.from_mapping(mol, names)
            equivalence = literal.oxygen_class()
        except CorrespondenceRefusal:
            raise
        except (ValueError, RuntimeError) as error:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.INDETERMINATE,
                f"candidate could not be assessed: {error}",
            ) from error
        if reference is not None and reference != equivalence:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.AMBIGUOUS,
                f"ambiguous named chemistry at {reference.difference(equivalence)}; supply atom names or a CCD",
            )
        reference = equivalence
        key = literal.publication_key()
        if best_key is not None and key <= best_key:
            continue
        # A class can differ only at terminal oxygen charges and integral orders.
        # Check just these constraints to select a whole literal, never the class.
        if any(charges.get(n, q) != q for n, _, q, _, _, _ in literal.atoms):
            continue
        if any(k != 4 and orders.get((a, b), k) != k for a, b, k, _ in literal.bonds):
            continue
        best, best_key = (literal, names), key

    if reference is None:
        raise CorrespondenceRefusal(
            CorrespondenceFailure.INCOMPATIBLE,
            "no complete mapping satisfies the deposited atoms, bonds, H anchors and charges",
        )
    if best is None:
        raise CorrespondenceRefusal(
            CorrespondenceFailure.INCOMPATIBLE,
            "no localization is compatible with the current formal charges and reported bond orders",
        )
    literal, names = best
    reserved = {n for r in (source.residue, site) for n in r.atom_site_names()}
    reserved.update("H" + n[1:] for n in tuple(reserved) if n.startswith(("D", "T")))
    sites = tuple(
        sorted((n, source.geometry.atom_geometry(n).altloc) for n in source.elements)
    )
    try:
        definition = literal.named_graph(
            source.residue.component_id, mol, names, reserved
        )
        reconstructed = NamedProjection.from_mapping(
            component_heavy_molecule(definition, cip_limit=SMILES_CIP_LIMIT),
            definition.heavy_atom_names,
        )
        if reconstructed != literal:
            raise CorrespondenceRefusal(
                CorrespondenceFailure.INDETERMINATE,
                "named graph could not preserve specified chemistry or stereo",
            )
        candidate = MatchedSmiles(site.residue_id, definition, sites, ())
        parents = selected_isolated_hydrogen_parents(candidate, structure)
    except CorrespondenceRefusal:
        raise
    except (ValueError, RuntimeError) as error:
        raise CorrespondenceRefusal(
            CorrespondenceFailure.INDETERMINATE,
            f"could not validate the named model: {error}",
        ) from error
    assert parents is not None
    pose = structure.residue_geometry(index)
    hydrogens = tuple(
        sorted(
            (h, site.atom_site(h).element, pose.atom_geometry(h).altloc, p)
            for h, p in parents.items()
        )
    )
    return replace(candidate, observed_hydrogens=hydrogens)
