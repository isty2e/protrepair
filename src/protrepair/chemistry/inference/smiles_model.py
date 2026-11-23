"""Loss-aware parsing and name-indexed chemical projections of caller SMILES."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
except ImportError:  # pragma: no cover
    Chem = None
    rdCIPLabeler = None

from protrepair.chemistry.retained_non_polymer.definition import (
    NAMED_COMPONENT_ELEMENTS,
    ComponentAtom,
    ComponentBond,
    NamedComponentGraph,
)
from protrepair.chemistry.retained_non_polymer.evidence import CorrespondenceFailure
from protrepair.errors import RdkitUnavailableError

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


SMILES_CIP_LIMIT = 100_000


class CorrespondenceRefusal(ValueError):
    """A site-local refusal, distinct from malformed request syntax."""

    def __init__(self, cause: CorrespondenceFailure, detail: str) -> None:
        super().__init__(detail)
        self.cause = cause


def strict_smiles_model(smiles: str, *, max_bytes: int, max_atoms: int) -> "Mol":
    """Read one complete closed-shell molecule without erasing user declarations."""
    if len(smiles) > max_bytes or len(smiles.encode()) > max_bytes:
        raise CorrespondenceRefusal(
            CorrespondenceFailure.SEARCH_LIMIT, "SMILES input size limit"
        )
    if Chem is None:
        raise RdkitUnavailableError("SMILES correspondence requires RDKit")
    params = Chem.SmilesParserParams()
    params.removeHs = params.allowCXSMILES = params.parseName = params.sanitize = False
    if "|" in smiles:
        # Parse the extension only to distinguish valid unsupported syntax from an error.
        params.allowCXSMILES = True
        params.strictCXSMILES = True
        if Chem.MolFromSmiles(smiles, params) is None:
            raise ValueError("invalid SMILES syntax for assignment")
        raise CorrespondenceRefusal(
            CorrespondenceFailure.UNSUPPORTED, "CXSMILES is not supported"
        )
    raw = Chem.MolFromSmiles(smiles, params)
    if raw is None:
        raise ValueError("invalid SMILES syntax for assignment")
    if raw.GetNumAtoms() > max_atoms:
        raise CorrespondenceRefusal(
            CorrespondenceFailure.SEARCH_LIMIT, "SMILES atom limit"
        )
    unsupported = CorrespondenceFailure.UNSUPPORTED
    if raw.GetNumAtoms() < 2 or len(Chem.GetMolFrags(raw)) != 1:
        raise CorrespondenceRefusal(
            unsupported, "expected one connected multi-heavy-atom molecule"
        )
    for a in raw.GetAtoms():
        if (
            a.GetSymbol().upper() not in NAMED_COMPONENT_ELEMENTS
            or a.GetAtomicNum() == 1
            or a.GetIsotope()
            or a.HasProp("molAtomMapNumber")
            or a.HasQuery()
        ):
            raise CorrespondenceRefusal(
                unsupported,
                "unsupported model element, separate H, isotope or atom map",
            )
        if a.GetChiralTag() and (
            a.GetChiralTag()
            not in (
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            )
            or not (
                a.GetDegree() == 4 or (a.GetDegree() == 3 and a.GetNumExplicitHs() == 1)
            )
        ):
            raise CorrespondenceRefusal(
                unsupported, "unsupported chiral center or lone-pair stereo"
            )
    if any(
        b.HasQuery()
        or b.GetBondType()
        not in (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        )
        for b in raw.GetBonds()
    ):
        raise CorrespondenceRefusal(unsupported, "unsupported bond type")

    molecule = Chem.Mol(raw)
    try:
        Chem.SanitizeMol(molecule)
        Chem.SetBondStereoFromDirections(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    except (ValueError, RuntimeError) as error:
        raise CorrespondenceRefusal(
            unsupported, f"cannot represent model valence or stereo: {error}"
        ) from error
    for a, b in zip(raw.GetAtoms(), molecule.GetAtoms(), strict=True):
        if (
            b.GetNumRadicalElectrons()
            or a.GetFormalCharge() != b.GetFormalCharge()
            or (a.GetChiralTag() and not b.GetChiralTag())
        ):
            raise CorrespondenceRefusal(
                unsupported,
                "sanitization changed declared charge/stereo or left a radical",
            )
    for b in raw.GetBonds():
        if b.GetBondDir() and (
            not molecule.GetBondWithIdx(b.GetIdx()).GetBondDir()
            or not any(
                adj.GetStereo()
                for idx in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for adj in molecule.GetAtomWithIdx(idx).GetBonds()
                if adj.GetBondType() == Chem.BondType.DOUBLE
            )
        ):
            raise CorrespondenceRefusal(
                unsupported, "bond direction does not describe a supported stereo bond"
            )
    # Canonicalize traversal before charging search work, never the deposited names.
    params.sanitize = True
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(molecule), params)
    if canonical is None:
        raise CorrespondenceRefusal(
            unsupported, "cannot reconstitute the sanitized molecular graph"
        )
    return canonical


@dataclass(frozen=True, slots=True)
class NamedProjection:
    """A candidate in deposited-name order, independent of the search permutation."""

    # name, element, formal charge, H count, aromatic, named tetrahedral parity
    atoms: tuple[tuple[str, str, int, int, bool, int], ...]
    # ordered names, bond kind (4 = aromatic), normalized side relation
    bonds: tuple[tuple[str, str, int, int], ...]

    @classmethod
    def from_mapping(cls, mol: "Mol", names: tuple[str, ...]) -> "NamedProjection":
        assert Chem is not None
        atoms = []
        for a in mol.GetAtoms():
            parity = 0
            if a.GetChiralTag():
                neighbors = [names[x.GetIdx()] for x in a.GetNeighbors()]
                if len(neighbors) == 3:
                    neighbors.append("")
                odd = sum(l > r for l, r in combinations(neighbors, 2)) % 2
                parity = 1 + (
                    odd ^ (a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                )
            atoms.append(
                (
                    names[a.GetIdx()],
                    a.GetSymbol().upper(),
                    a.GetFormalCharge(),
                    a.GetTotalNumHs(),
                    a.GetIsAromatic(),
                    parity,
                )
            )
        bonds = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            first, second = sorted((names[i], names[j]))
            side = 0
            if b.GetStereo():
                if b.GetStereo() not in (
                    Chem.BondStereo.STEREOE,
                    Chem.BondStereo.STEREOZ,
                ):
                    raise CorrespondenceRefusal(
                        CorrespondenceFailure.UNSUPPORTED,
                        "unsupported double-bond stereo",
                    )
                trans = int(b.GetStereo() == Chem.BondStereo.STEREOE)
                for here, there, ref in zip(
                    (i, j), (j, i), b.GetStereoAtoms(), strict=True
                ):
                    least = min(
                        names[z.GetIdx()]
                        for z in mol.GetAtomWithIdx(here).GetNeighbors()
                        if z.GetIdx() != there
                    )
                    trans ^= int(least != names[ref])
                side = 1 + trans
            bonds.append(
                (
                    first,
                    second,
                    4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()),
                    side,
                )
            )
        return cls(tuple(sorted(atoms)), tuple(sorted(bonds)))

    def oxygen_class(self) -> "NamedProjection":
        """Normalize only H-free terminal single O(-)/double O pairs on one center."""
        neighbors: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for first, second, order, side in self.bonds:
            neighbors[first].append((second, order, side))
            neighbors[second].append((first, order, side))
        stereo = {n for n, _, _, _, _, p in self.atoms if p}
        stereo.update(n for a, b, _, s in self.bonds if s for n in (a, b))
        stars: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for name, element, charge, h, aromatic, parity in self.atoms:
            if element != "O" or h or aromatic or parity or len(neighbors[name]) != 1:
                continue
            center, order, side = neighbors[name][0]
            if (
                center not in stereo
                and not side
                and (charge, order) in ((-1, 1), (0, 2))
            ):
                stars[center].append((name, charge, order))
        charges: dict[str, int] = {}
        orders: dict[tuple[str, ...], int] = {}
        for center, group in stars.items():
            for (name, _, _), (charge, order) in zip(
                sorted(group), sorted((q, k) for _, q, k in group), strict=True
            ):
                charges[name] = charge
                orders[tuple(sorted((name, center)))] = order
        return type(self)(
            tuple(
                (n, e, charges.get(n, q), h, a, p) for n, e, q, h, a, p in self.atoms
            ),
            tuple((a, b, orders.get((a, b), k), s) for a, b, k, s in self.bonds),
        )

    def publication_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(a[2] for a in self.atoms), tuple(b[2] for b in self.bonds)

    def difference(self, other: "NamedProjection") -> str:
        for left, right in zip(self.atoms, other.atoms, strict=True):
            if left != right:
                return f"{left[0]} (charge, H count or stereo)"
        for left_b, right_b in zip(self.bonds, other.bonds, strict=True):
            if left_b != right_b:
                return f"{left_b[0]}-{left_b[1]} / {right_b[0]}-{right_b[1]} (bond or stereo)"
        return "named chemistry"

    def named_graph(
        self,
        component: str,
        molecule: "Mol",
        names: tuple[str, ...],
        reserved: set[str],
    ) -> NamedComponentGraph:
        assert Chem is not None and rdCIPLabeler is not None
        checked = Chem.Mol(molecule)
        for item in (*checked.GetAtoms(), *checked.GetBonds()):
            item.ClearProp("_CIPCode")
        rdCIPLabeler.AssignCIPLabels(checked, maxRecursiveIterations=SMILES_CIP_LIMIT)
        labeled_atoms = {
            names[a.GetIdx()]: a.GetProp("_CIPCode")
            for a in checked.GetAtoms()
            if a.HasProp("_CIPCode")
        }
        labeled_bonds = {
            tuple(
                sorted((names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]))
            ): b.GetProp("_CIPCode")
            for b in checked.GetBonds()
            if b.HasProp("_CIPCode")
        }
        atoms = []
        bonds = []
        free = (f"H{i:03d}" for i in range(1, 1000) if f"H{i:03d}" not in reserved)
        for n, e, q, h, a, p in self.atoms:
            if p and n not in labeled_atoms:
                raise CorrespondenceRefusal(
                    CorrespondenceFailure.UNSUPPORTED, "unrepresentable atom stereo"
                )
            atoms.append(ComponentAtom(n, e, q, a, labeled_atoms.get(n) if p else None))
            for _ in range(h):
                hn = next(free, None)
                if hn is None:
                    raise CorrespondenceRefusal(
                        CorrespondenceFailure.SEARCH_LIMIT,
                        "generated H name space exhausted",
                    )
                atoms.append(ComponentAtom(hn, "H", 0, False, None))
                bonds.append(ComponentBond(n, hn, "SING", False, None))
        for left, right, k, s in self.bonds:
            if s and (left, right) not in labeled_bonds:
                raise CorrespondenceRefusal(
                    CorrespondenceFailure.UNSUPPORTED, "unrepresentable bond stereo"
                )
            bonds.append(
                ComponentBond(
                    left,
                    right,
                    ("SING", "DOUB", "TRIP", "AROM")[k - 1],
                    k == 4,
                    labeled_bonds.get((left, right)) if s else None,
                )
            )
        return NamedComponentGraph(
            component,
            tuple(sorted(atoms, key=lambda a: (a.element == "H", a.name))),
            tuple(sorted(bonds, key=lambda b: (b.first, b.second))),
        )
