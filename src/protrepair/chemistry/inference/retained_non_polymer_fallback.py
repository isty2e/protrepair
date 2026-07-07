"""RDKit-backed inference for retained non-polymer fallback chemistry."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

try:
    from rdkit import Chem, rdBase
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None
    rdBase = None

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Atom, Bond, Mol


_CARBONYL_C_O_DOUBLE_BOND_MIN_ANGSTROM = 1.16
_CARBONYL_C_O_DOUBLE_BOND_MAX_ANGSTROM = 1.30
_AROMATIC_CARBON_RING_SIZE = 6
_AROMATIC_LIKE_RING_SIZES = frozenset({5, 6})
_AROMATIC_CARBON_RING_BOND_MIN_ANGSTROM = 1.33
_AROMATIC_CARBON_RING_BOND_MAX_ANGSTROM = 1.45
_AMBIGUOUS_AROMATIC_CARBON_RING_BOND_MAX_ANGSTROM = 1.48
_AROMATIC_LIKE_ORGANIC_RING_ELEMENTS = frozenset({"C", "N", "O", "S"})
_HYDROGEN_NAME_PROJECTION_CANDIDATE_LIMIT = 4096
_NITRO_N_O_UNSUPPORTED_MAX_ANGSTROM = 1.36
_SULFUR_OXYGEN_UNSUPPORTED_MAX_ANGSTROM = 1.62
_PHOSPHORUS_OXYGEN_UNSUPPORTED_MAX_ANGSTROM = 1.72
_IMINE_C_N_DOUBLE_BOND_MIN_ANGSTROM = 1.18
_IMINE_C_N_DOUBLE_BOND_MAX_ANGSTROM = 1.36


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerRdkitFallbackInferenceResult:
    """Canonical RDKit fallback inference over one retained non-polymer pose."""

    hydrogenated_molecule: "Mol"
    rdkit_backend_version: str
    hydrogen_atom_names: tuple[str, ...]
    hydrogen_name_projection: tuple[tuple[str, str], ...]
    heavy_bond_definitions: tuple[BondDefinition, ...]
    hydrogen_bond_definitions: tuple[BondDefinition, ...]
    hydrogen_name_projection_candidate_count: int
    hydrogen_name_projection_candidate_limit: int

    def hydrogen_name_projection_map(self) -> Mapping[str, str]:
        """Return generated-to-final hydrogen names as a read-only mapping."""

        return MappingProxyType(dict(self.hydrogen_name_projection))


@dataclass(frozen=True, slots=True)
class _HydrogenNameProjectionResult:
    """Generated-to-final H-name projection plus bounded-resource metadata."""

    projection: tuple[tuple[str, str], ...]
    candidate_count: int
    candidate_limit: int

    def projection_map(self) -> Mapping[str, str]:
        """Return the generated-to-final projection as a read-only mapping."""

        return MappingProxyType(dict(self.projection))


def retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> tuple[str, ...]:
    """Return preferred expected hydrogen names from RDKit heuristic chemistry."""

    chem = Chem
    if chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback hydrogen expectation requires the "
            "optional rdkit dependency"
        )

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
        reject_ambiguous_valence=True,
    )
    hydrogenated_molecule = _add_hydrogens_with_stereo_guard(heavy_atom_molecule)
    return preferred_retained_non_polymer_expected_hydrogen_atom_names(
        inferred_hydrogen_count=len(
            _generated_hydrogen_atom_names(hydrogenated_molecule)
        ),
        present_hydrogen_atom_names=tuple(
            atom_site.name
            for atom_site in residue_site.atom_sites
            if atom_site.element == "H"
        ),
    )


def infer_retained_non_polymer_rdkit_fallback(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
    hydrogen_atom_names: tuple[str, ...] | None = None,
    present_hydrogen_atom_names: tuple[str, ...] | None = None,
    hydrogen_position_by_name: Mapping[str, Vec3] | None = None,
) -> RetainedNonPolymerRdkitFallbackInferenceResult:
    """Return the canonical RDKit fallback inference result for one pose."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback hydrogenation requires the "
            "optional rdkit dependency"
        )
    if rdBase is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback hydrogenation requires RDKit "
            "backend metadata"
        )

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
        reject_ambiguous_valence=True,
    )
    hydrogenated_molecule = _add_hydrogens_with_stereo_guard(heavy_atom_molecule)
    generated_hydrogen_atom_names = _generated_hydrogen_atom_names(
        hydrogenated_molecule
    )
    resolved_hydrogen_atom_names = hydrogen_atom_names
    if resolved_hydrogen_atom_names is None:
        if present_hydrogen_atom_names is None:
            present_hydrogen_atom_names = tuple(
                atom_site.name
                for atom_site in residue_site.atom_sites
                if atom_site.element == "H"
            )
        resolved_hydrogen_atom_names = (
            preferred_retained_non_polymer_expected_hydrogen_atom_names(
                inferred_hydrogen_count=len(generated_hydrogen_atom_names),
                present_hydrogen_atom_names=present_hydrogen_atom_names,
            )
        )
    hydrogen_name_projection = _retained_non_polymer_hydrogen_name_projection_result(
        hydrogenated_molecule,
        generated_hydrogen_atom_names=generated_hydrogen_atom_names,
        hydrogen_atom_names=resolved_hydrogen_atom_names,
        hydrogen_position_by_name=hydrogen_position_by_name,
    )
    return RetainedNonPolymerRdkitFallbackInferenceResult(
        hydrogenated_molecule=hydrogenated_molecule,
        rdkit_backend_version=str(rdBase.rdkitVersion),
        hydrogen_atom_names=resolved_hydrogen_atom_names,
        hydrogen_name_projection=hydrogen_name_projection.projection,
        heavy_bond_definitions=_heavy_bond_definitions_from_molecule(
            heavy_atom_molecule
        ),
        hydrogen_bond_definitions=_projected_hydrogen_bond_definitions(
            hydrogenated_molecule,
            hydrogen_name_projection=hydrogen_name_projection.projection_map(),
        ),
        hydrogen_name_projection_candidate_count=(
            hydrogen_name_projection.candidate_count
        ),
        hydrogen_name_projection_candidate_limit=(
            hydrogen_name_projection.candidate_limit
        ),
    )


def retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> "Mol":
    """Return one RDKit hydrogenated molecule inferred from the current pose."""

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback hydrogenation requires the "
            "optional rdkit dependency"
        )

    inference_result = infer_retained_non_polymer_rdkit_fallback(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    return inference_result.hydrogenated_molecule


def retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> tuple[BondDefinition, ...]:
    """Return heavy-atom bond definitions inferred from one retained pose.

    This intentionally uses a heavy-only projection for passive/readiness callers
    that do not need H placement or generated H-name projection.
    """

    if Chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback bond inference requires the "
            "optional rdkit dependency"
        )

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )
    return _heavy_bond_definitions_from_molecule(heavy_atom_molecule)


def _heavy_bond_definitions_from_molecule(
    heavy_atom_molecule: "Mol",
) -> tuple[BondDefinition, ...]:
    """Return heavy-atom bond definitions from one RDKit heavy pose."""

    return tuple(
        BondDefinition(
            atom_name_1=_rdkit_atom_name(
                heavy_atom_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
            ),
            atom_name_2=_rdkit_atom_name(
                heavy_atom_molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
            ),
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in heavy_atom_molecule.GetBonds()
    )


def retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions(
    hydrogenated_molecule: "Mol",
) -> tuple[BondDefinition, ...]:
    """Return generated H-heavy bond definitions from an RDKit fallback molecule."""

    atom_names_by_index = _rdkit_fallback_atom_names_by_index(
        hydrogenated_molecule
    )
    return tuple(
        BondDefinition(
            atom_name_1=atom_names_by_index[begin_atom.GetIdx()],
            atom_name_2=atom_names_by_index[end_atom.GetIdx()],
            order=max(1, round(bond.GetBondTypeAsDouble())),
            aromatic=bond.GetIsAromatic(),
        )
        for bond in hydrogenated_molecule.GetBonds()
        for begin_atom, end_atom in (
            (
                hydrogenated_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()),
                hydrogenated_molecule.GetAtomWithIdx(bond.GetEndAtomIdx()),
            ),
        )
        if (begin_atom.GetAtomicNum() == 1) != (end_atom.GetAtomicNum() == 1)
    )


def _projected_hydrogen_bond_definitions(
    hydrogenated_molecule: "Mol",
    *,
    hydrogen_name_projection: Mapping[str, str],
) -> tuple[BondDefinition, ...]:
    """Return generated H-heavy bonds projected through final H names."""

    generated_bonds = retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions(
        hydrogenated_molecule
    )
    return tuple(
        BondDefinition(
            atom_name_1=hydrogen_name_projection.get(
                bond_definition.atom_name_1,
                bond_definition.atom_name_1,
            ),
            atom_name_2=hydrogen_name_projection.get(
                bond_definition.atom_name_2,
                bond_definition.atom_name_2,
            ),
            order=bond_definition.order,
            aromatic=bond_definition.aromatic,
        )
        for bond_definition in generated_bonds
    )


def retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions_for_names(
    hydrogenated_molecule: "Mol",
    *,
    hydrogen_atom_names: tuple[str, ...],
    hydrogen_position_by_name: Mapping[str, Vec3] | None = None,
) -> tuple[BondDefinition, ...]:
    """Return fallback H-heavy bonds projected onto preferred H atom names."""

    hydrogen_name_projection = (
        _retained_non_polymer_hydrogen_name_projection_result(
            hydrogenated_molecule,
            hydrogen_atom_names=hydrogen_atom_names,
            hydrogen_position_by_name=hydrogen_position_by_name,
        )
    )
    return _projected_hydrogen_bond_definitions(
        hydrogenated_molecule,
        hydrogen_name_projection=hydrogen_name_projection.projection_map(),
    )


def retained_non_polymer_rdkit_fallback_hydrogen_name_projection(
    hydrogenated_molecule: "Mol",
    *,
    hydrogen_atom_names: tuple[str, ...],
    hydrogen_position_by_name: Mapping[str, Vec3] | None = None,
) -> Mapping[str, str]:
    """Return generated fallback H names projected onto canonical H names."""

    projection_result = _retained_non_polymer_hydrogen_name_projection_result(
        hydrogenated_molecule,
        hydrogen_atom_names=hydrogen_atom_names,
        hydrogen_position_by_name=hydrogen_position_by_name,
    )
    return projection_result.projection_map()


def _retained_non_polymer_hydrogen_name_projection_result(
    hydrogenated_molecule: "Mol",
    *,
    hydrogen_atom_names: tuple[str, ...],
    hydrogen_position_by_name: Mapping[str, Vec3] | None = None,
    generated_hydrogen_atom_names: tuple[str, ...] | None = None,
) -> _HydrogenNameProjectionResult:
    """Return generated fallback H names projected onto final H names."""

    if generated_hydrogen_atom_names is None:
        generated_hydrogen_atom_names = _generated_hydrogen_atom_names(
            hydrogenated_molecule
        )
    if len(generated_hydrogen_atom_names) != len(hydrogen_atom_names):
        raise ValueError(
            "retained non-polymer fallback hydrogen count must match the "
            "preferred hydrogen atom name count"
        )

    if (
        hydrogen_position_by_name is not None
        and set(hydrogen_atom_names).issubset(hydrogen_position_by_name)
    ):
        return _hydrogen_name_projection_by_nearest_position(
            hydrogenated_molecule,
            generated_hydrogen_atom_names=generated_hydrogen_atom_names,
            hydrogen_atom_names=hydrogen_atom_names,
            hydrogen_position_by_name=hydrogen_position_by_name,
        )

    return _HydrogenNameProjectionResult(
        projection=tuple(
            zip(generated_hydrogen_atom_names, hydrogen_atom_names, strict=True)
        ),
        candidate_count=len(generated_hydrogen_atom_names),
        candidate_limit=_HYDROGEN_NAME_PROJECTION_CANDIDATE_LIMIT,
    )


def retained_non_polymer_rdkit_fallback_supports_passive_context(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> bool:
    """Return whether fallback chemistry is coherent enough for passive context."""

    return (
        retained_non_polymer_rdkit_fallback_passive_heavy_bond_definitions(
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name=formal_charge_by_atom_name,
        )
        is not None
    )


def retained_non_polymer_rdkit_fallback_passive_heavy_bond_definitions(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
) -> tuple[BondDefinition, ...] | None:
    """Return passive-context heavy bonds when fallback chemistry is coherent."""

    chem = Chem
    if chem is None:
        raise RdkitUnavailableError(
            "retained non-polymer RDKit fallback passive context requires the "
            "optional rdkit dependency"
        )

    heavy_atom_molecule = _retained_non_polymer_rdkit_pose_molecule(
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
        reject_ambiguous_valence=True,
    )
    if heavy_atom_molecule.GetNumAtoms() <= 1:
        return ()

    if heavy_atom_molecule.GetNumBonds() <= 0:
        return None

    if len(chem.GetMolFrags(heavy_atom_molecule)) != 1:
        return None

    return _heavy_bond_definitions_from_molecule(heavy_atom_molecule)


def preferred_retained_non_polymer_expected_hydrogen_atom_names(
    *,
    inferred_hydrogen_count: int,
    present_hydrogen_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return preferred expected hydrogen names for one inferred hydrogen count."""

    if inferred_hydrogen_count <= 0:
        return ()

    if (
        present_hydrogen_atom_names
        and len(present_hydrogen_atom_names) == inferred_hydrogen_count
    ):
        return present_hydrogen_atom_names

    return tuple(
        f"H{index:03d}" for index in range(1, inferred_hydrogen_count + 1)
    )


def _retained_non_polymer_rdkit_pose_molecule(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    formal_charge_by_atom_name: Mapping[str, int | None] | None = None,
    reject_ambiguous_valence: bool = False,
) -> "Mol":
    """Return one sanitized RDKit heavy-atom pose for one residue payload."""

    assert Chem is not None

    heavy_atom_pdb_block = _retained_non_polymer_heavy_atom_pdb_block(
        residue_site,
        residue_geometry,
    )
    pose_molecule = Chem.MolFromPDBBlock(
        heavy_atom_pdb_block,
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if pose_molecule is None:
        raise ValueError(
            "RDKit could not parse retained non-polymer fallback heavy-atom pose "
            f"for {residue_site.residue_id.display_token()}"
        )

    normalized_formal_charges = (
        {}
        if formal_charge_by_atom_name is None
        else {
            atom_name.strip().upper(): formal_charge
            for atom_name, formal_charge in formal_charge_by_atom_name.items()
            if formal_charge is not None
        }
    )
    for atom in pose_molecule.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        if residue_info is None:
            continue

        atom_name = residue_info.GetName().strip().upper()
        formal_charge = normalized_formal_charges.get(atom_name)
        if formal_charge is None:
            continue

        atom.SetFormalCharge(formal_charge)

    _apply_high_confidence_fallback_bond_orders(
        pose_molecule,
        residue_geometry=residue_geometry,
        reject_ambiguous_valence=reject_ambiguous_valence,
    )
    _reject_unsupported_fallback_motifs(
        pose_molecule,
        residue_geometry=residue_geometry,
    )
    Chem.SanitizeMol(pose_molecule)
    return pose_molecule


def _add_hydrogens_with_stereo_guard(heavy_atom_molecule: "Mol") -> "Mol":
    """Return an RDKit-hydrogenated molecule without changing heavy stereo."""

    assert Chem is not None

    stereo_before = _heavy_atom_stereo_signature(heavy_atom_molecule)
    hydrogenated_molecule = Chem.AddHs(heavy_atom_molecule, addCoords=True)
    stereo_after = _heavy_atom_stereo_signature(hydrogenated_molecule)
    if stereo_before and stereo_after != stereo_before:
        raise ValueError(
            "retained non-polymer fallback stereochemistry changed during "
            "hydrogenation"
        )

    return hydrogenated_molecule


def _heavy_atom_stereo_signature(molecule: "Mol") -> tuple[tuple[str, str], ...]:
    """Return assigned heavy-atom chirality labels by atom name."""

    assert Chem is not None

    stereo_molecule = Chem.Mol(molecule)
    Chem.AssignAtomChiralTagsFromStructure(
        stereo_molecule,
        replaceExistingTags=True,
    )
    Chem.AssignStereochemistry(stereo_molecule, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(
        stereo_molecule,
        includeUnassigned=False,
        useLegacyImplementation=False,
    )
    return tuple(
        sorted(
            (
                _rdkit_atom_name(atom),
                str(stereo_label),
            )
            for atom_index, stereo_label in centers
            for atom in (stereo_molecule.GetAtomWithIdx(int(atom_index)),)
            if atom.GetAtomicNum() != 1
        )
    )


def _reject_unsupported_fallback_motifs(
    pose_molecule: "Mol",
    *,
    residue_geometry: ResidueGeometry,
) -> None:
    """Reject hetero multiple-bond motifs that need explicit chemistry."""

    for atom in pose_molecule.GetAtoms():
        element = atom.GetSymbol().upper()
        oxygen_bond_lengths = _oxygen_neighbor_bond_lengths(
            atom,
            pose_molecule=pose_molecule,
            residue_geometry=residue_geometry,
        )
        if element == "N" and sum(
            bond_length <= _NITRO_N_O_UNSUPPORTED_MAX_ANGSTROM
            for bond_length in oxygen_bond_lengths
        ) >= 2:
            raise ValueError(
                "unsupported retained non-polymer fallback nitro or nitrate motif"
            )

        if element == "S" and atom.GetDegree() >= 2 and any(
            bond_length <= _SULFUR_OXYGEN_UNSUPPORTED_MAX_ANGSTROM
            for bond_length in oxygen_bond_lengths
        ):
            raise ValueError(
                "unsupported retained non-polymer fallback sulfoxide, sulfone, "
                "or sulfate motif"
            )

        if element == "P" and sum(
            bond_length <= _PHOSPHORUS_OXYGEN_UNSUPPORTED_MAX_ANGSTROM
            for bond_length in oxygen_bond_lengths
        ) >= 2:
            raise ValueError(
                "unsupported retained non-polymer fallback phosphate motif"
            )

    for bond in pose_molecule.GetBonds():
        begin_atom = pose_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end_atom = pose_molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
        endpoint_elements = {
            begin_atom.GetSymbol().upper(),
            end_atom.GetSymbol().upper(),
        }
        if endpoint_elements != {"C", "N"}:
            continue

        bond_length = _rdkit_bond_length_angstrom(
            bond,
            pose_molecule=pose_molecule,
            residue_geometry=residue_geometry,
        )
        if (
            _IMINE_C_N_DOUBLE_BOND_MIN_ANGSTROM
            <= bond_length
            <= _IMINE_C_N_DOUBLE_BOND_MAX_ANGSTROM
        ):
            raise ValueError(
                "unsupported retained non-polymer fallback imine-like motif"
            )


def _oxygen_neighbor_bond_lengths(
    atom: "Atom",
    *,
    pose_molecule: "Mol",
    residue_geometry: ResidueGeometry,
) -> tuple[float, ...]:
    """Return bond lengths from one atom to oxygen neighbors."""

    return tuple(
        _rdkit_bond_length_angstrom(
            bond,
            pose_molecule=pose_molecule,
            residue_geometry=residue_geometry,
        )
        for bond in atom.GetBonds()
        for neighbor in (bond.GetOtherAtom(atom),)
        if neighbor.GetSymbol().upper() == "O"
    )


def _apply_high_confidence_fallback_bond_orders(
    pose_molecule: "Mol",
    *,
    residue_geometry: ResidueGeometry,
    reject_ambiguous_valence: bool,
) -> None:
    """Apply conservative geometry-backed bond orders before hydrogenation."""

    _apply_high_confidence_carbonyl_bond_orders(
        pose_molecule,
        residue_geometry=residue_geometry,
        reject_ambiguous_valence=reject_ambiguous_valence,
    )
    _apply_high_confidence_aromatic_ring_bond_orders(
        pose_molecule,
        residue_geometry=residue_geometry,
        reject_ambiguous_valence=reject_ambiguous_valence,
    )


def _apply_high_confidence_carbonyl_bond_orders(
    pose_molecule: "Mol",
    *,
    residue_geometry: ResidueGeometry,
    reject_ambiguous_valence: bool,
) -> None:
    """Promote unambiguous terminal C-O carbonyl bonds from pose geometry."""

    assert Chem is not None

    candidate_bond_indices_by_carbon_index: dict[int, list[int]] = {}
    ambiguous_short_c_o_bond_indices: list[int] = []
    for bond in pose_molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        carbon_atom, oxygen_atom = _carbon_oxygen_bond_atoms(
            pose_molecule,
            bond,
        )
        if carbon_atom is None or oxygen_atom is None:
            continue

        bond_length = _rdkit_bond_length_angstrom(
            bond,
            pose_molecule=pose_molecule,
            residue_geometry=residue_geometry,
        )
        if not (
            _CARBONYL_C_O_DOUBLE_BOND_MIN_ANGSTROM
            <= bond_length
            <= _CARBONYL_C_O_DOUBLE_BOND_MAX_ANGSTROM
        ):
            continue

        if (
            carbon_atom.GetFormalCharge() != 0
            or oxygen_atom.GetFormalCharge() != 0
            or carbon_atom.GetDegree() < 2
            or oxygen_atom.GetDegree() != 1
        ):
            ambiguous_short_c_o_bond_indices.append(bond.GetIdx())
            continue

        candidate_bond_indices_by_carbon_index.setdefault(
            carbon_atom.GetIdx(),
            [],
        ).append(bond.GetIdx())

    if ambiguous_short_c_o_bond_indices and reject_ambiguous_valence:
        raise ValueError("ambiguous retained non-polymer fallback carbonyl geometry")

    for candidate_bond_indices in candidate_bond_indices_by_carbon_index.values():
        if len(candidate_bond_indices) > 1:
            if reject_ambiguous_valence:
                raise ValueError(
                    "ambiguous retained non-polymer fallback carbonyl geometry"
                )
            continue

        pose_molecule.GetBondWithIdx(candidate_bond_indices[0]).SetBondType(
            Chem.BondType.DOUBLE
        )


def _apply_high_confidence_aromatic_ring_bond_orders(
    pose_molecule: "Mol",
    *,
    residue_geometry: ResidueGeometry,
    reject_ambiguous_valence: bool,
) -> None:
    """Mark unambiguous benzene-like carbon rings as aromatic."""

    assert Chem is not None

    for ring_atom_indices in Chem.GetSymmSSSR(pose_molecule):
        ring_atom_index_tuple = tuple(
            int(atom_index) for atom_index in ring_atom_indices
        )
        if len(ring_atom_index_tuple) not in _AROMATIC_LIKE_RING_SIZES:
            continue

        ring_atoms = tuple(
            pose_molecule.GetAtomWithIdx(atom_index)
            for atom_index in ring_atom_index_tuple
        )
        ring_bonds = _ring_bonds(pose_molecule, ring_atom_index_tuple)
        if len(ring_bonds) != len(ring_atom_index_tuple):
            continue

        ring_atom_symbols = tuple(atom.GetSymbol().upper() for atom in ring_atoms)
        ring_bond_lengths = tuple(
            _rdkit_bond_length_angstrom(
                bond,
                pose_molecule=pose_molecule,
                residue_geometry=residue_geometry,
            )
            for bond in ring_bonds
        )
        if not all(
            _AROMATIC_CARBON_RING_BOND_MIN_ANGSTROM
            <= bond_length
            <= _AMBIGUOUS_AROMATIC_CARBON_RING_BOND_MAX_ANGSTROM
            for bond_length in ring_bond_lengths
        ):
            continue

        if len(ring_atom_index_tuple) == _AROMATIC_CARBON_RING_SIZE and all(
            _AROMATIC_CARBON_RING_BOND_MIN_ANGSTROM
            <= bond_length
            <= _AROMATIC_CARBON_RING_BOND_MAX_ANGSTROM
            for bond_length in ring_bond_lengths
        ) and all(
            symbol == "C" and atom.GetFormalCharge() == 0
            for symbol, atom in zip(ring_atom_symbols, ring_atoms, strict=True)
        ):
            for atom in ring_atoms:
                atom.SetIsAromatic(True)
            for bond in ring_bonds:
                bond.SetBondType(Chem.BondType.AROMATIC)
                bond.SetIsAromatic(True)
            continue

        if not all(
            symbol in _AROMATIC_LIKE_ORGANIC_RING_ELEMENTS
            for symbol in ring_atom_symbols
        ):
            continue

        if reject_ambiguous_valence:
            raise ValueError(
                "ambiguous retained non-polymer fallback aromatic ring geometry"
            )


def _carbon_oxygen_bond_atoms(
    pose_molecule: "Mol",
    bond: "Bond",
) -> tuple["Atom | None", "Atom | None"]:
    """Return the carbon and oxygen endpoints for one C-O bond."""

    begin_atom = pose_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
    end_atom = pose_molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
    begin_element = begin_atom.GetSymbol().upper()
    end_element = end_atom.GetSymbol().upper()
    if begin_element == "C" and end_element == "O":
        return begin_atom, end_atom
    if begin_element == "O" and end_element == "C":
        return end_atom, begin_atom
    return None, None


def _ring_bonds(
    pose_molecule: "Mol",
    ring_atom_indices: tuple[int, ...],
) -> tuple["Bond", ...]:
    """Return the bonds connecting consecutive atoms in one RDKit ring."""

    ring_bonds: list[Bond] = []
    for atom_index, next_atom_index in zip(
        ring_atom_indices,
        ring_atom_indices[1:] + ring_atom_indices[:1],
        strict=True,
    ):
        bond = pose_molecule.GetBondBetweenAtoms(atom_index, next_atom_index)
        if bond is None:
            return ()
        ring_bonds.append(bond)
    return tuple(ring_bonds)


def _rdkit_bond_length_angstrom(
    bond: "Bond",
    *,
    pose_molecule: "Mol",
    residue_geometry: ResidueGeometry,
) -> float:
    """Return a residue-geometry bond length for one RDKit pose bond."""

    begin_atom = pose_molecule.GetAtomWithIdx(bond.GetBeginAtomIdx())
    end_atom = pose_molecule.GetAtomWithIdx(bond.GetEndAtomIdx())
    return residue_geometry.atom_geometry(_rdkit_atom_name(begin_atom)).distance_to(
        residue_geometry.atom_geometry(_rdkit_atom_name(end_atom))
    )


def _retained_non_polymer_heavy_atom_pdb_block(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
) -> str:
    """Return one heavy-atom-only PDB block for RDKit pose inference."""

    record_name = "HETATM" if residue_site.is_hetero else "ATOM  "
    residue_name = residue_site.component_id[:3].rjust(3)
    chain_id = residue_site.residue_id.chain_id[:1]
    insertion_code = (residue_site.residue_id.insertion_code or " ")[:1]
    pdb_lines: list[str] = []
    atom_serial = 1
    for atom_site in residue_site.atom_sites:
        if atom_site.element == "H":
            continue

        atom_geometry = residue_geometry.atom_geometry(atom_site.name)
        position = atom_geometry.position
        altloc = (atom_geometry.altloc or " ")[:1]
        occupancy = atom_geometry.occupancy
        b_factor = 0.0 if atom_geometry.b_factor is None else atom_geometry.b_factor
        pdb_lines.append(
            f"{record_name:<6}{atom_serial:5d} "
            f"{atom_site.name:>4}{altloc}{residue_name} {chain_id}"
            f"{residue_site.residue_id.seq_num:4d}{insertion_code}   "
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          "
            f"{_pdb_element_field(atom_site.element)}"
        )
        atom_serial += 1

    pdb_lines.append("END")
    return "\n".join(pdb_lines) + "\n"


def _pdb_element_field(element: str) -> str:
    """Return one PDB-aligned element field."""

    normalized_element = element.strip()
    if len(normalized_element) >= 2:
        normalized_element = (
            normalized_element[0].upper() + normalized_element[1].lower()
        )
    else:
        normalized_element = normalized_element.upper()

    return normalized_element.rjust(2)


def _rdkit_atom_name(atom: "Atom") -> str:
    """Return the PDB atom name attached to one RDKit atom."""

    residue_info = atom.GetPDBResidueInfo()
    if residue_info is None:
        raise ValueError("RDKit fallback atom is missing PDB residue metadata")

    return residue_info.GetName().strip()


def _rdkit_fallback_atom_names_by_index(
    hydrogenated_molecule: "Mol",
) -> dict[int, str]:
    """Return fallback atom names matching RDKit-generated H append order."""

    atom_names_by_index: dict[int, str] = {}
    hydrogen_index = 1
    for atom in hydrogenated_molecule.GetAtoms():
        atom_index = atom.GetIdx()
        if atom.GetAtomicNum() == 1:
            atom_names_by_index[atom_index] = f"H{hydrogen_index:03d}"
            hydrogen_index += 1
            continue

        atom_names_by_index[atom_index] = _rdkit_atom_name(atom)

    return atom_names_by_index


def _generated_hydrogen_atom_names(hydrogenated_molecule: "Mol") -> tuple[str, ...]:
    """Return generated fallback hydrogen names in RDKit atom order."""

    atom_names_by_index = _rdkit_fallback_atom_names_by_index(hydrogenated_molecule)
    return tuple(
        atom_names_by_index[atom.GetIdx()]
        for atom in hydrogenated_molecule.GetAtoms()
        if atom.GetAtomicNum() == 1
    )


def _hydrogen_name_projection_by_nearest_position(
    hydrogenated_molecule: "Mol",
    *,
    generated_hydrogen_atom_names: tuple[str, ...],
    hydrogen_atom_names: tuple[str, ...],
    hydrogen_position_by_name: Mapping[str, Vec3],
) -> _HydrogenNameProjectionResult:
    """Project generated H names to existing H names by nearest coordinates."""

    candidate_count = len(generated_hydrogen_atom_names) * len(hydrogen_atom_names)
    if candidate_count > _HYDROGEN_NAME_PROJECTION_CANDIDATE_LIMIT:
        raise ValueError(
            "retained non-polymer fallback hydrogen name projection exceeds "
            "the per-residue resource limit"
        )

    generated_position_by_name = _generated_hydrogen_position_by_name(
        hydrogenated_molecule
    )
    candidate_pairs = sorted(
        (
            generated_position_by_name[generated_name].distance_to(
                hydrogen_position_by_name[hydrogen_name]
            ),
            generated_name,
            hydrogen_name,
        )
        for generated_name in generated_hydrogen_atom_names
        for hydrogen_name in hydrogen_atom_names
    )
    projection: dict[str, str] = {}
    used_hydrogen_names: set[str] = set()
    for _distance, generated_name, hydrogen_name in candidate_pairs:
        if generated_name in projection or hydrogen_name in used_hydrogen_names:
            continue

        projection[generated_name] = hydrogen_name
        used_hydrogen_names.add(hydrogen_name)
        if len(projection) == len(generated_hydrogen_atom_names):
            break

    if len(projection) != len(generated_hydrogen_atom_names):
        raise ValueError(
            "retained non-polymer fallback could not project generated hydrogen "
            "names onto preferred hydrogen names"
        )

    return _HydrogenNameProjectionResult(
        projection=tuple(projection.items()),
        candidate_count=candidate_count,
        candidate_limit=_HYDROGEN_NAME_PROJECTION_CANDIDATE_LIMIT,
    )


def _generated_hydrogen_position_by_name(
    hydrogenated_molecule: "Mol",
) -> Mapping[str, Vec3]:
    """Return generated fallback hydrogen coordinates by generated H name."""

    atom_names_by_index = _rdkit_fallback_atom_names_by_index(hydrogenated_molecule)
    conformer = hydrogenated_molecule.GetConformer()
    return {
        atom_names_by_index[atom.GetIdx()]: Vec3(
            float(coordinates.x),
            float(coordinates.y),
            float(coordinates.z),
        )
        for atom in hydrogenated_molecule.GetAtoms()
        if atom.GetAtomicNum() == 1
        for coordinates in (conformer.GetAtomPosition(atom.GetIdx()),)
    }


def _is_generated_hydrogen_name(atom_name: str) -> bool:
    """Return whether a fallback atom name is one of our generated H names."""

    return len(atom_name) == 4 and atom_name.startswith("H") and atom_name[1:].isdigit()
