"""FASPR specialization of the generic side-chain packing backend seam."""

import subprocess
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from typing_extensions import Self

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import (
    PackingError,
    ProtrepairError,
    ResidueNotFoundError,
    StructureNormalizationError,
)
from protrepair.io.gemmi_ingress import MAX_STRUCTURE_INPUT_BYTES
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import AtomGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
    is_source_provenance,
)
from protrepair.transformer.packing.alphabet import PackingAlphabet
from protrepair.transformer.packing.domain import (
    PackingCapabilities,
    PackingPlan,
    PackingResult,
)
from protrepair.transformer.packing.faspr.paths import (
    faspr_executable_path,
)
from protrepair.transformer.packing.faspr.projection import FasprPDBProjection
from protrepair.transformer.packing.faspr.selection import (
    FIXED_BACKBONE_ATOM_NAMES,
    FasprPackingSelection,
)

BACKBONE_ATOM_NAMES: frozenset[str] = frozenset({"N", "CA", "C", "O"})
FASPR_HYDROGEN_PRESERVATION_HEAVY_ATOM_TOLERANCE_ANGSTROM = 0.05
FASPR_CAPABILITIES = PackingCapabilities(
    supports_full_structure_packing=True,
    supports_local_packing=True,
    supports_partial_sequence=True,
    supports_refinement=False,
    supports_noncanonical_components=False,
    deterministic_given_same_inputs=True,
)
DEFAULT_FASPR_EXECUTION_TIMEOUT_SECONDS = 1800.0
MAX_FASPR_CAPTURED_OUTPUT_BYTES = 64 * 1024
FASPR_ALPHABET = PackingAlphabet(
    {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
)


class PackingBackendError(ProtrepairError):
    """Raised when a side-chain packing backend cannot satisfy one request."""


class PackingBackendExecutionError(PackingBackendError):
    """Raised when a side-chain packing subprocess fails."""


@dataclass(frozen=True, slots=True)
class FasprPackingBackend:
    """Subprocess-backed adapter for the packaged FASPR executable."""

    executable_path: Path | None = None
    timeout_seconds: float = DEFAULT_FASPR_EXECUTION_TIMEOUT_SECONDS
    _runtime_assets: "FasprRuntimeAssets | None" = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Normalize backend-local subprocess execution settings."""

        object.__setattr__(
            self,
            "timeout_seconds",
            normalize_faspr_timeout_seconds(self.timeout_seconds),
        )

    def capabilities(self) -> PackingCapabilities:
        """Return the declared capability surface of the FASPR backend."""

        return FASPR_CAPABILITIES

    def pack(self, plan: PackingPlan) -> PackingResult:
        """Execute one canonical packing plan through FASPR."""

        plan.assert_supported_by(self.capabilities())

        # A protected no-op must not hide a malformed sequence or mutation.
        requested_sequence = build_faspr_sequence_override(plan) or "".join(
            FASPR_ALPHABET.sequence_for_residues(plan.polymer_residue_sites())
        )
        components = FASPR_ALPHABET.components_for_sequence(requested_sequence)

        mutations = frozenset(
            residue.residue_id
            for residue, component in zip(
                plan.polymer_residue_sites(), components, strict=True
            )
            if residue.component_id != component
        )

        selection = FasprPackingSelection.from_plan(plan)
        linked_mutations = mutations & frozenset(selection.linked_residue_ids)
        if linked_mutations:
            raise PackingError(
                "FASPR mutation would invalidate linked chemistry at "
                + ", ".join(rid.display_token() for rid in sorted(linked_mutations))
            )

        frozen_mutations = mutations & frozenset(plan.fixed_residue_ids())
        if frozen_mutations:
            raise PackingError(
                "packing target_sequence would mutate frozen residues: "
                + ", ".join(rid.display_token() for rid in sorted(frozen_mutations))
            )

        effective_plan = selection.restricted_plan()
        issues = _linked_sidechain_packing_issues(selection)
        if effective_plan is None:
            return PackingResult(
                packed_structure=plan.structure,
                changed_residue_ids=(),
                issues=issues,
                backend_name="faspr",
            )

        execution_input = FasprExecutionInput.from_plan(effective_plan)
        try:
            runtime_assets = self.runtime_assets()
        except (OSError, PackingBackendError) as error:
            raise PackingBackendExecutionError(
                f"FASPR runtime assets are unavailable: {error}"
            ) from error

        packed_structure = run_faspr(
            execution_input,
            executable_path=runtime_assets.executable_path,
            timeout_seconds=self.timeout_seconds,
        )
        changed_residue_ids = plan.changed_residue_ids_after(packed_structure)
        issues = (*issues, *infer_packing_issues(plan, packed_structure))
        return PackingResult(
            packed_structure=packed_structure,
            changed_residue_ids=changed_residue_ids,
            issues=issues,
            backend_name="faspr",
            backend_version=None,
        )

    def runtime_assets(self) -> "FasprRuntimeAssets":
        """Return cached FASPR runtime assets after validating availability."""

        runtime_assets = self._runtime_assets
        if runtime_assets is None:
            runtime_assets = FasprRuntimeAssets.from_executable_path(
                self.executable_path
            )
            object.__setattr__(self, "_runtime_assets", runtime_assets)

        runtime_assets.assert_available()
        return runtime_assets


@dataclass(frozen=True, slots=True)
class FasprRuntimeAssets:
    """Resolved FASPR executable and rotamer-library paths."""

    executable_path: Path
    rotamer_library_path: Path

    @classmethod
    def from_executable_path(cls, executable_path: Path | None) -> "FasprRuntimeAssets":
        """Resolve and validate runtime assets for one backend instance."""

        resolved_executable_path = resolve_faspr_executable_path(executable_path)
        return cls(
            executable_path=resolved_executable_path,
            rotamer_library_path=validate_rotamer_library_near(
                resolved_executable_path
            ),
        )

    def assert_available(self) -> None:
        """Raise when cached runtime assets are no longer available."""

        if not self.executable_path.exists():
            raise FileNotFoundError(
                f"FASPR executable does not exist: {self.executable_path}"
            )

        if not self.executable_path.is_file():
            raise PackingBackendError(
                f"FASPR executable path is not a file: {self.executable_path}"
            )

        if not self.rotamer_library_path.exists():
            raise PackingBackendError(
                "FASPR requires dun2010bbdep.bin to exist beside the executable"
            )


@dataclass(frozen=True, slots=True)
class FasprExecutionInput:
    """FASPR transport and the canonical plan governing its output.

    Parameters
    ----------
    structure : ProteinStructure
        Polymer-H-stripped context; retained ligands are excluded at native
        serialization.
    plan : PackingPlan
        Current chemical state, requested sequence and movement authority.
    sequence_override : str or None, default=None
        Native sequence encoding, including fixed-residue labels.
    """

    structure: ProteinStructure
    plan: PackingPlan
    sequence_override: str | None = None

    @classmethod
    def from_plan(cls, plan: PackingPlan) -> Self:
        """Build one FASPR execution input from a generic packing plan."""

        prepared_structure = prepare_structure_for_faspr(plan)
        sequence_override = build_faspr_sequence_override(plan)
        return cls(
            structure=prepared_structure,
            plan=plan,
            sequence_override=sequence_override,
        )


def resolve_faspr_executable_path(executable_path: Path | None) -> Path:
    """Return a usable FASPR executable path."""

    resolved_path = (
        faspr_executable_path() if executable_path is None else executable_path
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"FASPR executable does not exist: {resolved_path}")

    if not resolved_path.is_file():
        raise PackingBackendError(
            f"FASPR executable path is not a file: {resolved_path}"
        )

    return resolved_path.absolute()


def normalize_faspr_timeout_seconds(timeout_seconds: float) -> float:
    """Return a finite positive FASPR subprocess timeout."""

    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        int | float,
    ):
        raise TypeError("timeout_seconds must be a real number")

    timeout = float(timeout_seconds)
    if not isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")

    return timeout


def validate_rotamer_library_near(executable_path: Path) -> Path:
    """Return the expected rotamer-library path beside one FASPR executable."""

    sibling_library_path = executable_path.parent / "dun2010bbdep.bin"
    if not sibling_library_path.exists():
        raise PackingBackendError(
            "FASPR requires dun2010bbdep.bin to exist beside the executable"
        )

    return sibling_library_path


def prepare_structure_for_faspr(plan: PackingPlan) -> ProteinStructure:
    """Return one canonical structure compatible with FASPR expectations."""

    fixed_residue_ids = set(plan.fixed_residue_ids())
    component_library = build_standard_component_library()
    prepared_chain_sites: list[ChainSite] = []
    removed_atom_refs: list[AtomRef] = []
    incomplete_fixed_residues: list[str] = []
    for chain_site in plan.polymer_chain_sites():
        prepared_residue_sites: list[ResidueSite] = []
        for residue_site in chain_site.residues:
            validate_faspr_residue_site(residue_site)
            if residue_site.residue_id in fixed_residue_ids:
                missing_heavy_atoms = tuple(
                    name
                    for name in component_library.require(
                        residue_site.component_id
                    ).expected_polymer_heavy_atom_names()
                    if not residue_site.has_atom_site(name)
                )
                if missing_heavy_atoms:
                    # FASPR unfreezes incomplete fixed residues before packing.
                    incomplete_fixed_residues.append(
                        f"fixed residue {residue_site.residue_id.display_token()} "
                        f"(missing heavy atoms: {', '.join(missing_heavy_atoms)})"
                    )

            prepared_residue_sites.append(
                strip_hydrogens_from_residue_site(residue_site)
            )
            removed_atom_refs.extend(_hydrogen_atom_refs_for_residue_site(residue_site))

        prepared_chain_sites.append(chain_site.with_residues(prepared_residue_sites))

    if incomplete_fixed_residues:
        raise PackingBackendError(
            "FASPR cannot preserve incomplete "
            f"{'; '.join(incomplete_fixed_residues)}. Include these residues in "
            "mutable_residue_ids (and remove them from frozen_residue_ids), "
            "or complete heavy atoms first."
        )

    removed_atom_ref_set = frozenset(removed_atom_refs)
    removed_atom_indices = frozenset(
        plan.structure.constitution.atom_index(atom_ref)
        for atom_ref in removed_atom_ref_set
    )
    removed_atom_index_values = {
        atom_index.value for atom_index in removed_atom_indices
    }
    updated_constitution = plan.structure.constitution.with_chains(prepared_chain_sites)
    return ProteinStructure.from_payload(
        constitution=updated_constitution,
        geometry=plan.structure.geometry.without_atoms(removed_atom_indices),
        topology=StructureTopology(
            constitution=updated_constitution,
            atom_topologies=tuple(
                atom_topology
                for atom_index, atom_topology in enumerate(
                    plan.structure.topology.atom_topologies
                )
                if atom_index not in removed_atom_index_values
            ),
            bonds=plan.structure.topology.bonds_for_constitution(
                source_constitution=plan.structure.constitution,
                target_constitution=updated_constitution,
            ),
        ),
        polymer_blueprint=plan.structure.polymer_blueprint,
        provenance=plan.structure.provenance,
    )


def validate_faspr_residue_site(residue_site: ResidueSite) -> None:
    """Raise when one residue cannot be represented in a FASPR request."""

    if residue_site.is_hetero:
        raise PackingBackendError(
            "FASPR does not support hetero residue "
            f"{residue_site.residue_id.display_token()}"
        )

    if not FASPR_ALPHABET.supports_component(residue_site.component_id):
        raise PackingBackendError(
            f"FASPR does not support component {residue_site.component_id}"
        )

    missing_backbone_atoms = tuple(
        atom_name
        for atom_name in BACKBONE_ATOM_NAMES
        if not residue_site.has_atom_site(atom_name)
    )
    if missing_backbone_atoms:
        raise PackingBackendError(
            f"FASPR requires complete backbone atoms for "
            f"{residue_site.residue_id.display_token()}: "
            f"{', '.join(missing_backbone_atoms)}"
        )


def validate_faspr_output_shape(
    plan: PackingPlan,
    packed_structure: ProteinStructure,
) -> None:
    """Raise when FASPR output no longer matches the input polymer sequence."""

    packed_residue_sites = tuple(
        residue
        for chain in packed_structure.constitution.chains
        for residue in chain.residues
    )
    original_residue_sites = plan.polymer_residue_sites()
    if len(packed_residue_sites) != len(original_residue_sites):
        raise PackingBackendExecutionError(
            "FASPR changed the number of polymer residues unexpectedly"
        )

    for original_residue_site, packed_residue_site in zip(
        original_residue_sites,
        packed_residue_sites,
        strict=True,
    ):
        if original_residue_site.residue_id != packed_residue_site.residue_id:
            raise PackingBackendExecutionError(
                "FASPR changed residue identifiers or order unexpectedly"
            )

    expected_sequence = build_faspr_sequence_override(plan) or "".join(
        FASPR_ALPHABET.sequence_for_residues(original_residue_sites)
    )
    for residue, expected in zip(packed_residue_sites, expected_sequence, strict=True):
        if any(site.is_hydrogen() for site in residue.atom_sites):
            raise PackingBackendExecutionError(
                "FASPR returned unexpected hydrogen atoms"
            )
        if not FASPR_ALPHABET.supports_component(residue.component_id) or (
            FASPR_ALPHABET.require_token(residue.component_id) != expected.upper()
        ):
            raise PackingBackendExecutionError(
                "FASPR changed a component outside the requested sequence"
            )


def strip_hydrogens_from_residue_site(residue_site: ResidueSite) -> ResidueSite:
    """Return one residue site with hydrogen atom sites removed."""

    return residue_site.without_atom_sites(
        tuple(
            atom_site.name
            for atom_site in residue_site.atom_sites
            if atom_site.is_hydrogen()
        )
    )


def _hydrogen_atom_refs_for_residue_site(
    residue_site: ResidueSite,
) -> tuple[AtomRef, ...]:
    """Return canonical atom references for one residue site's hydrogens."""

    return tuple(
        AtomRef(residue_id=residue_site.residue_id, atom_name=atom_site.name)
        for atom_site in residue_site.atom_sites
        if atom_site.is_hydrogen()
    )


def build_faspr_sequence_override(plan: PackingPlan) -> str | None:
    """Return one FASPR sequence override string when the request needs it."""

    if plan.spec.target_sequence is None and not plan.fixed_residue_ids():
        return None

    sequence_override = list(
        FASPR_ALPHABET.sequence_for_residues(plan.polymer_residue_sites())
    )
    replacement_tokens = (
        None if plan.spec.target_sequence is None else tuple(plan.spec.target_sequence)
    )
    if replacement_tokens is not None:
        if plan.spec.mutable_residue_ids is None and len(replacement_tokens) != len(
            sequence_override
        ):
            raise PackingError(
                "full-structure target_sequence must match polymer residue count"
            )
        FASPR_ALPHABET.components_for_sequence(replacement_tokens)

        if plan.spec.mutable_residue_ids is None:
            sequence_override = list(replacement_tokens)
        else:
            replacement_by_residue_id = dict(
                zip(
                    plan.spec.mutable_residue_ids,
                    replacement_tokens,
                    strict=True,
                )
            )
            for index, residue_id in enumerate(plan.polymer_residue_ids):
                replacement_token = replacement_by_residue_id.get(residue_id)
                if replacement_token is not None:
                    sequence_override[index] = replacement_token

    fixed_residue_ids = set(plan.fixed_residue_ids())
    for index, residue_id in enumerate(plan.polymer_residue_ids):
        if residue_id in fixed_residue_ids:
            sequence_override[index] = sequence_override[index].lower()

    return "".join(sequence_override)


def run_faspr(
    execution_input: FasprExecutionInput,
    *,
    executable_path: Path,
    timeout_seconds: float,
) -> ProteinStructure:
    """Run FASPR on one prepared structure and return the packed result."""

    sequence = execution_input.sequence_override or "".join(
        FASPR_ALPHABET.sequence_for_residues(
            execution_input.plan.polymer_residue_sites()
        )
    )
    expected_components = FASPR_ALPHABET.components_for_sequence(sequence)
    try:
        projection = FasprPDBProjection(execution_input.structure.constitution.chains)
        input_text = projection.render(
            _polymer_only_structure(execution_input.structure)
        )
        projected_input = projection.decode(
            input_text,
            expected_components=tuple(
                residue.component_id
                for residue in execution_input.plan.polymer_residue_sites()
            ),
        )
    except (ValueError, StructureNormalizationError) as error:
        raise PackingBackendError(f"FASPR input projection failed: {error}") from error

    with TemporaryDirectory(prefix="protrepair-faspr-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.pdb"
        output_path = temp_dir / "output.pdb"
        input_path.write_text(input_text, encoding="utf-8")

        command = [str(executable_path), "-i", str(input_path), "-o", str(output_path)]
        if execution_input.sequence_override is not None:
            sequence_path = temp_dir / "sequence.txt"
            sequence_path.write_text(
                f"{execution_input.sequence_override}\n",
                encoding="utf-8",
            )
            command.extend(["-s", str(sequence_path)])

        try:
            stdout_path = temp_dir / "stdout.txt"
            stderr_path = temp_dir / "stderr.txt"
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_file,
                stderr_path.open("w", encoding="utf-8") as stderr_file,
            ):
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    cwd=executable_path.parent,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as error:
            raise PackingBackendExecutionError(
                "FASPR execution timed out after "
                f"{timeout_seconds:g} seconds: {executable_path}"
            ) from error
        except OSError as error:
            raise PackingBackendExecutionError(
                "FASPR execution could not start: "
                f"{executable_path} ({_os_error_reason(error)})"
            ) from error
        if completed.returncode != 0:
            error_message = _bounded_output_excerpt(
                stderr_path,
            ) or _bounded_output_excerpt(stdout_path)
            error_message = projection.describe_native_error(
                error_message, max_bytes=MAX_FASPR_CAPTURED_OUTPUT_BYTES
            )
            raise PackingBackendExecutionError(
                "FASPR execution failed with "
                f"exit code {completed.returncode}: {error_message}"
            )

        if not output_path.exists():
            raise PackingBackendExecutionError(
                "FASPR completed without producing an output PDB"
            )

        try:
            _assert_faspr_output_size(output_path)
            packed_core = projection.decode(
                output_path.read_text(encoding="utf-8"),
                expected_components=expected_components,
            )
        except (StructureNormalizationError, ValueError) as error:
            raise PackingBackendExecutionError(
                f"FASPR output did not normalize as a polymer-only structure: {error}"
            ) from error

    validate_faspr_output_shape(execution_input.plan, packed_core)
    try:
        packed_core = _restore_faspr_canonical_state(
            packed_core=packed_core,
            projected_input=projected_input,
            plan=execution_input.plan,
        )
        return _merge_packed_polymer_with_original_ligands(
            packed_core=packed_core,
            original_structure=execution_input.plan.structure,
        )
    except ResidueNotFoundError as error:
        raise PackingBackendExecutionError(
            "FASPR produced an unknown residue identifier"
        ) from error


def _restore_faspr_canonical_state(
    *,
    packed_core: ProteinStructure,
    projected_input: ProteinStructure,
    plan: PackingPlan,
) -> ProteinStructure:
    """Interpret coordinate-only output before H validity and topology merge."""

    original = plan.structure
    fixed = frozenset(plan.fixed_residue_ids())
    chains: list[ChainSite] = []
    geometries: list[AtomGeometry] = []
    charges: list[AtomTopology | None] = []
    mutations: set[ResidueId] = set()
    for chain in packed_core.constitution.chains:
        residues: list[ResidueSite] = []
        for native in chain.residues:
            current = plan.residue_site(native.residue_id)
            same_component = current.component_id == native.component_id
            if not same_component:
                mutations.add(native.residue_id)
            native_by_name = {site.name: site for site in native.atom_sites}
            current_by_name = {site.name: site for site in current.atom_sites}
            added_names = native_by_name.keys() - current_by_name.keys()
            if not same_component or added_names:
                expected_names = frozenset(
                    build_standard_component_library()
                    .require(native.component_id)
                    .expected_polymer_heavy_atom_names()
                )
                # Standard FASPR atom names encode C, N, O or S in their first letter.
                if (
                    not native_by_name.keys() <= expected_names | {"OXT"}
                    or (
                        not same_component
                        and not expected_names <= native_by_name.keys()
                    )
                    or any(
                        site.element != site.name[0]
                        for site in native.atom_sites
                        if not same_component or site.name in added_names
                    )
                ):
                    raise PackingBackendExecutionError(
                        "FASPR returned an incompatible component scaffold"
                    )
            sites = [
                site
                for site in current.atom_sites
                if not site.is_hydrogen()
                and (same_component or site.name in BACKBONE_ATOM_NAMES | {"OXT"})
            ]
            site_names = {site.name for site in sites}
            sites.extend(
                site for site in native.atom_sites if site.name not in site_names
            )
            if native.residue_id in fixed and any(
                site.name not in current_by_name for site in sites
            ):
                raise PackingBackendExecutionError(
                    "FASPR added atoms to a fixed residue "
                    f"{native.residue_id.display_token()}"
                )

            for site in sites:
                ref = AtomRef(native.residue_id, site.name)
                current_index = original.constitution.resolve_atom_index(ref)
                native_site = native_by_name.get(site.name)
                protected = native.residue_id in fixed or site.name in (
                    BACKBONE_ATOM_NAMES | {"OXT"}
                )
                if native_site is None and site.name != "OXT":
                    raise PackingBackendExecutionError(
                        f"FASPR omitted required atom {ref.display_token()}"
                    )
                if native_site is not None and native_site.element != site.element:
                    raise PackingBackendExecutionError(
                        f"FASPR changed atom element {ref.display_token()}"
                    )

                if protected:
                    if current_index is None:
                        raise PackingBackendExecutionError(
                            f"FASPR added protected atom {ref.display_token()}"
                        )
                    if native_site is not None:
                        _require_faspr_projection_equivalence(
                            ref, projected_input, packed_core
                        )
                    geometry = original.geometry.atom_geometry(current_index)
                else:
                    native_index = packed_core.constitution.atom_index(ref)
                    position = packed_core.geometry.atom_geometry(native_index).position
                    geometry = (
                        original.geometry.atom_geometry(current_index).with_position(
                            position
                        )
                        if same_component and current_index is not None
                        else AtomGeometry(position=position)
                    )
                geometries.append(geometry)
                charges.append(
                    original.topology.atom_topologies[current_index.value]
                    if same_component and current_index is not None
                    else None
                )

            residues.append(
                ResidueSite(
                    component_id=native.component_id,
                    residue_id=native.residue_id,
                    atom_sites=tuple(sites),
                    is_hetero=current.is_hetero,
                )
            )
        chains.append(chain.with_residues(residues))

    constitution = packed_core.constitution.with_chains(chains)
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution, atom_geometries=tuple(geometries)
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(charges),
            bonds=_faspr_merged_heavy_bonds(
                original, constitution, mutations=frozenset(mutations)
            ),
        ),
        polymer_blueprint=original.polymer_blueprint,
        provenance=(
            replace(
                original.provenance,
                microstate_choices=tuple(
                    choice
                    for choice in original.provenance.microstate_choices
                    if choice.residue_id not in mutations
                ),
            )
            if mutations
            else original.provenance
        ),
    )


def _require_faspr_projection_equivalence(
    ref: AtomRef, projected_input: ProteinStructure, packed_core: ProteinStructure
) -> None:
    before = projected_input.geometry.atom_geometry(
        projected_input.constitution.atom_index(ref)
    ).position
    after = packed_core.geometry.atom_geometry(
        packed_core.constitution.atom_index(ref)
    ).position
    for expected, actual in zip(before, after, strict=True):
        # FASPR stores float32 coordinates and writes three decimal places.
        conversion_error = abs(float(np.float32(expected)) - expected)
        if abs(actual - expected) > conversion_error + 0.0005 + 1e-10:
            raise PackingBackendExecutionError(
                f"FASPR moved protected atom {ref.display_token()}"
            )


def _faspr_merged_heavy_bonds(
    original: ProteinStructure,
    constitution: StructureConstitution,
    *,
    mutations: frozenset[ResidueId],
) -> tuple[TopologyBond, ...]:
    retained_pairs: set[frozenset[AtomRef]] = set()
    for bond in original.topology.bonds:
        if any(
            original.constitution.atom_site_at(index).is_hydrogen()
            for index in bond.endpoint_pair()
        ):
            continue
        refs = tuple(
            original.constitution.atom_ref_at(index) for index in bond.endpoint_pair()
        )
        altered = any(
            ref.residue_id in mutations
            and ref.atom_name not in FIXED_BACKBONE_ATOM_NAMES
            for ref in refs
        )
        if altered:
            external = refs[0].residue_id != refs[1].residue_id
            intrinsic = False
            if not external and bond.relationship_type is BondRelationshipType.COVALENT:
                residue = original.constitution.residue_site_at(
                    original.constitution.residue_index(refs[0].residue_id)
                )
                intrinsic = refs[1].atom_name in (
                    build_standard_component_library()
                    .require(residue.component_id)
                    .definition.bonded_atom_names(refs[0].atom_name)
                )
            if (
                external
                or bond.relationship_type is not BondRelationshipType.COVALENT
                or (is_source_provenance(bond) and not intrinsic)
            ):
                raise PackingBackendExecutionError(
                    "FASPR mutation would invalidate linked chemistry"
                )
            continue
        retained_pairs.add(frozenset(refs))

    bonds = [
        bond
        for bond in original.topology.bonds_for_constitution(
            source_constitution=original.constitution,
            target_constitution=constitution,
        )
        if frozenset(constitution.atom_ref_at(index) for index in bond.endpoint_pair())
        in retained_pairs
    ]
    occupied = {bond.endpoint_pair() for bond in bonds}
    for residue in constitution.residue_slots:
        new_names = {
            site.name
            for site in residue.atom_sites
            if original.constitution.resolve_atom_index(
                AtomRef(residue.residue_id, site.name)
            )
            is None
        }
        if residue.residue_id not in mutations and not new_names:
            continue
        template = build_standard_component_library().require(residue.component_id)
        for definition in template.definition.bonds:
            names = (definition.atom_name_1, definition.atom_name_2)
            if residue.residue_id not in mutations and not new_names.intersection(
                names
            ):
                continue
            indices = tuple(
                constitution.resolve_atom_index(AtomRef(residue.residue_id, name))
                for name in names
            )
            if indices[0] is None or indices[1] is None:
                continue
            bond = TopologyBond(
                atom_index_1=indices[0],
                atom_index_2=indices[1],
                order=definition.order,
                aromatic=definition.aromatic,
                provenance=BondProvenance.TEMPLATE_RESOLVED,
            )
            if bond.endpoint_pair() not in occupied:
                bonds.append(bond)
                occupied.add(bond.endpoint_pair())
    return tuple(bonds)


def _os_error_reason(error: OSError) -> str:
    """Return one concise launch-error reason for diagnostics."""

    return error.strerror or str(error)


def _bounded_output_excerpt(path: Path) -> str:
    """Return a bounded FASPR output excerpt for error reporting."""

    with path.open("rb") as output_file:
        payload = output_file.read(MAX_FASPR_CAPTURED_OUTPUT_BYTES + 1)

    truncated = len(payload) > MAX_FASPR_CAPTURED_OUTPUT_BYTES
    if truncated:
        payload = payload[:MAX_FASPR_CAPTURED_OUTPUT_BYTES]

    excerpt = payload.decode("utf-8", errors="replace").strip()
    if not excerpt:
        return ""
    if truncated:
        return f"{excerpt}... [truncated after {MAX_FASPR_CAPTURED_OUTPUT_BYTES} bytes]"

    return excerpt


def _assert_faspr_output_size(path: Path) -> None:
    """Reject oversized FASPR output before reading it into memory."""

    output_size = path.stat().st_size
    if output_size > MAX_STRUCTURE_INPUT_BYTES:
        raise PackingBackendExecutionError(
            f"FASPR output exceeded {MAX_STRUCTURE_INPUT_BYTES} bytes: {path.name}"
        )


def _polymer_only_structure(structure: ProteinStructure) -> ProteinStructure:
    """Return one structure restricted to polymer constitution payload."""

    polymer_only_constitution = structure.constitution.with_ligands(())
    polymer_geometry = StructureGeometry(
        constitution=polymer_only_constitution,
        atom_geometries=tuple(
            structure.geometry.residue_geometry(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(
                    residue_site.residue_id
                ),
            ).atom_geometry(atom_site.name)
            for residue_site in polymer_only_constitution.residue_slots
            for atom_site in residue_site.atom_sites
        ),
    )
    polymer_topology = StructureTopology(
        constitution=polymer_only_constitution,
        atom_topologies=tuple(
            (
                None
                if formal_charge is None
                else AtomTopology(formal_charge=formal_charge)
            )
            for residue_site in polymer_only_constitution.residue_slots
            for formal_charge_by_name in (
                dict(
                    structure.topology.residue_formal_charge_by_atom_name(
                        constitution=structure.constitution,
                        residue_index=structure.constitution.residue_index(
                            residue_site.residue_id
                        ),
                    )
                ),
            )
            for atom_site in residue_site.atom_sites
            for formal_charge in (formal_charge_by_name.get(atom_site.name),)
        ),
        bonds=structure.topology.bonds_for_constitution(
            source_constitution=structure.constitution,
            target_constitution=polymer_only_constitution,
        ),
    )

    return ProteinStructure.from_payload(
        constitution=polymer_only_constitution,
        geometry=polymer_geometry,
        topology=polymer_topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _merge_packed_polymer_with_original_ligands(
    *,
    packed_core: ProteinStructure,
    original_structure: ProteinStructure,
) -> ProteinStructure:
    """Return one packed polymer structure with original ligands restored."""

    updated_constitution = _packed_constitution_with_preserved_hydrogens(
        packed_core=packed_core,
        original_structure=original_structure,
    ).with_ligands(original_structure.constitution.ligands)
    ligand_residue_id_set = frozenset(
        ligand.residue_id for ligand in original_structure.constitution.ligands
    )
    updated_geometry = StructureGeometry(
        constitution=updated_constitution,
        atom_geometries=tuple(
            (
                original_structure.geometry.residue_geometry(
                    constitution=original_structure.constitution,
                    residue_index=original_structure.constitution.residue_index(
                        residue_site.residue_id
                    ),
                )
                if (
                    residue_site.residue_id in ligand_residue_id_set
                    or atom_site.is_hydrogen()
                )
                else packed_core.geometry.residue_geometry(
                    constitution=packed_core.constitution,
                    residue_index=packed_core.constitution.residue_index(
                        residue_site.residue_id
                    ),
                )
            ).atom_geometry(atom_site.name)
            for residue_site in updated_constitution.residue_slots
            for atom_site in residue_site.atom_sites
        ),
    )
    updated_topology = StructureTopology(
        constitution=updated_constitution,
        atom_topologies=_merged_atom_topologies(
            updated_constitution=updated_constitution,
            packed_core=packed_core,
            original_structure=original_structure,
            ligand_residue_id_set=ligand_residue_id_set,
        ),
        bonds=(
            *packed_core.topology.bonds_for_constitution(
                source_constitution=packed_core.constitution,
                target_constitution=updated_constitution,
            ),
            *(
                bond
                for bond in original_structure.topology.bonds_for_constitution(
                    source_constitution=original_structure.constitution,
                    target_constitution=updated_constitution,
                )
                if any(
                    updated_constitution.atom_site_at(index).is_hydrogen()
                    or updated_constitution.atom_ref_at(index).residue_id
                    in ligand_residue_id_set
                    for index in bond.endpoint_pair()
                )
            ),
        ),
    )
    return ProteinStructure.from_payload(
        constitution=updated_constitution,
        geometry=updated_geometry,
        topology=updated_topology,
        polymer_blueprint=original_structure.polymer_blueprint,
        provenance=packed_core.provenance,
    )


def _merged_atom_topologies(
    *,
    updated_constitution: StructureConstitution,
    packed_core: ProteinStructure,
    original_structure: ProteinStructure,
    ligand_residue_id_set: frozenset[ResidueId],
) -> tuple[AtomTopology | None, ...]:
    """Return atom topologies aligned to one FASPR merged constitution."""

    atom_topologies: list[AtomTopology | None] = []
    for residue_site in updated_constitution.residue_slots:
        original_formal_charge_by_name = dict(
            original_structure.topology.residue_formal_charge_by_atom_name(
                constitution=original_structure.constitution,
                residue_index=original_structure.constitution.residue_index(
                    residue_site.residue_id
                ),
            )
        )
        packed_formal_charge_by_name = (
            {}
            if (
                residue_site.residue_id in ligand_residue_id_set
                or packed_core.constitution.residue_or_ligand(residue_site.residue_id)
                is None
            )
            else dict(
                packed_core.topology.residue_formal_charge_by_atom_name(
                    constitution=packed_core.constitution,
                    residue_index=packed_core.constitution.residue_index(
                        residue_site.residue_id
                    ),
                )
            )
        )
        for atom_site in residue_site.atom_sites:
            formal_charge = (
                original_formal_charge_by_name.get(atom_site.name)
                if (
                    residue_site.residue_id in ligand_residue_id_set
                    or atom_site.is_hydrogen()
                )
                else packed_formal_charge_by_name.get(atom_site.name)
            )
            atom_topologies.append(
                None
                if formal_charge is None
                else AtomTopology(formal_charge=formal_charge)
            )

    return tuple(atom_topologies)


def _packed_constitution_with_preserved_hydrogens(
    *,
    packed_core: ProteinStructure,
    original_structure: ProteinStructure,
) -> StructureConstitution:
    """Return packed polymer constitution with still-valid original H restored."""

    original_chains_by_id = {
        chain_site.chain_id: chain_site
        for chain_site in original_structure.constitution.chains
    }
    merged_chains: list[ChainSite] = []
    for packed_chain_site in packed_core.constitution.chains:
        original_chain_site = original_chains_by_id.get(packed_chain_site.chain_id)
        if original_chain_site is None:
            merged_chains.append(packed_chain_site)
            continue

        original_residues_by_id = {
            residue_site.residue_id: residue_site
            for residue_site in original_chain_site.residues
        }
        merged_residues: list[ResidueSite] = []
        for packed_residue_site in packed_chain_site.residues:
            original_residue_site = original_residues_by_id.get(
                packed_residue_site.residue_id
            )
            if original_residue_site is not None and _should_preserve_hydrogens(
                original_structure=original_structure,
                packed_core=packed_core,
                original_residue_site=original_residue_site,
                packed_residue_site=packed_residue_site,
            ):
                merged_residues.append(
                    _packed_residue_with_original_hydrogens(
                        packed_residue_site=packed_residue_site,
                        original_residue_site=original_residue_site,
                    )
                )
            else:
                merged_residues.append(packed_residue_site)

        merged_chains.append(packed_chain_site.with_residues(merged_residues))

    return packed_core.constitution.with_chains(merged_chains)


def _should_preserve_hydrogens(
    *,
    original_structure: ProteinStructure,
    packed_core: ProteinStructure,
    original_residue_site: ResidueSite,
    packed_residue_site: ResidueSite,
) -> bool:
    """Return whether original residue hydrogens remain valid after packing."""

    if not _hydrogen_atom_sites(original_residue_site):
        return False

    if _heavy_atom_site_signature(original_residue_site) != _heavy_atom_site_signature(
        packed_residue_site
    ):
        return False

    if _heavy_atom_formal_charge_by_name(
        structure=original_structure,
        residue_site=original_residue_site,
    ) != _heavy_atom_formal_charge_by_name(
        structure=packed_core,
        residue_site=packed_residue_site,
    ):
        return False

    original_geometry = original_structure.geometry.residue_geometry(
        constitution=original_structure.constitution,
        residue_index=original_structure.constitution.residue_index(
            original_residue_site.residue_id
        ),
    )
    packed_geometry = packed_core.geometry.residue_geometry(
        constitution=packed_core.constitution,
        residue_index=packed_core.constitution.residue_index(
            packed_residue_site.residue_id
        ),
    )
    for atom_site in original_residue_site.atom_sites:
        if atom_site.is_hydrogen():
            continue

        distance = original_geometry.atom_geometry(atom_site.name).distance_to(
            packed_geometry.atom_geometry(atom_site.name)
        )
        if distance > FASPR_HYDROGEN_PRESERVATION_HEAVY_ATOM_TOLERANCE_ANGSTROM:
            return False

    return True


def _packed_residue_with_original_hydrogens(
    *,
    packed_residue_site: ResidueSite,
    original_residue_site: ResidueSite,
) -> ResidueSite:
    """Return a packed residue with original H sites restored in source order."""

    return packed_residue_site.with_atom_sites(
        _hydrogen_atom_sites(original_residue_site)
    ).reordered_atom_sites(original_residue_site.atom_site_names())


def _hydrogen_atom_sites(residue_site: ResidueSite) -> tuple[AtomSite, ...]:
    """Return polymer hydrogen atom sites in residue order."""

    return tuple(
        atom_site for atom_site in residue_site.atom_sites if atom_site.is_hydrogen()
    )


def _heavy_atom_site_signature(
    residue_site: ResidueSite,
) -> tuple[str, bool, tuple[tuple[str, str], ...]]:
    """Return heavy atom names and elements in residue order."""

    return (
        residue_site.component_id,
        residue_site.is_hetero,
        tuple(
            (atom_site.name, atom_site.element)
            for atom_site in residue_site.atom_sites
            if not atom_site.is_hydrogen()
        ),
    )


def _heavy_atom_formal_charge_by_name(
    *,
    structure: ProteinStructure,
    residue_site: ResidueSite,
) -> tuple[tuple[str, int | None], ...]:
    """Return formal charges for one residue's heavy atoms."""

    residue_index = structure.constitution.residue_index(residue_site.residue_id)
    heavy_atom_names = {
        atom_site.name
        for atom_site in residue_site.atom_sites
        if not atom_site.is_hydrogen()
    }
    return tuple(
        (atom_name, formal_charge)
        for atom_name, formal_charge in (
            structure.topology.residue_formal_charge_by_atom_name(
                constitution=structure.constitution,
                residue_index=residue_index,
            )
        )
        if atom_name in heavy_atom_names
    )


def _linked_sidechain_packing_issues(
    selection: FasprPackingSelection,
) -> tuple[ValidationIssue, ...]:
    """Report excluded movement separately from unconstrained reconstruction."""

    issues: list[ValidationIssue] = []
    for kind, residue_ids, message in (
        (
            ValidationIssueKind.PACKING_PRESERVED_LINKED_SIDECHAIN,
            selection.protected_residue_ids,
            (
                "FASPR preserved this complete linked sidechain instead of repacking "
                "it; existing link geometry was not optimized"
            ),
        ),
        (
            ValidationIssueKind.PACKING_REBUILT_LINKED_SIDECHAIN,
            selection.incomplete_linked_residue_ids,
            (
                "FASPR rebuilt this incomplete linked sidechain; canonical links were "
                "retained, but their geometry was not constrained"
            ),
        ),
    ):
        for residue_id in residue_ids:
            issues.append(
                ValidationIssue.for_residue(
                    kind=kind,
                    severity=IssueSeverity.WARNING,
                    message=message,
                    residue_id=residue_id,
                )
            )
    return tuple(issues)


def infer_packing_issues(
    plan: PackingPlan,
    packed: ProteinStructure,
) -> tuple[ValidationIssue, ...]:
    """Return structural warnings inferred from one packed structure."""

    issues: list[ValidationIssue] = []
    for residue_site in packed.constitution.iter_residues():
        try:
            original_residue_site = plan.residue_site(residue_site.residue_id)
        except ResidueNotFoundError as error:
            raise PackingBackendExecutionError(
                "FASPR produced an unknown residue identifier"
            ) from error

        missing_hydrogen_names = _missing_original_polymer_hydrogen_names(
            original_residue_site=original_residue_site,
            packed_residue_site=residue_site,
        )
        if missing_hydrogen_names:
            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.PACKING_INVALIDATED_HYDROGENS,
                    severity=IssueSeverity.WARNING,
                    message=(
                        "FASPR invalidated polymer hydrogens "
                        f"{', '.join(missing_hydrogen_names)}; run polymer "
                        "hydrogen completion to restore hydrogen coverage"
                    ),
                    residue_id=residue_site.residue_id,
                )
            )

        if residue_site.is_hetero:
            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.UNSUPPORTED_COMPONENT,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"FASPR returned unexpected hetero residue "
                        f"{residue_site.residue_id.display_token()}"
                    ),
                    residue_id=residue_site.residue_id,
                )
            )

    return tuple(issues)


def _missing_original_polymer_hydrogen_names(
    *,
    original_residue_site: ResidueSite,
    packed_residue_site: ResidueSite,
) -> tuple[str, ...]:
    """Return original polymer hydrogens absent from packed output."""

    packed_atom_names = set(packed_residue_site.atom_site_names())
    return tuple(
        atom_site.name
        for atom_site in original_residue_site.atom_sites
        if atom_site.is_hydrogen() and atom_site.name not in packed_atom_names
    )
