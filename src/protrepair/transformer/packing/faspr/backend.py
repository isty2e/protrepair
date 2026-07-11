"""FASPR specialization of the generic side-chain packing backend seam."""

import subprocess
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from typing_extensions import Self

from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import (
    PackingError,
    ProtrepairError,
    ResidueNotFoundError,
    StructureNormalizationError,
)
from protrepair.io import write_structure_string
from protrepair.io.gemmi_ingress import (
    MAX_STRUCTURE_INPUT_BYTES,
    read_structure_string_with_policy,
)
from protrepair.io.ingress_policy import (
    LigandHandling,
    StructureNormalizationPolicy,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.packing.alphabet import PackingAlphabet
from protrepair.transformer.packing.domain import (
    PackingCapabilities,
    PackingPlan,
    PackingResult,
)
from protrepair.transformer.packing.faspr.paths import (
    faspr_executable_path,
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
        execution_input = FasprExecutionInput.from_plan(plan)
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
        validate_faspr_output_shape(plan, packed_structure)
        changed_residue_ids = plan.changed_residue_ids_after(packed_structure)
        issues = infer_packing_issues(plan, packed_structure)
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
    """Backend-specific execution input derived from a generic packing plan."""

    structure: ProteinStructure
    original_structure: ProteinStructure
    sequence_override: str | None = None

    @classmethod
    def from_plan(cls, plan: PackingPlan) -> Self:
        """Build one FASPR execution input from a generic packing plan."""

        prepared_structure = prepare_structure_for_faspr(plan)
        sequence_override = build_faspr_sequence_override(plan)
        return cls(
            structure=prepared_structure,
            original_structure=plan.structure,
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

    prepared_chain_sites: list[ChainSite] = []
    removed_atom_refs: list[AtomRef] = []
    for chain_site in plan.polymer_chain_sites():
        prepared_residue_sites: list[ResidueSite] = []
        for residue_site in chain_site.residues:
            validate_faspr_residue_site(residue_site)
            prepared_residue_sites.append(
                strip_hydrogens_from_residue_site(residue_site)
            )
            removed_atom_refs.extend(_hydrogen_atom_refs_for_residue_site(residue_site))

        prepared_chain_sites.append(chain_site.with_residues(prepared_residue_sites))

    removed_atom_ref_set = frozenset(removed_atom_refs)
    removed_atom_indices = frozenset(
        plan.structure.constitution.atom_index(atom_ref)
        for atom_ref in removed_atom_ref_set
    )
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
                if atom_index
                not in {
                    removed_atom_index.value
                    for removed_atom_index in removed_atom_indices
                }
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
        if plan.spec.mutable_residue_ids is None:
            if len(replacement_tokens) != len(sequence_override):
                raise PackingError(
                    "full-structure target_sequence must match polymer residue count"
                )
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

    with TemporaryDirectory(prefix="protrepair-faspr-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.pdb"
        output_path = temp_dir / "output.pdb"
        input_path.write_text(
            write_structure_string(
                _polymer_only_structure(execution_input.structure),
                FileFormat.PDB,
            ),
            encoding="utf-8",
        )

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
            with stdout_path.open("w", encoding="utf-8") as stdout_file:
                with stderr_path.open("w", encoding="utf-8") as stderr_file:
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
            packed_core = read_structure_string_with_policy(
                output_path.read_text(encoding="utf-8"),
                FileFormat.PDB,
                policy=StructureNormalizationPolicy(
                    ligand_handling=LigandHandling.REJECT
                ),
                source_name=execution_input.structure.provenance.ingress.source_name,
            )
        except StructureNormalizationError as error:
            raise PackingBackendExecutionError(
                "FASPR output did not normalize as a polymer-only structure"
            ) from error

    try:
        return _merge_packed_polymer_with_original_ligands(
            packed_core=packed_core,
            original_structure=execution_input.original_structure,
        )
    except ResidueNotFoundError as error:
        raise PackingBackendExecutionError(
            "FASPR produced an unknown residue identifier"
        ) from error


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
        bonds=original_structure.topology.bonds_for_constitution(
            source_constitution=original_structure.constitution,
            target_constitution=updated_constitution,
        ),
    )
    return ProteinStructure.from_payload(
        constitution=updated_constitution,
        geometry=updated_geometry,
        topology=updated_topology,
        polymer_blueprint=original_structure.polymer_blueprint,
        provenance=original_structure.provenance,
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
