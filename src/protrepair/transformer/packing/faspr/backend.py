"""FASPR specialization of the generic side-chain packing backend seam."""

import subprocess
from dataclasses import dataclass
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
from protrepair.io.gemmi_ingress import read_structure_string_with_policy
from protrepair.io.ingress_policy import (
    LigandHandling,
    StructureNormalizationPolicy,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef
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
FASPR_CAPABILITIES = PackingCapabilities(
    supports_full_structure_packing=True,
    supports_local_packing=True,
    supports_partial_sequence=True,
    supports_refinement=False,
    supports_noncanonical_components=False,
    deterministic_given_same_inputs=True,
)
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

    def capabilities(self) -> PackingCapabilities:
        """Return the declared capability surface of the FASPR backend."""

        return FASPR_CAPABILITIES

    def pack(self, plan: PackingPlan) -> PackingResult:
        """Execute one canonical packing plan through FASPR."""

        plan.assert_supported_by(self.capabilities())
        execution_input = FasprExecutionInput.from_plan(plan)
        executable_path = resolve_faspr_executable_path(self.executable_path)
        validate_rotamer_library_near(executable_path)

        packed_structure = run_faspr(
            execution_input,
            executable_path=executable_path,
        )
        changed_residue_ids = plan.changed_residue_ids_after(packed_structure)
        issues = infer_packing_issues(plan, packed_structure)
        return PackingResult(
            packed_structure=packed_structure,
            changed_residue_ids=changed_residue_ids,
            issues=issues,
            backend_name="faspr",
            backend_version=None,
        )


@dataclass(frozen=True, slots=True)
class FasprExecutionInput:
    """Backend-specific execution input derived from a generic packing plan."""

    structure: ProteinStructure
    sequence_override: str | None = None

    @classmethod
    def from_plan(cls, plan: PackingPlan) -> Self:
        """Build one FASPR execution input from a generic packing plan."""

        prepared_structure = prepare_structure_for_faspr(plan)
        sequence_override = build_faspr_sequence_override(plan)
        return cls(
            structure=prepared_structure,
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

    return resolved_path


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


def strip_hydrogens_from_residue_site(residue_site: ResidueSite) -> ResidueSite:
    """Return one residue site with hydrogen atom sites removed."""

    return residue_site.without_atom_sites(
        tuple(
            atom_site.name
            for atom_site in residue_site.atom_sites
            if atom_site.element == "H"
        )
    )


def _hydrogen_atom_refs_for_residue_site(
    residue_site: ResidueSite,
) -> tuple[AtomRef, ...]:
    """Return canonical atom references for one residue site's hydrogens."""

    return tuple(
        AtomRef(residue_id=residue_site.residue_id, atom_name=atom_site.name)
        for atom_site in residue_site.atom_sites
        if atom_site.element == "H"
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

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=executable_path.parent,
        )
        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip()
            raise PackingBackendExecutionError(
                "FASPR execution failed with "
                f"exit code {completed.returncode}: {error_message}"
            )

        if not output_path.exists():
            raise PackingBackendExecutionError(
                "FASPR completed without producing an output PDB"
            )

        try:
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

    return _merge_packed_polymer_with_original_ligands(
        packed_core=packed_core,
        original_structure=execution_input.structure,
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

    updated_constitution = packed_core.constitution.with_ligands(
        original_structure.constitution.ligands
    )
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
                if residue_site.residue_id in ligand_residue_id_set
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
        atom_topologies=tuple(
            (
                None
                if formal_charge is None
                else AtomTopology(formal_charge=formal_charge)
            )
            for residue_site in updated_constitution.residue_slots
            for formal_charge_by_name in (
                dict(
                    original_structure.topology.residue_formal_charge_by_atom_name(
                        constitution=original_structure.constitution,
                        residue_index=original_structure.constitution.residue_index(
                            residue_site.residue_id
                        ),
                    )
                )
                if residue_site.residue_id in ligand_residue_id_set
                else dict(
                    packed_core.topology.residue_formal_charge_by_atom_name(
                        constitution=packed_core.constitution,
                        residue_index=packed_core.constitution.residue_index(
                            residue_site.residue_id
                        ),
                    )
                ),
            )
            for atom_site in residue_site.atom_sites
            for formal_charge in (formal_charge_by_name.get(atom_site.name),)
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


def infer_packing_issues(
    plan: PackingPlan,
    packed: ProteinStructure,
) -> tuple[ValidationIssue, ...]:
    """Return structural warnings inferred from one packed structure."""

    issues: list[ValidationIssue] = []
    for residue_site in packed.constitution.iter_residues():
        try:
            plan.residue_site(residue_site.residue_id)
        except ResidueNotFoundError as error:
            raise PackingBackendExecutionError(
                "FASPR produced an unknown residue identifier"
            ) from error

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
