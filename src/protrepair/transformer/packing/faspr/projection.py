"""Reversible residue identities for the FASPR calculation PDB boundary."""

import re
from dataclasses import dataclass, replace
from string import ascii_lowercase, ascii_uppercase, digits

from protrepair.io.gemmi_ingress import read_structure_string_with_policy
from protrepair.io.gemmi_writer import write_pdb_structure_string_without_conect
from protrepair.io.ingress_policy import LigandHandling, StructureNormalizationPolicy
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, StructureConstitution
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.topology import StructureTopology

FASPR_INSERTION_ALPHABET = " " + ascii_uppercase + ascii_lowercase + digits
FASPR_RESIDUE_NUMBER_COUNT = 10999
FASPR_RESIDUE_CAPACITY = FASPR_RESIDUE_NUMBER_COUNT * len(FASPR_INSERTION_ALPHABET)
_NATIVE_LOCATION = re.compile(r"(residue:? )([A-Z]{3})(-?\d+)(?=[,\s]|$)")


def faspr_residue_id(ordinal: int) -> ResidueId:
    """Allocate one globally unique decimal number and insertion-code pair.

    Parameters
    ----------
    ordinal : int
        Zero-based residue position in native sequence order.

    Returns
    -------
    ResidueId
        A single-character chain, decimal residue number and insertion code.

    Raises
    ------
    ValueError
        The ordinal is negative or exhausts the calculation namespace.
    """

    if not 0 <= ordinal < FASPR_RESIDUE_CAPACITY:
        raise ValueError(
            f"FASPR calculation namespace supports at most {FASPR_RESIDUE_CAPACITY} residues"
        )
    insertion, number = divmod(ordinal, FASPR_RESIDUE_NUMBER_COUNT)
    return ResidueId("A", number - 999, FASPR_INSERTION_ALPHABET[insertion])


@dataclass(frozen=True, slots=True)
class FasprPDBProjection:
    """Request-local correspondence between native labels and source residues.

    Parameters
    ----------
    source_chains : tuple[ChainSite, ...]
        Hydrogen-stripped polymer chains in packing sequence order. These
        identifiers, not the native file's labels, determine the restored chains.

    Raises
    ------
    ValueError
        The residue count exhausts the calculation namespace.
    """

    source_chains: tuple[ChainSite, ...]

    def __post_init__(self) -> None:
        count = sum(len(chain.residues) for chain in self.source_chains)
        if count:
            faspr_residue_id(count - 1)

    def render(self, structure: ProteinStructure) -> str:
        """Render polymer-only input with calculation labels before Gemmi sees it.

        Parameters
        ----------
        structure : ProteinStructure
            Prepared polymer-only coordinates matching the source residue sites.

        Returns
        -------
        str
            Calculation PDB with selected altloc labels blanked by the writer.

        Raises
        ------
        ValueError
            The input residue inventory differs from the correspondence.
        """

        residues = tuple(structure.constitution.iter_residues())
        source_residues = tuple(
            residue for chain in self.source_chains for residue in chain.residues
        )
        if structure.constitution.ligands or residues != source_residues:
            raise ValueError(
                "FASPR projection input does not match its source residues"
            )

        chains = (
            ChainSite(
                chain_id="A",
                residues=tuple(
                    replace(residue, residue_id=faspr_residue_id(index))
                    for index, residue in enumerate(residues)
                ),
            ),
        )
        transport = _with_transport_chains(structure, chains)
        return write_pdb_structure_string_without_conect(transport)

    def decode(
        self, pdb_text: str, *, expected_components: tuple[str, ...]
    ) -> ProteinStructure:
        """Validate native records and restore source chain partitions and IDs.

        Parameters
        ----------
        pdb_text : str
            Actual serialized input or native output.
        expected_components : tuple[str, ...]
            Input components, or requested output components after mutation,
            in native sequence order.

        Returns
        -------
        ProteinStructure
            Coordinate transport with original residue IDs, without source
            observations, blueprint or saved chemistry choices.

        Raises
        ------
        ValueError
            Native residue inventory, order, components or atom names disagree
            with the request. No partial output is accepted.
        StructureNormalizationError
            The remaining coordinate payload cannot be normalized.
        """

        self._require_records(pdb_text, expected_components)
        native = read_structure_string_with_policy(
            pdb_text,
            FileFormat.PDB,
            policy=StructureNormalizationPolicy(ligand_handling=LigandHandling.REJECT),
        )
        residues = tuple(native.constitution.iter_residues())
        if len(residues) != len(expected_components):
            raise ValueError(
                "FASPR changed the number of polymer residues unexpectedly"
            )

        chains = []
        offset = 0
        for source_chain in self.source_chains:
            stop = offset + len(source_chain.residues)
            chains.append(
                ChainSite(
                    chain_id=source_chain.chain_id,
                    residues=tuple(
                        replace(residue, residue_id=source.residue_id)
                        for residue, source in zip(
                            residues[offset:stop], source_chain.residues, strict=True
                        )
                    ),
                )
            )
            offset = stop
        return _with_transport_chains(native, tuple(chains))

    def _require_records(
        self, pdb_text: str, expected_components: tuple[str, ...]
    ) -> None:
        if len(expected_components) != sum(
            len(chain.residues) for chain in self.source_chains
        ):
            raise ValueError(
                "FASPR component inventory does not match the source residues"
            )
        previous: str | None = None
        ordinal = -1
        names: set[str] = set()
        model_count = 0
        for line in pdb_text.splitlines():
            if line.startswith("MODEL "):
                model_count += 1
                if model_count > 1:
                    raise ValueError("FASPR returned multiple models")
            if line.startswith("HETATM"):
                raise ValueError("FASPR output is not a polymer-only structure")
            if not line.startswith("ATOM  "):
                continue

            key = line[20:27]
            if key != previous:
                ordinal += 1
                if ordinal >= len(expected_components):
                    raise ValueError(
                        "FASPR changed the number of polymer residues unexpectedly"
                    )
                expected = faspr_residue_id(ordinal)
                expected_key = (
                    f" A{expected.seq_num:4d}{expected.insertion_code or ' '}"
                )
                if key != expected_key:
                    raise ValueError(
                        "FASPR changed residue identifiers or order unexpectedly"
                    )
                previous = key
                names = set()
            if line[17:20].strip() != expected_components[ordinal]:
                raise ValueError(
                    "FASPR changed a component outside the requested sequence"
                )
            name = line[12:16].strip().upper()
            if not name or name in names or line[16:17] != " ":
                raise ValueError("FASPR returned duplicate or alternate atom records")
            names.add(name)

        if ordinal + 1 != len(expected_components):
            raise ValueError(
                "FASPR changed the number of polymer residues unexpectedly"
            )

    def describe_native_error(self, excerpt: str, *, max_bytes: int = 64 * 1024) -> str:
        """Translate only unambiguous stock component/number log locations.

        Parameters
        ----------
        excerpt : str
            Bounded native stdout or stderr excerpt.
        max_bytes : int, default=65536
            Maximum translated payload, excluding the prefix and truncation note.

        Returns
        -------
        str
            Explicitly marked native output, with proven source locations restored.
        """

        locations: dict[tuple[str, int], list[ResidueId]] = {}
        for ordinal, residue in enumerate(
            residue for chain in self.source_chains for residue in chain.residues
        ):
            key = (residue.component_id, faspr_residue_id(ordinal).seq_num)
            locations.setdefault(key, []).append(residue.residue_id)

        def restore(match: re.Match[str]) -> str:
            candidates = locations.get((match[2], int(match[3])), ())
            if len(candidates) != 1:
                return match[0]
            return f"{match[1]}{match[2]} {candidates[0].display_token()}"

        payload = _NATIVE_LOCATION.sub(restore, excerpt).encode("utf-8")
        if len(payload) > max_bytes:
            text = payload[:max_bytes].decode("utf-8", errors="replace")
            text += f"... [truncated after {max_bytes} bytes]"
        else:
            text = payload.decode("utf-8")
        return "native FASPR output: " + text


def _with_transport_chains(
    structure: ProteinStructure, chains: tuple[ChainSite, ...]
) -> ProteinStructure:
    # Slot order is unchanged. Source observations, blueprint and saved chemistry
    # choices are deliberately absent from this coordinate-only transport.
    constitution = StructureConstitution(chains=chains, ligands=())
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution,
            atom_geometries=structure.geometry.atom_geometries,
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=structure.topology.bonds,
        ),
        provenance=StructureProvenance(ingress=StructureIngress(FileFormat.PDB)),
    )
