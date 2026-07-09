"""Canonical structure-carrier entities and facets."""

from protrepair.structure.address_space import (
    StructureAddressSpaceKey,
    address_space_key_without_atoms,
    atom_count_in_address_space,
)
from protrepair.structure.aggregate import ProteinStructure, ResidueFacetPayload
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.element import ElementIdentity
from protrepair.structure.endpoint import (
    StructureCarrierHandle,
    StructureCarrierKind,
    StructureEndpoint,
    StructureRealizationSelector,
)
from protrepair.structure.geometry import (
    AtomGeometry,
    ResidueGeometry,
    StructureGeometry,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.polymer_blueprint import (
    PolymerBlueprint,
    PolymerChainBlueprint,
    PolymerResidueSlot,
)
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
    StructureProvenanceOrigin,
)
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)

__all__ = [
    "AtomGeometry",
    "AtomRef",
    "AtomSite",
    "AtomTopology",
    "BondProvenance",
    "BondRelationshipType",
    "ChainSite",
    "ElementIdentity",
    "FileFormat",
    "PolymerBlueprint",
    "PolymerChainBlueprint",
    "PolymerResidueSlot",
    "ProteinStructure",
    "ResidueFacetPayload",
    "ResidueGeometry",
    "ResidueId",
    "ResidueSite",
    "SourceBondMetadata",
    "SourceBondRecordType",
    "StructureAddressSpaceKey",
    "StructureCarrierHandle",
    "StructureCarrierKind",
    "StructureConstitution",
    "StructureEndpoint",
    "StructureGeometry",
    "StructureIngress",
    "StructureProvenance",
    "StructureProvenanceOrigin",
    "StructureRealizationSelector",
    "StructureTopology",
    "TopologyBond",
    "address_space_key_without_atoms",
    "atom_count_in_address_space",
]
