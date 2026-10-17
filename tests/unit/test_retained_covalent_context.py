"""Isolated templates and proximity fallback cannot explain an extra bond."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import (
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_external_bonds,
)
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.geometry import Vec3
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)


def _source(component, relationship, provenance):
    if component == "SAN":
        semantics = (
            build_default_component_library().require(component).hydrogen_semantics
        )
        assert isinstance(semantics, IdealGeometryHydrogenSemantics)
        atoms = tuple(
            atom_payload(
                atom.atom_name, atom.element, Vec3.from_iterable(atom.ideal_position)
            )
            for atom in semantics.component.heavy_atoms()
            if atom.ideal_position is not None
        )
        anchor = "N1"
    else:
        atoms = (
            atom_payload("C1", "C", Vec3(-1.43, 0, 0)),
            atom_payload("O1", "O", Vec3(0, 0, 0)),
        )
        anchor = "O1"
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=component,
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=atoms,
            ),
            residue_payload(
                component_id="YYY",
                residue_id=ResidueId("L", 2),
                is_hetero=True,
                atoms=(atom_payload("C2", "C", Vec3(20, 0, 0)),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    if relationship is None:
        return source
    external = TopologyBond(
        source.constitution.atom_index(AtomRef(ResidueId("L", 1), anchor)),
        source.constitution.atom_index(AtomRef(ResidueId("L", 2), "C2")),
        relationship_type=relationship,
        provenance=provenance,
    )
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(external,),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


@pytest.mark.parametrize("component", ("SAN", "XXX"))
@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.UNKNOWN,
    ),
)
@pytest.mark.parametrize("provenance", tuple(BondProvenance))
def test_isolated_chemistry_refuses_unexplained_external_edges(
    component, relationship, provenance
):
    source = _source(component, relationship, provenance)
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "external" in resolution.failure_reason
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(
            source,
            build_default_component_library(),
            target_residue_ids=frozenset((ResidueId("L", 1),)),
        )
        assert result.structure is source and not result.repairs and result.issues


@pytest.mark.parametrize("component", ("SAN", "XXX"))
@pytest.mark.parametrize(
    "relationship",
    (None, BondRelationshipType.METAL_COORDINATION, BondRelationshipType.HYDROGEN_BOND),
)
def test_typed_noncovalent_context_is_not_an_external_covalent_adduct(
    component, relationship
):
    source = _source(component, relationship, BondProvenance.SOURCE_EXPLICIT)
    RetainedCovalentContext(source, ResidueId("L", 1)).require_admitted(())
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("L", 1),))
    )
    if relationship is BondRelationshipType.METAL_COORDINATION:
        # This fixture connects N/O to carbon, not to an admitted metal partner.
        assert result.structure is source and result.issues and not result.repairs
        return
    assert not result.issues or all(
        "inferred via RDKit" in issue.message for issue in result.issues
    )
    ligand = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in ligand.atom_sites) == (
        8 if component == "SAN" else 4
    )
    if relationship is not None:
        original = source.topology.bonds[0]
        refs = tuple(
            source.constitution.atom_ref_at(index) for index in original.endpoint_pair()
        )
        actual = result.structure.topology.bond_between(
            *(result.structure.constitution.atom_index(ref) for ref in refs)
        )
        assert actual is not None
        assert actual.order == original.order
        assert actual.aromatic == original.aromatic
        assert actual.relationship_type is original.relationship_type
        assert actual.provenance is original.provenance
        assert actual.source_metadata == original.source_metadata


@pytest.mark.parametrize("incomplete", (False, True))
@pytest.mark.parametrize("order", (None, 2))
def test_unsupported_plp_imine_does_not_claim_to_be_an_acyl_application(
    incomplete, order
):
    template = build_default_component_library().require("PLP")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    ligand_id = ResidueId("L", 1)
    lysine_id = ResidueId("A", 1)
    source = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="LYS",
                        residue_id=lysine_id,
                        atoms=(atom_payload("NZ", "N", Vec3(10, 0, 0)),),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="PLP",
                residue_id=ligand_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        atom.atom_name,
                        atom.element,
                        Vec3.from_iterable(atom.ideal_position),
                    )
                    for atom in semantics.component.heavy_atoms()
                    if atom.ideal_position is not None
                    and (not incomplete or atom.atom_name != "P")
                ),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    source.constitution.atom_index(AtomRef(ligand_id, "C4A")),
                    source.constitution.atom_index(AtomRef(lysine_id, "NZ")),
                    order=order,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=build_default_component_library(),
    )
    assert resolution.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
    assert "external covalent relationship" in resolution.failure_reason
    for transform in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transform(
            source,
            build_default_component_library(),
            target_residue_ids=frozenset((ligand_id,)),
        )
        assert result.structure is source and not result.repairs
        assert "external covalent relationship" in result.issues[0].message


def test_unsupported_mapped_evidence_uses_relationship_diagnostic():
    source = _source(
        "XXX", BondRelationshipType.COVALENT, BondProvenance.SOURCE_EXPLICIT
    )
    evidence = RetainedNonPolymerChemistryEvidence(
        residue_id=ResidueId("L", 1),
        smiles="CO",
        heavy_atom_names=("C1", "O1"),
    )
    with pytest.raises(ValueError, match="external covalent relationship"):
        retained_non_polymer_evidence_external_bonds(evidence, source)


def test_absent_residue_has_no_external_relationships():
    source = _source("SAN", None, BondProvenance.SOURCE_EXPLICIT)
    assert RetainedCovalentContext(source, ResidueId("missing", 99)).attachments == ()
