"""Rotatable-hydrogen orientation primitives and packed environments."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.chemistry import (
    HydrogenSemantics,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
    RotatableHydrogenKind,
)
from protrepair.geometry import InternalCoordinateFrame, Vec3
from protrepair.transformer.completion.hydrogen.scoring import (
    RotatableHydrogenEnvironment,
    RotatableHydrogenLocalSite,
    RotatableHydrogenSearch,
    max_rotatable_hydrogen_interaction_horizon_angstrom,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

__all__ = [
    "ROTATABLE_HYDROGEN_SPECS",
    "RotatableHydrogenEnvironment",
    "RotatableHydrogenLocalSite",
    "RotatableHydrogenPlacementSpec",
    "RotatableHydrogenSearch",
    "build_rotatable_hydrogen_environments",
    "build_rotatable_hydrogen_search",
    "optimization_residue_number",
    "rotatable_hydrogen_donor_atom_name",
    "rotatable_hydrogen_placement_spec",
]

@dataclass(frozen=True, slots=True)
class _RotatableHydrogenEnvironmentSite:
    """One pre-packed heavy atom available for rotatable-H environment filtering."""

    residue_index: int
    x: float
    y: float
    z: float
    element: str
    charge: float
    sigma_nm: float
    epsilon_kj_mol: float


@dataclass(frozen=True, slots=True)
class RotatableHydrogenPlacementSpec:
    """Execution metadata for one rotatable donor hydrogen family."""

    hydrogen_atom_name: str
    donor_atom_name: str
    inner_anchor_atom_name: str
    outer_anchor_atom_name: str
    donor_element: str
    build_bond_length: float
    reproject_bond_length: float
    partial_charge: float
    sigma: float
    epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hydrogen_atom_name",
            self.hydrogen_atom_name.strip().upper(),
        )
        object.__setattr__(
            self,
            "donor_atom_name",
            self.donor_atom_name.strip().upper(),
        )
        object.__setattr__(
            self,
            "inner_anchor_atom_name",
            self.inner_anchor_atom_name.strip().upper(),
        )
        object.__setattr__(
            self,
            "outer_anchor_atom_name",
            self.outer_anchor_atom_name.strip().upper(),
        )
        object.__setattr__(
            self,
            "donor_element",
            self.donor_element.strip().upper(),
        )


ROTATABLE_HYDROGEN_SPECS: dict[
    RotatableHydrogenKind, RotatableHydrogenPlacementSpec
] = {
    RotatableHydrogenKind.CYS: RotatableHydrogenPlacementSpec(
        hydrogen_atom_name="HG",
        donor_atom_name="SG",
        inner_anchor_atom_name="CB",
        outer_anchor_atom_name="CA",
        donor_element="S",
        build_bond_length=1.34,
        reproject_bond_length=0.96,
        partial_charge=0.19,
        sigma=0.11,
        epsilon=0.07,
    ),
    RotatableHydrogenKind.SER: RotatableHydrogenPlacementSpec(
        hydrogen_atom_name="HG",
        donor_atom_name="OG",
        inner_anchor_atom_name="CB",
        outer_anchor_atom_name="CA",
        donor_element="O",
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
    ),
    RotatableHydrogenKind.THR: RotatableHydrogenPlacementSpec(
        hydrogen_atom_name="HG1",
        donor_atom_name="OG1",
        inner_anchor_atom_name="CB",
        outer_anchor_atom_name="CA",
        donor_element="O",
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
    ),
    RotatableHydrogenKind.TYR: RotatableHydrogenPlacementSpec(
        hydrogen_atom_name="HH",
        donor_atom_name="OH",
        inner_anchor_atom_name="CZ",
        outer_anchor_atom_name="CE2",
        donor_element="O",
        build_bond_length=0.96,
        reproject_bond_length=0.96,
        partial_charge=0.37,
        sigma=0.0,
        epsilon=0.0,
    ),
}


def rotatable_hydrogen_placement_spec(
    hydrogen_semantics: HydrogenSemantics | IdealGeometryHydrogenSemantics | None,
) -> RotatableHydrogenPlacementSpec | None:
    """Return execution metadata for one rotatable hydrogen semantics block."""

    if (
        hydrogen_semantics is None
        or not isinstance(hydrogen_semantics, HydrogenSemantics)
        or hydrogen_semantics.rotatable_kind is None
    ):
        return None

    return ROTATABLE_HYDROGEN_SPECS[hydrogen_semantics.rotatable_kind]


def optimization_residue_number(
    residue: CompletionResiduePayload,
) -> str:
    """Return the legacy ProtRepair residue-number token for rotatable-H refinement."""

    return str(residue.residue_id.seq_num)


def build_rotatable_hydrogen_search(
    residue: CompletionResiduePayload,
    *,
    spec: RotatableHydrogenPlacementSpec,
) -> RotatableHydrogenSearch | None:
    """Return one search object for an already-placed rotatable hydrogen."""

    required_atom_names = (
        spec.outer_anchor_atom_name,
        spec.inner_anchor_atom_name,
        spec.donor_atom_name,
        spec.hydrogen_atom_name,
    )
    if not all(residue.has_atom(atom_name) for atom_name in required_atom_names):
        return None

    residue_geometry = residue.residue_geometry
    outer_anchor = residue_geometry.position(spec.outer_anchor_atom_name)
    inner_anchor = residue_geometry.position(spec.inner_anchor_atom_name)
    donor = residue_geometry.position(spec.donor_atom_name)
    hydrogen = residue_geometry.position(spec.hydrogen_atom_name)
    return RotatableHydrogenSearch(
        outer_anchor=list(outer_anchor),
        inner_anchor=list(inner_anchor),
        donor=list(donor),
        hydrogen=list(hydrogen),
        build_bond_length=spec.build_bond_length,
        reproject_bond_length=spec.reproject_bond_length,
        dihedral=InternalCoordinateFrame.torsion(
            outer_anchor,
            inner_anchor,
            donor,
            hydrogen,
        ),
        partial_charge=spec.partial_charge,
        sigma=spec.sigma,
        epsilon=spec.epsilon,
        donor_element=spec.donor_element,
    )


def build_rotatable_hydrogen_environments(
    *,
    residues: tuple[CompletionResiduePayload, ...],
    residue_numbers: list[str],
    templates: Sequence[ResidueTemplate | None],
) -> tuple[RotatableHydrogenEnvironment, ...]:
    """Pack per-residue interaction environments for rotatable hydrogen search."""

    placement_specs = tuple(
        None
        if template is None
        else rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
        for template in templates
    )
    eligible_donor_by_residue = tuple(
        placement_spec is not None
        and residue.has_atom(placement_spec.donor_atom_name)
        for residue, placement_spec in zip(residues, placement_specs, strict=True)
    )
    environment_sites: tuple[_RotatableHydrogenEnvironmentSite, ...] = ()
    max_hydrogen_site_interaction_horizon_angstrom: float | None = None
    if any(eligible_donor_by_residue):
        environment_sites = _rotatable_hydrogen_environment_sites(
            residues=residues,
            templates=templates,
        )
        max_hydrogen_site_interaction_horizon_angstrom = (
            max_rotatable_hydrogen_interaction_horizon_angstrom(
                site.element for site in environment_sites
            )
        )

    environments: list[RotatableHydrogenEnvironment] = []
    for residue_index, (
        residue,
        residue_number,
        template,
        placement_spec,
        eligible_donor,
    ) in enumerate(
        zip(
            residues,
            residue_numbers,
            templates,
            placement_specs,
            eligible_donor_by_residue,
            strict=True,
        )
    ):
        if not eligible_donor:
            environments.append(
                RotatableHydrogenEnvironment(
                    residue_number=residue_number,
                    atom_x=(),
                    atom_y=(),
                    atom_z=(),
                    elements=(),
                    charges=(),
                    sigmas_nm=(),
                    epsilons_kj_mol=(),
                )
            )
            continue

        assert placement_spec is not None
        assert max_hydrogen_site_interaction_horizon_angstrom is not None
        atom_x: list[float] = []
        atom_y: list[float] = []
        atom_z: list[float] = []
        elements: list[str] = []
        charges: list[float] = []
        sigmas_nm: list[float] = []
        epsilons_kj_mol: list[float] = []
        donor_position = residue.residue_geometry.position(
            placement_spec.donor_atom_name
        )
        interaction_radius_sq = _rotatable_hydrogen_environment_radius_sq(
            placement_spec,
            max_hydrogen_site_interaction_horizon_angstrom=(
                max_hydrogen_site_interaction_horizon_angstrom
            ),
        )

        for site in environment_sites:
            if site.residue_index == residue_index:
                continue
            if not _site_within_donor_radius(
                site,
                donor_position=donor_position,
                radius_sq=interaction_radius_sq,
            ):
                continue

            atom_x.append(site.x)
            atom_y.append(site.y)
            atom_z.append(site.z)
            elements.append(site.element)
            charges.append(site.charge)
            sigmas_nm.append(site.sigma_nm)
            epsilons_kj_mol.append(site.epsilon_kj_mol)

        environments.append(
            RotatableHydrogenEnvironment(
                residue_number=residue_number,
                atom_x=tuple(atom_x),
                atom_y=tuple(atom_y),
                atom_z=tuple(atom_z),
                elements=tuple(elements),
                charges=tuple(charges),
                sigmas_nm=tuple(sigmas_nm),
                epsilons_kj_mol=tuple(epsilons_kj_mol),
                local_sites=_rotatable_hydrogen_local_sites(
                    residue=residue,
                    template=template,
                ),
            )
        )

    return tuple(environments)


def _rotatable_hydrogen_environment_sites(
    *,
    residues: tuple[CompletionResiduePayload, ...],
    templates: Sequence[ResidueTemplate | None],
) -> tuple[_RotatableHydrogenEnvironmentSite, ...]:
    """Return reusable chain-local heavy sites for rotatable-H scoring."""

    sites: list[_RotatableHydrogenEnvironmentSite] = []
    for residue_index, (residue, template) in enumerate(
        zip(residues, templates, strict=True)
    ):
        residue_geometry = residue.residue_geometry
        parameters_by_atom = (
            {} if template is None else template.forcefield_parameters
        )
        for atom_site in residue.atom_sites:
            if atom_site.element == "H":
                continue

            atom_position = residue_geometry.position(atom_site.name)
            atom_parameters = parameters_by_atom.get(atom_site.name)
            sites.append(
                _RotatableHydrogenEnvironmentSite(
                    residue_index=residue_index,
                    x=atom_position.x,
                    y=atom_position.y,
                    z=atom_position.z,
                    element=atom_site.element,
                    charge=0.0 if atom_parameters is None else atom_parameters.charge,
                    sigma_nm=(
                        0.0 if atom_parameters is None else atom_parameters.sigma_nm
                    ),
                    epsilon_kj_mol=(
                        0.0
                        if atom_parameters is None
                        else atom_parameters.epsilon_kj_mol
                    ),
                )
            )

    return tuple(sites)


def _rotatable_hydrogen_environment_radius_sq(
    placement_spec: RotatableHydrogenPlacementSpec,
    *,
    max_hydrogen_site_interaction_horizon_angstrom: float,
) -> float:
    """Return donor-centered radius squared that safely covers scored candidates."""

    max_candidate_displacement_angstrom = max(
        placement_spec.build_bond_length,
        placement_spec.reproject_bond_length,
    )
    environment_radius_angstrom = (
        max_hydrogen_site_interaction_horizon_angstrom
        + max_candidate_displacement_angstrom
    )
    return environment_radius_angstrom * environment_radius_angstrom


def _site_within_donor_radius(
    site: _RotatableHydrogenEnvironmentSite,
    *,
    donor_position: Vec3,
    radius_sq: float,
) -> bool:
    """Return whether one heavy site can interact with a donor-centered H scan."""

    delta_x = donor_position.x - site.x
    delta_y = donor_position.y - site.y
    delta_z = donor_position.z - site.z
    return (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z) <= radius_sq


def _rotatable_hydrogen_local_sites(
    *,
    residue: CompletionResiduePayload,
    template: ResidueTemplate | None,
) -> tuple[RotatableHydrogenLocalSite, ...]:
    """Return same-residue heavy sites relevant to rotatable-H scoring."""

    donor_atom_name = rotatable_hydrogen_donor_atom_name(template)
    if donor_atom_name is None:
        return ()

    assert template is not None
    residue_geometry = residue.residue_geometry
    local_sites: list[RotatableHydrogenLocalSite] = []
    for atom_site in residue.atom_sites:
        if atom_site.element == "H":
            continue

        atom_position = residue_geometry.position(atom_site.name)
        local_sites.append(
            RotatableHydrogenLocalSite(
                atom_name=atom_site.name,
                element=atom_site.element,
                x=atom_position.x,
                y=atom_position.y,
                z=atom_position.z,
                bond_hops_from_donor=template.bond_hop_distance(
                    donor_atom_name,
                    atom_site.name,
                ),
            )
        )

    return tuple(local_sites)


def rotatable_hydrogen_donor_atom_name(
    template: ResidueTemplate | None,
) -> str | None:
    """Return the donor heavy atom used by a rotatable-H residue template."""

    if template is None:
        return None

    spec = rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
    if spec is None:
        return None

    return spec.donor_atom_name
