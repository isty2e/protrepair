"""Focused tests for placement-time rotatable-hydrogen scoring."""

from dataclasses import replace

import numpy as np
import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    completion_payload,
)

from protrepair.chemistry import RotatableHydrogenKind, UnknownElementRadiusError
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import GeometryPlacementError, InternalCoordinateFrame, Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen.domain import (
    HydrogenCompletionEnvironment,
)
from protrepair.transformer.completion.hydrogen.geometry import (
    CoordinateLike,
    scale_bond,
)
from protrepair.transformer.completion.hydrogen.rotatable import (
    ROTATABLE_HYDROGEN_SPECS,
    RotatableHydrogenEnvironment,
    RotatableHydrogenSearch,
    build_rotatable_hydrogen_environments,
    rotatable_hydrogen_placement_spec,
)
from protrepair.transformer.completion.hydrogen.scoring import (
    hydrogen_steric_penalty_against_site,
    max_rotatable_hydrogen_interaction_horizon_angstrom,
    max_rotatable_hydrogen_steric_cutoff_angstrom,
    rotatable_hydrogen_vdw_radius_angstrom,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def test_optimize_rotatable_hydrogen_penalizes_same_residue_two_hop_clashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-hop same-residue heavy contacts should influence rotatable-H scoring."""

    candidate_hydrogens = (Vec3(0.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                7,
                (
                    atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.7, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(8.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(9.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 2.0, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 1.0, 0.0)),
                ),
            ),
        ),
        templates=(build_standard_component_library().require("SER"),),
    )

    result = search_template().optimized_coordinate(environments[0])

    assert result == Vec3(3.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_keeps_probable_hydrogen_bond_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probable donor-H...acceptor contacts should not be penalized as clashes."""

    candidate_hydrogens = (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    result = search_template().optimized_coordinate(
        RotatableHydrogenEnvironment(
            residue_id=ResidueId("A", 11),
            atom_x=(2.8, 0.0),
            atom_y=(0.0, 2.7),
            atom_z=(0.0, 0.0),
            elements=("O", "C"),
            charges=(0.0, 0.0),
            sigmas_nm=(0.0, 0.0),
            epsilons_kj_mol=(0.0, 0.0),
        )
    )

    assert result == Vec3(1.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_selects_least_bad_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every orientation clashes, the least-overlapping candidate should win."""

    candidate_hydrogens = (
        Vec3(0.8, 0.0, 0.0),
        Vec3(1.2, 0.0, 0.0),
        Vec3(1.8, 0.0, 0.0),
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    result = search_template().optimized_coordinate(
        RotatableHydrogenEnvironment(
            residue_id=ResidueId("A", 19),
            atom_x=(0.0,),
            atom_y=(0.0,),
            atom_z=(0.0,),
            elements=("C",),
            charges=(0.0,),
            sigmas_nm=(0.0,),
            epsilons_kj_mol=(0.0,),
        )
    )

    assert result == Vec3(1.8, 0.0, 0.0)


def test_rotatable_hydrogen_scan_covers_six_evenly_spaced_torsions() -> None:
    """The six-step scan should cover one complete donor-bond rotation."""

    search = nondegenerate_search(bond_length=1.0)

    candidates = search.candidate_positions()
    torsion_offsets = sorted(
        (
            InternalCoordinateFrame.torsion(
                search.outer_anchor,
                search.inner_anchor,
                search.donor,
                candidate,
            )
            - search.dihedral
        )
        % 360.0
        for candidate in candidates
    )

    assert len(candidates) == 6
    assert (
        len(
            {
                tuple(round(float(coordinate), 8) for coordinate in candidate)
                for candidate in candidates
            }
        )
        == 6
    )
    assert torsion_offsets == pytest.approx((0.0, 60.0, 120.0, 180.0, 240.0, 300.0))
    assert tuple(
        three_point_angle_degrees(
            search.inner_anchor,
            search.donor,
            candidate,
        )
        for candidate in candidates
    ) == pytest.approx((search.bond_angle_degrees,) * 6)


def test_rotatable_hydrogen_scan_keeps_slots_independent_after_failed_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed torsion must not shift the angular identity of later slots."""

    attempted_torsions: list[float] = []

    def place_candidate(
        _frame: InternalCoordinateFrame,
        *,
        bond_length: float,
        bond_angle_degrees: float,
        dihedral_degrees: float,
    ) -> Vec3:
        attempted_torsions.append(dihedral_degrees)
        if dihedral_degrees == 210.0:
            raise GeometryPlacementError("synthetic failed torsion")
        return Vec3(dihedral_degrees, bond_angle_degrees, bond_length)

    monkeypatch.setattr(InternalCoordinateFrame, "place", place_candidate)
    search = nondegenerate_search(bond_length=1.0)

    candidates = search.candidate_positions()

    assert attempted_torsions == pytest.approx((90, 150, 210, 270, 330, 390))
    assert len(candidates) == 5


def test_rotatable_hydrogen_candidates_preserve_chemical_bond_length() -> None:
    """Candidate reprojection must not replace a thiol-like bond with O-H length."""

    search = nondegenerate_search(bond_length=1.34)

    for candidate in search.candidate_positions():
        donor_hydrogen_distance = InternalCoordinateFrame.distance(
            search.donor,
            candidate,
        )
        assert donor_hydrogen_distance == pytest.approx(1.34)


@pytest.mark.parametrize("invalid_bond_length", (0.0, -1.0, float("inf"), float("nan")))
def test_rotatable_hydrogen_models_reject_invalid_bond_length(
    invalid_bond_length: float,
) -> None:
    """Placement metadata and bound searches should share one length invariant."""

    cys_template = build_standard_component_library().require("CYS")
    placement_spec = rotatable_hydrogen_placement_spec(cys_template.hydrogen_semantics)
    assert placement_spec is not None

    with pytest.raises(ValueError, match="finite and positive"):
        replace(placement_spec, bond_length=invalid_bond_length)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(
            nondegenerate_search(bond_length=1.0),
            bond_length=invalid_bond_length,
        )


@pytest.mark.parametrize(
    "invalid_bond_angle_degrees",
    (0.0, -1.0, 180.0, 181.0, float("inf"), float("nan")),
)
def test_rotatable_hydrogen_search_rejects_invalid_bond_angle(
    invalid_bond_angle_degrees: float,
) -> None:
    """The donor-H bond angle must be finite and geometrically nondegenerate."""

    with pytest.raises(ValueError, match="between 0 and 180 degrees"):
        replace(
            nondegenerate_search(bond_length=1.0),
            bond_angle_degrees=invalid_bond_angle_degrees,
        )
    with pytest.raises(ValueError, match="between 0 and 180 degrees"):
        replace(
            ROTATABLE_HYDROGEN_SPECS[RotatableHydrogenKind.CYS],
            bond_angle_degrees=invalid_bond_angle_degrees,
        )


def test_rotatable_hydrogen_specs_use_amber_equilibrium_bond_angles() -> None:
    """Each donor chemistry should retain its AMBER protein equilibrium angle."""

    assert {
        kind: spec.bond_angle_degrees for kind, spec in ROTATABLE_HYDROGEN_SPECS.items()
    } == {
        RotatableHydrogenKind.CYS: 96.0,
        RotatableHydrogenKind.SER: 108.5,
        RotatableHydrogenKind.THR: 108.5,
        RotatableHydrogenKind.TYR: 113.0,
    }


def test_rotatable_hydrogen_environment_reports_unknown_radii_once() -> None:
    """Rotatable-H scoring should fail before candidate loops for unknown elements."""

    with pytest.raises(UnknownElementRadiusError) as error_info:
        RotatableHydrogenEnvironment(
            residue_id=ResidueId("A", 11),
            atom_x=(0.0, 0.0),
            atom_y=(0.0, 1.0),
            atom_z=(0.0, 0.0),
            elements=("XX", "C1"),
            charges=(0.0, 0.0),
            sigmas_nm=(0.0, 0.0),
            epsilons_kj_mol=(0.0, 0.0),
        )

    error_message = str(error_info.value)
    assert "rotatable hydrogen scoring environment" in error_message
    assert "XX" in error_message
    assert "C1" in error_message


def test_rotatable_hydrogen_environment_rejects_misaligned_packed_fields() -> None:
    """Packed interaction columns must share one atom-site cardinality."""

    with pytest.raises(ValueError, match="packed fields must have equal lengths"):
        RotatableHydrogenEnvironment(
            residue_id=ResidueId("A", 11),
            atom_x=(0.0,),
            atom_y=(),
            atom_z=(0.0,),
            elements=("C",),
            charges=(0.0,),
            sigmas_nm=(0.0,),
            epsilons_kj_mol=(0.0,),
        )


def test_rotatable_hydrogen_environments_preserve_exact_residue_identity() -> None:
    """Insertion-code siblings must not share one rotatable-H environment identity."""

    library = build_standard_component_library()
    residues = (
        build_residue(
            "SER",
            "A",
            11,
            (atom_payload("OG", "O", Vec3(0.0, 0.0, 0.0)),),
            insertion_code="A",
        ),
        build_residue(
            "SER",
            "A",
            11,
            (atom_payload("OG", "O", Vec3(3.0, 0.0, 0.0)),),
            insertion_code="B",
        ),
    )

    environments = build_rotatable_hydrogen_environments(
        residues=residues,
        templates=(library.require("SER"), library.require("SER")),
    )

    assert tuple(environment.residue_id for environment in environments) == (
        ResidueId("A", 11, "A"),
        ResidueId("A", 11, "B"),
    )


def test_hydrogen_completion_environment_rejects_duplicate_residue_identity() -> None:
    """Chain-local environment slots must retain one-to-one residue identity."""

    environment = minimal_environment(11, marker=1.0)

    with pytest.raises(ValueError, match="must not repeat residue identities"):
        HydrogenCompletionEnvironment(
            rotatable_environments=(environment, environment),
            sg_positions=(),
        )


def test_rotatable_hydrogen_coincident_site_has_infinite_potential() -> None:
    """A candidate on a heavy atom must be rejected without division by zero."""

    environment = RotatableHydrogenEnvironment(
        residue_id=ResidueId("A", 11),
        atom_x=(0.0,),
        atom_y=(0.0,),
        atom_z=(0.0,),
        elements=("C",),
        charges=(0.0,),
        sigmas_nm=(0.0,),
        epsilons_kj_mol=(0.0,),
    )

    assert search_template().potential_energy(Vec3(0.0, 0.0, 0.0), environment) == (
        float("inf")
    )


def test_rotatable_hydrogen_steric_penalty_uses_radius_derived_cutoff() -> None:
    """Broad-radius sites beyond the legacy fixed cutoff should still be scored."""

    penalty = hydrogen_steric_penalty_against_site(
        hydrogen_x=0.0,
        hydrogen_y=0.0,
        hydrogen_z=0.0,
        site_x=3.10,
        site_y=0.0,
        site_z=0.0,
        site_element="CS",
        hydrogen_vdw_radius=rotatable_hydrogen_vdw_radius_angstrom("H"),
        site_vdw_radius=rotatable_hydrogen_vdw_radius_angstrom("CS"),
        donor_x=-1.0,
        donor_y=0.0,
        donor_z=0.0,
        donor_element="O",
        allow_hydrogen_bond=True,
    )

    assert penalty > 0.0


def test_rotatable_hydrogen_bond_exemption_requires_plausible_angle() -> None:
    """Distance alone must not exempt an acute donor-H-acceptor contact."""

    common = {
        "hydrogen_x": 1.0,
        "hydrogen_y": 0.0,
        "hydrogen_z": 0.0,
        "site_element": "O",
        "hydrogen_vdw_radius": rotatable_hydrogen_vdw_radius_angstrom("H"),
        "site_vdw_radius": rotatable_hydrogen_vdw_radius_angstrom("O"),
        "donor_x": 0.0,
        "donor_y": 0.0,
        "donor_z": 0.0,
        "donor_element": "O",
        "allow_hydrogen_bond": True,
    }

    linear_penalty = hydrogen_steric_penalty_against_site(
        **common,
        site_x=2.8,
        site_y=0.0,
        site_z=0.0,
    )
    acute_penalty = hydrogen_steric_penalty_against_site(
        **common,
        site_x=0.1,
        site_y=1.55,
        site_z=0.0,
    )

    assert linear_penalty == 0.0
    assert acute_penalty > 0.0


def test_rotatable_hydrogen_bond_exemption_rejects_degenerate_donor_vector() -> None:
    """A donor coincident with H has no defensible hydrogen-bond angle."""

    penalty = hydrogen_steric_penalty_against_site(
        hydrogen_x=0.0,
        hydrogen_y=0.0,
        hydrogen_z=0.0,
        site_x=1.8,
        site_y=0.0,
        site_z=0.0,
        site_element="O",
        hydrogen_vdw_radius=rotatable_hydrogen_vdw_radius_angstrom("H"),
        site_vdw_radius=rotatable_hydrogen_vdw_radius_angstrom("O"),
        donor_x=0.0,
        donor_y=0.0,
        donor_z=0.0,
        donor_element="O",
        allow_hydrogen_bond=True,
    )

    assert penalty > 0.0


def test_optimize_rotatable_hydrogen_preserves_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best score must return the candidate that produced it."""

    candidate_hydrogens = (Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        lambda self, hydrogen, environment: (
            1.0 if Vec3.coerce(hydrogen).x == 2.0 else 5.0
        ),
    )

    result = search_template().optimized_coordinate(minimal_environment(19, marker=1.0))

    assert result == Vec3(2.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_keeps_first_candidate_on_score_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tie handling should remain deterministic in candidate scan order."""

    candidate_hydrogens = (Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        lambda self, hydrogen, environment: 1.0,
    )

    result = search_template().optimized_coordinate(minimal_environment(19, marker=1.0))

    assert result == Vec3(1.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_does_not_score_empty_candidate_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty candidate scan should fall back without calling the scorer."""

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: (),
    )

    def score_empty_candidate_scan(self, hydrogen, environment) -> float:
        pytest.fail("empty candidate scans must not call candidate_score")

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        score_empty_candidate_scan,
    )

    result = search_template().optimized_coordinate(minimal_environment(1, marker=1.0))

    assert result == Vec3(9.0, 9.0, 9.0)


def test_rotatable_hydrogen_search_skips_undefined_candidate_frames() -> None:
    """Degenerate anchor frames should fall back instead of leaking math errors."""

    search = RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[1.0, 0.0, 0.0],
        donor=[1.0, 0.0, 0.0],
        hydrogen=[9.0, 9.0, 9.0],
        bond_length=1.0,
        bond_angle_degrees=109.5,
        dihedral=0.0,
        partial_charge=0.0,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )

    assert search.candidate_positions() == ()
    assert search.optimized_coordinate(
        RotatableHydrogenEnvironment(
            residue_id=ResidueId("A", 7),
            atom_x=(0.0,),
            atom_y=(0.0,),
            atom_z=(0.0,),
            elements=("C",),
            charges=(0.0,),
            sigmas_nm=(0.0,),
            epsilons_kj_mol=(0.0,),
        )
    ) == Vec3(9.0, 9.0, 9.0)


def test_hydrogen_scale_bond_rejects_degenerate_candidate() -> None:
    """Hydrogen-specific bond scaling should not divide by a zero norm."""

    origin = np.array((1.0, 1.0, 1.0), dtype=np.float64)

    with pytest.raises(GeometryPlacementError, match="degenerate bond vector"):
        scale_bond(origin, origin.copy(), 1.0)


def test_build_rotatable_hydrogen_environments_pack_local_steric_sites() -> None:
    """Rotatable residues should carry donor-centered same-residue steric sites."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                    atom_payload("OG", "O", Vec3(1.0, 2.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (
                    atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                ),
            ),
        ),
        templates=(
            library.require("SER"),
            library.require("GLY"),
        ),
    )

    serine_sites = {
        local_site.atom_name: local_site for local_site in environments[0].local_sites
    }

    assert serine_sites["OG"].bond_hops_from_donor == 0
    assert serine_sites["CB"].bond_hops_from_donor == 1
    assert serine_sites["CA"].bond_hops_from_donor == 2
    assert serine_sites["N"].bond_hops_from_donor == 3
    assert environments[1].local_sites == ()


def test_build_rotatable_hydrogen_environments_filters_far_chain_sites() -> None:
    """Rotatable-H environments should not pack heavy sites outside score range."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 1.0, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 2.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (atom_payload("N", "N", Vec3(0.0, 4.8, 0.0)),),
            ),
            build_residue(
                "GLY",
                "A",
                3,
                (atom_payload("N", "N", Vec3(0.0, 20.0, 0.0)),),
            ),
        ),
        templates=(
            library.require("SER"),
            library.require("GLY"),
            library.require("GLY"),
        ),
    )

    assert environments[0].atom_y == (4.8,)
    assert environments[1].atom_y == ()
    assert environments[2].atom_y == ()


def test_rotatable_hydrogen_environment_covers_potential_energy_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Potential scoring should include sites beyond the steric-only donor radius."""

    candidate_hydrogens = (Vec3(-0.96, 0.0, 0.0), Vec3(0.96, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("OG", "O", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("HG", "H", Vec3(0.96, 0.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (atom_payload("N", "N", Vec3(3.30, 0.0, 0.0)),),
            ),
        ),
        templates=(library.require("SER"), library.require("GLY")),
    )
    search = RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[0.0, 0.0, 0.0],
        donor=[0.0, 0.0, 0.0],
        hydrogen=[0.96, 0.0, 0.0],
        bond_length=0.96,
        bond_angle_degrees=109.5,
        dihedral=0.0,
        partial_charge=0.41,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )

    max_interaction_horizon = max_rotatable_hydrogen_interaction_horizon_angstrom(
        ("N",)
    )
    steric_only_donor_radius = (
        max_rotatable_hydrogen_steric_cutoff_angstrom(("N",)) + 0.96
    )
    assert max_interaction_horizon == 2.5
    assert 3.30 > steric_only_donor_radius
    assert 3.30 <= max_interaction_horizon + 0.96
    assert environments[0].atom_x == (3.30,)
    assert search.potential_energy(candidate_hydrogens[0], environments[0]) == 0.0
    assert search.potential_energy(candidate_hydrogens[1], environments[0]) < 0.0
    assert search.optimized_coordinate(environments[0]) == Vec3(0.96, 0.0, 0.0)


def test_rotatable_hydrogen_environment_radius_covers_cys_bond_length() -> None:
    """CYS packing should account for the longer chemical S-H bond length."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "CYS",
                "A",
                1,
                (
                    atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("HG", "H", Vec3(0.96, 0.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (atom_payload("N", "N", Vec3(3.70, 0.0, 0.0)),),
            ),
        ),
        templates=(library.require("CYS"), library.require("GLY")),
    )

    assert environments[0].atom_x == (3.70,)


def test_rotatable_hydrogen_environment_defers_unknown_radii_for_noop_path() -> None:
    """Unsupported sites should not fail when no rotatable donor needs scoring."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "GLY",
                "A",
                1,
                (atom_payload("X1", "XX", Vec3(0.0, 0.0, 0.0)),),
            ),
        ),
        templates=(library.require("GLY"),),
    )

    assert environments[0].elements == ()


def test_rotatable_hydrogen_environment_radius_covers_broad_radius_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment packing should include sites that can affect steric scoring."""

    candidate_hydrogens = (Vec3(3.0, 0.0, 0.0), Vec3(-3.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )
    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("N", "N", Vec3(0.0, -2.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.0, -1.0, 0.0)),
                    atom_payload("C", "C", Vec3(0.0, 8.0, 0.0)),
                    atom_payload("O", "O", Vec3(0.0, 9.0, 0.0)),
                    atom_payload("CB", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("OG", "O", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("HG", "H", Vec3(3.0, 0.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (atom_payload("CS1", "CS", Vec3(6.10, 0.0, 0.0)),),
            ),
        ),
        templates=(library.require("SER"), library.require("GLY")),
    )

    result = search_template().optimized_coordinate(environments[0])

    assert environments[0].atom_x == (6.10,)
    assert result == Vec3(-3.0, 0.0, 0.0)


def search_template() -> RotatableHydrogenSearch:
    """Return a minimal rotatable-hydrogen search input for direct scoring tests."""

    return RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[0.0, 0.0, 0.0],
        donor=[0.0, 0.0, 0.0],
        hydrogen=[9.0, 9.0, 9.0],
        bond_length=1.0,
        bond_angle_degrees=109.5,
        dihedral=0.0,
        partial_charge=0.0,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )


def nondegenerate_search(
    *,
    bond_length: float,
) -> RotatableHydrogenSearch:
    """Return a nondegenerate donor frame for direct candidate-scan tests."""

    return RotatableHydrogenSearch(
        outer_anchor=Vec3(0.0, 0.0, 0.0),
        inner_anchor=Vec3(1.0, 0.0, 0.0),
        donor=Vec3(1.0, 1.0, 0.0),
        hydrogen=Vec3(1.0, 1.0, 1.0),
        bond_length=bond_length,
        bond_angle_degrees=90.0,
        dihedral=90.0,
        partial_charge=0.1,
        sigma=0.1,
        epsilon=0.1,
    )


def minimal_environment(
    seq_num: int,
    *,
    marker: float,
) -> RotatableHydrogenEnvironment:
    """Return one marker-bearing environment for scoring-focused tests."""

    return RotatableHydrogenEnvironment(
        residue_id=ResidueId("A", seq_num),
        atom_x=(0.0,),
        atom_y=(marker,),
        atom_z=(0.0,),
        elements=("C",),
        charges=(0.0,),
        sigmas_nm=(0.0,),
        epsilons_kj_mol=(0.0,),
    )


def three_point_angle_degrees(
    point_1: CoordinateLike,
    center: CoordinateLike,
    point_3: CoordinateLike,
) -> float:
    """Return a test-only angle around one nondegenerate center point."""

    center_vector = Vec3.coerce(center).to_array()
    vector_1 = Vec3.coerce(point_1).to_array() - center_vector
    vector_3 = Vec3.coerce(point_3).to_array() - center_vector
    cosine = float(np.dot(vector_1, vector_3)) / (
        float(np.linalg.norm(vector_1)) * float(np.linalg.norm(vector_3))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
    *,
    insertion_code: str | None = None,
) -> CompletionResiduePayload:
    """Return one canonical completion payload for hydrogen-scoring tests."""

    return completion_payload(
        component_id=component_id,
        residue_id=ResidueId(
            chain_id=chain_id,
            seq_num=seq_num,
            insertion_code=insertion_code,
        ),
        atoms=atoms,
    )
