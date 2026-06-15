"""Peptide backbone geometry owners for heavy-atom completion."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.geometry import InternalCoordinateFrame, Vec3


@dataclass(frozen=True, slots=True)
class PeptideCarbonylFrame:
    """Behaviorful peptide carbonyl frame for oxygen and OXT placement."""

    nitrogen: Vec3
    alpha_carbon: Vec3
    carbonyl_carbon: Vec3

    def backbone_oxygen(
        self,
        *,
        psi_degrees: float,
        clash_reference: Vec3,
    ) -> Vec3:
        """Return the preferred backbone oxygen coordinate for this carbonyl."""

        if psi_degrees >= 100:
            trial_dihedrals = (-30.0, 140.0, -140.0, 40.0)
        elif 0 < psi_degrees < 100:
            trial_dihedrals = (-140.0, 140.0, -30.0, 40.0)
        else:
            trial_dihedrals = (140.0, -140.0, -30.0, 40.0)

        frame = InternalCoordinateFrame(
            self.nitrogen,
            self.alpha_carbon,
            self.carbonyl_carbon,
        )
        candidate = frame.place(
            bond_length=1.23,
            bond_angle_degrees=120.5,
            dihedral_degrees=trial_dihedrals[0],
        )
        for dihedral in trial_dihedrals:
            candidate = frame.place(
                bond_length=1.23,
                bond_angle_degrees=120.5,
                dihedral_degrees=dihedral,
            )
            if candidate.distance_to(clash_reference) > 2.0:
                return candidate

        return candidate

    def terminal_oxygen(self, backbone_oxygen_coordinate: Vec3) -> Vec3:
        """Return the OXT coordinate implied by this terminal carbonyl frame."""

        dihedral = InternalCoordinateFrame.torsion(
            self.nitrogen,
            self.alpha_carbon,
            self.carbonyl_carbon,
            backbone_oxygen_coordinate,
        )
        return InternalCoordinateFrame(
            self.nitrogen,
            self.alpha_carbon,
            self.carbonyl_carbon,
        ).place(
            bond_length=1.25,
            bond_angle_degrees=122.5,
            dihedral_degrees=180.0 + dihedral,
        )


def backbone_psi_degrees(psi_points: Sequence[Vec3]) -> float:
    """Return the peptide psi torsion used to choose oxygen dihedrals."""

    return InternalCoordinateFrame.torsion(
        psi_points[0],
        psi_points[1],
        psi_points[2],
        psi_points[3],
    )
