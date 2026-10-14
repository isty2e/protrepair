# Histidine Protonation

ProtRepair preserves supported source histidine chemistry. When source evidence
does not distinguish the two neutral tautomers, preparation uses local polar
contacts to prefer delta or epsilon HIS; a tie falls back to epsilon. This does
not predict pKa, solvent exposure, or a coupled hydrogen-bond network. Additional
cationic histidines require an explicit request.

## Neutral Tautomer Selection

Only source-compatible neutral candidates enter the contact comparison. The
comparison screens gross crowding at the prospective ring H positions, then
checks local donor/acceptor contacts. It uses resolved chemical roles and existing
canonically attached N/O-H donors; ambiguous neighboring HIS and untyped
chemistry supply no positive support.

The hydrogen-bond distance and angle criteria follow
[OpenMM 8.4's Modeller](https://github.com/openmm/openmm/blob/8.4.0/wrappers/python/openmm/app/modeller.py).
ProtRepair adds a short-contact screen and conservative chemical eligibility;
it does not run OpenMM or reproduce its complete preparation algorithm.

Applying a selected default records it separately from original observations.
Repeating preparation retains that choice while its chemical context remains
applicable. Before application, a coordinate change re-evaluates the contact
preference; an explicit reset returns to current source-compatible preferences.
Neither a default nor a contact preference can override source evidence.

## PRAS Ratio Method

`PrasRatioHistidineProtonationRequest` exposes the deterministic behavior
inherited from PRAS. For each chain, ProtRepair:

1. collects `HIS` residues in chain order
2. computes `floor(number_of_histidines * ratio)`
3. requests cationic HIS, with one H on each ring nitrogen, for that many residues
   from the start of the list

The default ratio is `0.2`, and accepted ratios are finite values in the closed
interval `[0.0, 1.0]`. The older `protonate_histidines=True` boundary argument
normalizes to the default PRAS-ratio request. It cannot be supplied together
with an explicit `histidine_protonation` request.

The ratio method reflects the legacy PRAS assumption of roughly pH 7. It is a
deterministic selection rule, not an environment-aware protonation model.

## Assignment Contract

The selected residues become complete chemical-site requests before H placement:
ring H counts, net charge +1, and bond orders are resolved together. Both direct
hydrogenation and the workflow apply that graph atomically; neither just appends
HD1 to a neutral ring. Original H identities and superseded source constraints
remain recorded. Unsupported or uninterpretable chemistry is diagnosed rather
than forced into the requested graph.

Repeating preparation without a new request retains an applicable applied
choice. A changed ratio replaces prior generated choices; unselected HIS return
to source-compatible preparation preferences. A ratio of zero therefore clears
generated cation assignments, but does not erase contrary original evidence.
See [coupled microstate resolution](topology-bond-policy.md#coupled-microstate-resolution)
for evidence precedence and readiness checks.

A future pKa-backed or environment-aware method should supply site-level
chemical preferences or explicit constraints to that resolver, not pretend to
be another histidine-ratio rule. Neither method should update H, charge, and
bond order independently.

## Geometry Restraints

HIS ring restraints follow the delta, epsilon and doubly protonated states in
[Malinska et al. (2015), Table 4](https://doi.org/10.1107/S1399004715007816),
using the CSD-derived targets and reported uncertainties.

The diagnostic and FF paths use the same snapshot-bound dictionary selection.
Applying a preference is not enough: current heavy charges and bond orders must
match that supported state, and existing H attachments must be compatible.
Missing H alone does not invalidate an otherwise established heavy chemical
state. Unresolved or unrealized HIS keeps only its state-independent scaffold
targets. Ring bonds use the broad covalent-radius check, and ring angles without
a target use the broad angle window. These fallback checks can flag large
distortions, but do not supply ring restraints to the FF or count as
restraint-backed evidence for automatic intrinsic-geometry repair. Ring geometry
is not used to choose protonation. A caller's custom restraint dictionary remains
authoritative.
