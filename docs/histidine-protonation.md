# Histidine Protonation

ProtRepair does not infer histidine protonation automatically from pKa,
hydrogen-bond networks, or solvent exposure. Without contrary source evidence,
preparation chooses neutral epsilon HIS. Additional cationic histidines require
an explicit request.

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

Repeating preparation without a new request retains an applicable explicit
choice. A changed ratio replaces prior generated choices; unselected HIS return
to source-compatible preparation preferences. A ratio of zero therefore clears
generated cation assignments, but does not erase contrary original evidence.
See [coupled microstate resolution](topology-bond-policy.md#coupled-microstate-resolution)
for evidence precedence and readiness checks.

A future pKa-backed or environment-aware method should supply site-level
chemical preferences or explicit constraints to that resolver, not pretend to
be another histidine-ratio rule. Neither method should update H, charge, and
bond order independently.
