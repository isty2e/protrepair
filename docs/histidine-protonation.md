# Histidine Protonation

ProtRepair does not infer histidine protonation automatically from pKa,
hydrogen-bond networks, or solvent exposure. Additional histidine delta
protonation is disabled unless the caller requests a method explicitly.

## PRAS Ratio Method

`PrasRatioHistidineProtonationRequest` exposes the deterministic behavior
inherited from PRAS. For each chain, ProtRepair:

1. collects `HIS` residues in chain order
2. computes `floor(number_of_histidines * ratio)`
3. assigns delta protonation to that many residues from the start of the list

The default ratio is `0.2`, and accepted ratios are finite values in the closed
interval `[0.0, 1.0]`. The older `protonate_histidines=True` boundary argument
normalizes to the default PRAS-ratio request. It cannot be supplied together
with an explicit `histidine_protonation` request.

The ratio method reflects the legacy PRAS assumption of roughly pH 7. It is a
deterministic selection rule, not an environment-aware protonation model.

## Assignment Contract

The request is resolved into explicit
`HistidineDeltaProtonationAssignment` values before hydrogen directives are
built. A future explicit-residue, tautomer, pKa-backed, or environment-aware
method should produce the same assignment contract rather than broadening the
meaning of the PRAS-ratio request.
