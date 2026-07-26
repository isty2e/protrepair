# Retained-Ligand Chemistry Policy

Retained non-polymers include ligands, cofactors, and other non-polymer
components that remain in the canonical structure. Their coordinates are
preserved unless a selected transformation has explicit authority to change
them.

## Chemistry Resolution

ProtRepair resolves retained-ligand chemistry in this order:

1. explicit chemistry evidence supplied at ingress
2. a bundled component template that can add hydrogens
3. RDKit fallback inference when fallback is allowed

Existing hydrogen names are preferred when they are compatible with the
resolved chemistry. Explicit overrides are validated against the retained
heavy-atom set before workflow execution. Invalid SMILES/evidence mappings or
mismatched overrides raise `ValueError`; they do not silently fall through to
another chemistry source.

## RDKit Fallback

RDKit fallback is enabled by default for retained components without usable
explicit or bundled chemistry. A successful fallback emits the structured
warning `RETAINED_NON_POLYMER_FALLBACK_USED` and records the RDKit backend
version in the resolved chemistry facts.

The fallback is conservative:

- Existing hydrogens are reused only when local anchor geometry and source
  topology are compatible.
- Unsupported hetero multiple-bond inference is rejected.
- Inference that would change retained-ligand stereochemistry is rejected.
- A rejected inference leaves the retained ligand unchanged and reports the
  reason instead of guessing.

Set
`WorkflowTransformRequests(allow_retained_non_polymer_rdkit_fallback=False)`
to require a bundled template or explicit chemistry evidence. If requested
hydrogen completion would otherwise need RDKit, the workflow emits
`RETAINED_NON_POLYMER_FALLBACK_BLOCKED` and leaves that retained component
unchanged.

## Scope Boundaries

The fallback infers chemistry for the heavy atoms already present. It is not a
general heavy-atom completion mechanism for arbitrary nonstandard components.
Single-atom metals and ions are retained as not-applicable chemistry rather
than passed through organic-molecule hydrogenation.
