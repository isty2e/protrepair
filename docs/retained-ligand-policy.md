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

## Incomplete Templates

A template can place hydrogens on the heavy atoms that are present, even when
part of the component is missing. That partial placement is not evidence that
the original hydrogens are wrong. If it cannot preserve all of them, ProtRepair
leaves the residue unchanged and reports `UNSUPPORTED_HYDROGENATION`, identifying
the missing heavy atoms and unmatched hydrogens.

Failed hydrogenation leaves the residue out of the subsequent H clash cleanup.
Other retained components can still be completed. This does not certify the
unchanged residue as chemically complete or ready for force-field refinement.
Validated chemistry overrides remain a separate route for replacing incompatible
source hydrogens.

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

## Component Dictionaries

Coordinate files and chemical-component dictionaries have different roles.
PDB/mmCIF ingress keeps typed source coordination records, including declarations
to partners removed by normalization. The component-dictionary reader instead
builds covalent templates with integer atom charges. It rejects coordination or
metal-pi flags, unresolved metal flags, and incomplete bond columns rather than silently
converting them into ordinary single bonds.

Explicit unknown atom charges (`?` or `.`) are also rejected by that reader. An
absent charge column uses the dictionary's
[default of zero](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp_atom.charge.html);
an explicit null cell does not. This is dictionary parsing, not a policy for
assigning charges to atoms in a deposited structure.

## Excluded Coordination Partners

If normalization removes a source-declared partner of a retained metal, the
original observation keeps its identity and connection metadata without adding
an atom or bond to the selected structure. Final processing emits a non-blocking
`SOURCE_COORDINATION_PARTNER_OMITTED` warning while that partner remains absent.

This warning does not restore water, infer oxidation or protonation, or certify
a complete metal site. It covers typed source declarations, not coordination
guessed from distances or untyped CONECT records. A missing endpoint, another
model, or an unchosen alternate location is not reported as a policy exclusion.

Output contains only selected atoms and their topology. Declarations to excluded
partners are not written as dangling bonds, so re-reading the output does not
recover that original off-graph evidence. Keep the processing result and original
input when the warning's source history matters.
