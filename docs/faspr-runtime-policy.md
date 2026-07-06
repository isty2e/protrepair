# FASPR Runtime Policy

ProtRepair treats FASPR as a heavy-atom side-chain packing backend. FASPR does
not own general hydrogen completion, retained-ligand chemistry, or final
workflow planning.

## Polymer Hydrogen Merge

The FASPR adapter prepares input by stripping polymer hydrogens because FASPR
ignores input side-chain atoms and returns a heavy-atom packing result. The
merge step must therefore decide whether an original polymer hydrogen is still
valid in the packed structure.

Hydrogen preservation is residue-local and evidence-based:

- Preserve original polymer hydrogens only when the residue heavy-atom state is
  equivalent before and after packing.
- Treat fixed-vs-packed labels as hints, not proof. A fixed residue whose heavy
  atoms moved has invalidated hydrogens; a selected residue whose heavy atoms are
  unchanged may retain hydrogens.
- Preserve atom sites, coordinates, and topology bonds together. A preserved
  hydrogen without its anchor bond is invalid canonical topology.
- Drop hydrogen atom sites and hydrogen topology bonds together when heavy atoms
  changed meaningfully.
- Surface dropped hydrogens as a structured issue/state so the workflow planner
  can apply the normal polymer hydrogen completion transformer when requested.

The backend must not silently claim hydrogen-complete output after removing
polymer hydrogens, and it must not place replacement hydrogens itself. Hydrogen
placement remains owned by the hydrogen completion transformer.

## Retained Ligands And Noncanonical Components

FASPR operates on polymer-only input. Retained ligands are restored from the
original structure after packing and are not interpreted by FASPR.

FASPR continues to reject hetero or unsupported polymer components at the
packing boundary. Do not broaden retained-ligand or noncanonical chemistry
semantics inside the FASPR backend.

## Runtime Boundaries

FASPR launch, executable discovery, rotamer-library discovery, and native build
flags are backend concerns. They should fail through typed packing backend
errors or documented release gates rather than leaking raw subprocess, path, or
packaging assumptions to callers.
