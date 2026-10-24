# FASPR Runtime Policy

ProtRepair treats FASPR as a heavy-atom side-chain packing backend. FASPR does
not own general hydrogen completion, retained-ligand chemistry, or final
workflow planning.

Target sequences use FASPR's 20 standard amino-acid codes. Unsupported codes,
including `X` and `B`, raise `PackingError` before FASPR is launched.

## Coordinate And Chemical Authority

The packing plan decides which sidechains may move. Backbone atoms, OXT,
fixed residues and retained ligands keep their exact canonical coordinates.
The adapter compares protected native coordinates with the serialized input,
allowing only FASPR's float32 conversion and three-decimal output rounding.
Actual movement outside the plan is a packing error, not a change to adopt.
An explicit `mutable_residue_ids` set also constrains full-structure packing.

Fixed residues must have complete standard heavy-atom scaffolds before packing;
OXT is optional. FASPR otherwise unfreezes and repacks them. ProtRepair rejects
such requests before launching FASPR, naming the residue and missing atoms.
Complete its heavy atoms first, or include it in `mutable_residue_ids` and remove
any explicit freeze.

`reference_sidechain_packing` runs on the input before workflow heavy-atom
completion, so the same check can reject a workflow request. The error lists
all incomplete fixed residues rather than requiring one retry per residue.

For unchanged components, packing preserves current atom charges, occupancy,
B factors and altloc metadata. Blank charge fields in FASPR output do not mean
neutral or unknown chemistry. Original input observations remain separate;
packing does not replay them over an applied microstate or explicit override.

FASPR input uses an internal calculation PDB, not the public file writer's
full-fidelity policy. Selected altloc labels are blank in that temporary input
so FASPR does not skip those atoms; canonical labels are preserved in the result.
Large B factors do not block packing: the merge restores
canonical scalar metadata rather than adopting the native file's values.

Every request uses temporary calculation labels rather than the original chain
and residue identifiers. The adapter restores the original identifiers and
chain partitions before merging coordinates. Long chain names, insertion codes
and residue numbers outside PDB's range therefore do not prevent packing.
The calculation namespace holds 692,937 residues; larger requests are rejected
before temporary files or a subprocess are created. Output with missing,
duplicate, reordered or foreign residue records is rejected rather than partly
adopted. Native error locations are translated only when they identify a unique
source residue; other excerpts are marked as native FASPR output.

Requested component mutations discard the old residue's H, charges and saved
microstate choices. New sidechain bonds come from the new component template,
not from old bonds between atoms with matching names. Source-declared bonds
within the standard residue scaffold can be replaced by an authorized mutation;
the original declarations remain in the input observation. Inter-residue,
non-covalent and nonstandard explicit sidechain links are rejected rather than
losing linked chemistry. FASPR does not choose the new protonation state.

When the original input record names a different component, ordinary H
completion cannot yet fully prepare the mutated residue: the microstate
resolver cannot reconcile the two identities. Nonpolar template H may be
added, but backbone and polar-site H can remain unresolved with warnings.
The H-inventory goal excludes unresolved sites, so satisfying that goal does
not certify mutation chemistry as complete.

For same-component repacking, FASPR preserves a complete sidechain when one of
its heavy atoms has an inter-residue covalent, disulfide, coordination or untyped
link. Both source-declared and inferred links count. Backbone-only links and
hydrogen bonds do not exclude a sidechain. Completeness means that all expected
standard heavy atoms are present; protection does not depend on a geometry
cutoff or certify that the input geometry is good.

Other selected residues are still packed. Each protected residue is reported by
`packing_preserved_linked_sidechain`, with its residue identifier. This applies
to full, subset and workflow requests alike. If every remaining target is
protected, the packing input is returned unchanged without starting FASPR or
changing its hydrogens. Automatic workflow planning omits these no-work proposals.
Linked mutations remain errors rather than being silently skipped.
A target sequence also cannot mutate a residue explicitly frozen by the caller.

An incomplete linked sidechain can still be rebuilt. The original links remain
in the canonical topology, but FASPR does not constrain their geometry; the
result reports `packing_rebuilt_linked_sidechain` for that residue. In particular,
an existing linked atom does not protect the residue if another expected heavy
atom is missing. Link-aware construction and joint optimization of connected
sidechains are not supported by this packing path. Preservation also does not
guarantee clash-free surroundings: other packed sidechains can approach the
protected atoms.

## Polymer Hydrogen Merge

The FASPR adapter prepares input by stripping polymer hydrogens because FASPR
ignores input side-chain atoms and returns a heavy-atom packing result. The
merge step must therefore decide whether an original polymer hydrogen is still
valid in the packed structure.

Hydrogen preservation is residue-local and evidence-based:

- Preserve original polymer hydrogens only when the residue heavy-atom state is
  equivalent before and after packing.
- Fixed labels alone are not proof: protected movement is checked before the
  merge. A selected residue whose heavy atoms are unchanged may retain hydrogens.
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

## Packaging And Source Builds

Built packages and wheels include the vendored FASPR executable and
`dun2010bbdep.bin` rotamer library. Prefer a built wheel when using the packaged
side-chain packing backend.

Source installs, including direct GitHub installs, build FASPR through
`scikit-build-core` and CMake. They require CMake 3.18 or newer and a working
C++ compiler toolchain. Direct source-tree imports are not guaranteed to have
compiled FASPR assets available; install the package or wheel first.

Transformer-layer development may use a separately built binary by supplying
an explicit FASPR `executable_path`. The executable's directory must also
contain `dun2010bbdep.bin`.
