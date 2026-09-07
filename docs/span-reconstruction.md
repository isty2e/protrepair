# External Span Reconstruction

ProtRepair can fill a missing polymer span when another structure supplies the
corresponding residues. The donor may be an AlphaFold model, a homologous
experimental structure, or another canonical `ProteinStructure`. It is a
conformation seed, not a second coordinate frame that can be pasted into the
source unchanged.

## Request Contract

Each `ExternalSpanReconstructionSpec` supplies:

- an `AbsentResidueSpanScope` naming the missing source residues and one or two
  present anchors
- an ordered donor residue span of the same length

Both spans must be contiguous within one polymer chain. An internal source span
requires donor context on both sides. A terminal span requires donor context at
its anchored side. Donor components also need canonical component chemistry so
the inserted atoms can enter the structure with a canonical bond graph.

You may submit more than one donor for exactly the same source span. The
workflow keeps those requests as separate candidates. Different source spans
must not overlap or use a residue missing from another request as an anchor;
that would make their meaning depend on execution order.

AlphaFold requests can be built from an observed-to-UniProt alignment with
`build_alphafold_span_reconstruction_specs()`. Internal gaps are selected by
default. Prefix and suffix gaps require an explicit
`ExternalSpanGapSelectionPolicy` because one-anchor placement cannot perform a
closed-loop fit. The source chain must match the chain used to build the
alignment, and the alignment reference, AlphaFold metadata, and donor residue
sequence must agree.

## Placement And Closure

ProtRepair first maps the donor flank into a local frame defined by the source
anchor. It then handles the two span classes differently:

1. For an internal span, deterministic cyclic coordinate descent rotates donor
   backbone phi and psi degrees of freedom until the opposite donor flank
   matches the source anchor triad within the closure endpoint tolerance.
2. For a terminal span, the donor is projected from its one available anchor.
   No second-anchor closure claim is made.

Rotations follow the donor's covalent graph and component chemistry. Only
single, non-aromatic bonds that separate the moving atoms from the fixed stem
can act as independent torsions. This preserves Pro's ring and amide-N geometry,
and leaves an acyclic N-substituent on the nitrogen side of a phi rotation.
The preceding anchor's CA-C bond is not a closure axis: its oxygen stays fixed.

CCD starts with the axis that most reduces endpoint error in a one-rotation
probe. If it does not close, it retries from the same donor seed with reversed
and original axis orders. Each distinct order has the full configured iteration
limit, with at most four orders. No random restart or force-field minimization
is hidden in this step. Endpoint fit guides this search; it is not a score for
clashes or similarity to an unknown native loop.

The operation moves only donor-derived coordinates. Atoms that were already in
the source retain their original coordinates.

## Acceptance And Failure

A low endpoint RMSD is necessary for an internal closure, but it is not enough
to accept the result. Before accepting the aggregate change, ProtRepair also
requires:

- plausible peptide C-N distances, junction angles, and peptide-plane torsions
  at both source boundaries and between every pair of inserted residues
- complete heavy-atom neighborhoods around amide N and planar N-substituents,
  including those on the following source anchor
- supported component chemistry for every inserted residue
- residue-local heavy-atom geometry within the accepted template or restraint
  bounds
- valid standard alpha-carbon chirality and supported side-chain stereochemistry

On success, constitution, geometry, formal-charge payloads, and topology are
updated as one aggregate change. Intra-residue bonds come from component
chemistry, while new peptide C-N bonds are sequence-inferred. Donor hydrogens
are discarded so hydrogen completion remains a separate workflow action.

On failure, the source structure is returned unchanged for that span and the
result carries `SPAN_RECONSTRUCTION_FAILED`. No partial span is committed.

The junction planarity gate accepts both cis and trans families, but rejects a
torsion more than 30 degrees from either plane. This follows the twisted-peptide
boundary used by MolProbity and wwPDB validation.
Amide-N substituents are screened separately with a 30-degree planarity
tolerance; backbone omega alone cannot detect a pyramidalized Pro nitrogen.

Clash assessment remains part of workflow state and candidate ranking rather
than the closure kernel. A geometrically admissible insertion may therefore be
followed by local refinement when the planned target state requires it.

The closure gate does not apply a probability-based Ramachandran score. Request
Ramachandran analysis separately when the reconstructed backbone conformation
is part of the acceptance decision.

## Scientific Scope

AlphaFold confidence is useful evidence for choosing a donor, but it does not
override the source anchors or the geometric acceptance gates. ProtRepair does
not impose a pLDDT cutoff, nor does it copy donor B factors or pLDDT values into
the reconstructed residues. A closed span is not a claim of experimental
correctness; it is a chemically screened structural hypothesis that should be
checked against available density, contacts, and downstream validation.

The implementation deliberately avoids a full molecular-dynamics stack. Its
closure model follows the cyclic-coordinate-descent approach described by:

- Canutescu and Dunbrack, *Cyclic coordinate descent: A robotics algorithm for
  protein loop closure*, [DOI: 10.1110/ps.0242703](https://doi.org/10.1110/ps.0242703)
- Williams et al., *MolProbity: More and better reference data for improved
  all-atom structure validation*, [DOI: 10.1002/pro.3330](https://doi.org/10.1002/pro.3330)
- Jumper et al., *Highly accurate protein structure prediction with
  AlphaFold*, [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
