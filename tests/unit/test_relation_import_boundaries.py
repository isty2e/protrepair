"""Import-boundary tests for structure provenance and relation evidence."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from protrepair.scope import AtomSetScope, WholeStructureScope
from protrepair.structure.labels import AtomRef, ResidueId


def test_cold_import_orders_preserve_relation_evidence_behavior() -> None:
    """Relation evidence must work regardless of the initial import order."""

    pythonpath = os.pathsep.join(
        part
        for part in (
            str(Path("src").resolve()),
            os.environ.get("PYTHONPATH", ""),
        )
        if part
    )
    env = {**os.environ, "PYTHONPATH": pythonpath}
    snippets = (
        "import protrepair.structure.provenance; import protrepair.relation",
        "import protrepair.relation; import protrepair.structure.provenance",
        "import protrepair.relation.evidence; import protrepair.structure.provenance",
    )
    exercise = """
from protrepair.scope import WholeStructureScope
from protrepair.relation import ExternalCorrespondenceEvidence, StructureEndpoint
from protrepair.structure.provenance import StructureProvenanceOrigin
source = StructureProvenanceOrigin(
    structure_endpoint=StructureEndpoint.source(WholeStructureScope())
)
support = StructureProvenanceOrigin(
    structure_endpoint=StructureEndpoint.supporting(WholeStructureScope())
)
evidence = ExternalCorrespondenceEvidence(
    source_origin=source, counterpart_origin=support, evidence_label=" evidence "
)
assert evidence.evidence_label == "evidence"
assert evidence.source_origin.is_source()
assert evidence.counterpart_origin.is_supporting()
"""

    for snippet in snippets:
        subprocess.run(
            [sys.executable, "-c", snippet + "\n" + exercise],
            check=True,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )


def test_relation_and_structure_provenance_support_both_import_orders() -> None:
    """Relation evidence and structure provenance should import in either order."""

    structure_provenance = importlib.import_module("protrepair.structure.provenance")
    relation = importlib.import_module("protrepair.relation")

    source_endpoint = relation.StructureEndpoint.source(WholeStructureScope())
    supporting_endpoint = relation.StructureEndpoint.supporting(
        AtomSetScope(atom_refs=(AtomRef(ResidueId("A", 1), "CA"),))
    )
    source_origin = structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=source_endpoint,
    )
    supporting_origin = structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=supporting_endpoint,
    )

    assert source_origin.is_source()
    assert supporting_origin.is_supporting()

    relation = importlib.import_module("protrepair.relation")
    structure_provenance = importlib.import_module("protrepair.structure.provenance")

    assert (
        relation.ExternalCorrespondenceEvidence(
            source_origin=source_origin,
            counterpart_origin=supporting_origin,
            evidence_label=" evidence ",
        ).evidence_label
        == "evidence"
    )
    assert structure_provenance.StructureProvenanceOrigin(
        structure_endpoint=relation.StructureEndpoint.source(WholeStructureScope()),
    ).is_source()
