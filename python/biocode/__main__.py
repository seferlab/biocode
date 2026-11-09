"""Command line entry point: ``python -m biocode <command> [options]``.

Commands (Scala class in brackets):

    learn               evolve programs for a target network      [LearnGraph]
    grow                select / analyse an evolved program       [GrowCode.GrowGraph]
    ensemble            grow many graphs from one program         [GrowCode.GrowEnsemble]
    analyze-ensemble    compare an ensemble with the target       [GrowCode.AnalyzeEnsemble]
    analyze-population  MDS map of an evolved population          [GrowCode.AnalyzePopulation]
    edge-names-to-id    convert named edge lists to integer IDs   [EdgeNamesToID]
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "learn": "learn_graph",
    "grow": "grow_graph",
    "ensemble": "grow_ensemble",
    "analyze-ensemble": "analyze_ensemble",
    "analyze-population": "analyze_population",
    "edge-names-to-id": "edge_names_to_id",
    # Scala class names also work
    "LearnGraph": "learn_graph",
    "GrowGraph": "grow_graph",
    "GrowEnsemble": "grow_ensemble",
    "AnalyzeEnsemble": "analyze_ensemble",
    "AnalyzePopulation": "analyze_population",
    "EdgeNamesToID": "edge_names_to_id",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print("unknown command %r\n%s" % (cmd, __doc__), file=sys.stderr)
        return 2
    mod = importlib.import_module("." + COMMANDS[cmd], __package__)
    mod.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
