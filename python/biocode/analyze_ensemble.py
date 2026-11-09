"""AnalyzeEnsemble: compare every graph in a folder with a target graph
(port of ``AnalyzeEnsemble.scala``).

Usage::

    python -m biocode analyze-ensemble -i ensemble/ -g examples/PrefAttach/PrefAttach.edg -o stats.txt

Output: a header with the target's assortativity and average clustering, then
one ``assortativity<TAB>avg_clustering<TAB>shape_L1_distance`` line per graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .console import ProgressBar
from .graph_algorithms import assortativity, avg_clustering, shape
from .graph_readers import EdgeListReader


def analyze_ensemble(ensemble_dir, orig_graph, out_file):
    files = sorted(Path(ensemble_dir).rglob("*.edg"))
    tgt = EdgeListReader.from_file_sg(orig_graph)
    tgt_shape = shape(tgt)
    tgt_assort = assortativity(tgt)
    tgt_clust = avg_clustering(tgt)
    pb = ProgressBar(len(files), "=")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as out:
        out.write("TargetAssortativity=%f\tTargetAvgClustering=%f\tTargetShapeDist=0.0\n"
                  % (tgt_assort, tgt_clust))
        for gn, f in enumerate(files):
            pb.update(gn)
            g = EdgeListReader.from_file_sg(str(f))
            diff = float(np.sum(np.abs(tgt_shape - shape(g))))
            out.write("%r\t%r\t%r\n" % (assortativity(g), avg_clustering(g), diff))
    pb.done()
    return len(files)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="AnalyzeEnsemble")
    ap.add_argument("-i", "--inputDir", required=True, help="Directory containing input graphs")
    ap.add_argument("-g", "--origGraph", required=True, help="Original (Target) graph")
    ap.add_argument("-o", "--output", required=True, help="Where output should be written")
    args = ap.parse_args(argv)
    n = analyze_ensemble(args.inputDir, args.origGraph, args.output)
    print("analysed %d graphs -> %s" % (n, args.output))


if __name__ == "__main__":
    main()
