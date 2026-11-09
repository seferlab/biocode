"""EdgeNamesToID: turn an edge list with string node names into integer IDs
(port of ``EdgeNamesToID.scala``).

Each undirected edge is written once (names ordered so that the pair is
canonical); IDs are assigned in order of first appearance.  Output goes to
stdout and ends with a ``# nodes: N,# edges: M`` comment line.

Usage::

    python -m biocode edge-names-to-id named_edges.txt > edges.edg
"""

from __future__ import annotations

import argparse
import sys


def convert(lines):
    node_ids = {}
    edges = {}
    for line in lines:
        toks = line.split()
        if len(toks) < 2:
            continue
        v, w = toks[0], toks[1]
        if v > w:
            v, w = w, v
        for n in (v, w):
            if n not in node_ids:
                node_ids[n] = len(node_ids)
        edges[(v, w)] = None
    return node_ids, list(edges)


def main(argv=None, out=sys.stdout):
    ap = argparse.ArgumentParser(prog="EdgeNamesToID")
    ap.add_argument("edges", help="file with one 'name1 name2' edge per line")
    args = ap.parse_args(argv)
    with open(args.edges) as fh:
        node_ids, edges = convert(fh)
    for v, w in edges:
        out.write("%d %d\n" % (node_ids[v], node_ids[w]))
    out.write("# nodes: %d,# edges: %d\n" % (len(node_ids), len(edges)))


if __name__ == "__main__":
    main()
