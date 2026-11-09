"""AnalyzePopulation: 2-D map of an evolved population (port of
``AnalyzePopulation.scala``).

Programs are compared by global alignment of their text, embedded in 2-D with
classical MDS and drawn as a scatter plot coloured by fitness.

Usage::

    python -m biocode analyze-population examples/PrefAttach/population.grw -o population_mds.png

Differences from the Scala tool: it handles the comma separated
multi-objective fitness strings written by LearnGraph (``--objective`` picks
which one colours the plot, default 0), uses ``min(n, 50)`` programs instead
of requiring at least 50, and saves the plot to a file.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import marshall
from .global_alignment import alignment_score
from .mds import classical_mds


def population_embedding(progs, n=50, objective=0, compact=False):
    progs = progs[:min(n, len(progs))]
    strings = [p.compact_str if compact else p.shortStr for p in progs]
    fitness = np.array([float(p.properties["fitness"].split(",")[objective]) for p in progs])
    k = len(progs)
    D = np.zeros((k, k))
    for x in range(k):
        for y in range(x, k):
            D[x, y] = D[y, x] = alignment_score(strings[x], strings[y])
    embed, err = classical_mds(D, 2)
    max_fit = abs(fitness.min()) if fitness.min() != 0 else 1.0
    scaled = 1.0 - np.abs(fitness) / max_fit
    return embed, err, fitness, scaled


def main(argv=None):
    ap = argparse.ArgumentParser(prog="AnalyzePopulation")
    ap.add_argument("population", help="population file written by LearnGraph")
    ap.add_argument("-n", type=int, default=50, help="number of programs to use (default 50)")
    ap.add_argument("--objective", type=int, default=0,
                    help="index of the fitness objective used for colouring (default 0)")
    ap.add_argument("--compact", action="store_true",
                    help="align compact one-character instruction strings (faster)")
    ap.add_argument("-o", "--output", default="population_mds.png", help="plot file")
    args = ap.parse_args(argv)

    progs = marshall.read(args.population)
    embed, err, fitness, _ = population_embedding(progs, args.n, args.objective, args.compact)
    print("MDS residual error: %r" % err)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; printing coordinates instead")
        for (y, x), f in zip(embed, fitness):
            print("%r\t%r\t%r" % (x, y, f))
        return
    fig, ax = plt.subplots()
    sc = ax.scatter(embed[:, 1], embed[:, 0], c=fitness, cmap="viridis", s=25)
    fig.colorbar(sc, ax=ax, label="fitness objective %d" % args.objective)
    ax.set_title("Population (classical MDS of program alignment scores)")
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print("saved plot %s" % args.output)


if __name__ == "__main__":
    main()
