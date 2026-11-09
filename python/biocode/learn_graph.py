"""LearnGraph: evolve an ensemble of GrowCode programs for a target network.

Python port of ``LearnGraph.scala`` and the ECJ components it plugs into
(``GCInitializer``, ``GCBit``, ``BestProgramProblem``, ``GCFinisher``).  The
genetic search is NSGA-II built from DEAP's non-dominated sorting and crowding
distance, with ECJ-style operators so the example ``.params`` files keep their
meaning:

* individuals are variable-length lists of GrowCode instructions, initialised
  with ``genome-size`` random instructions (``duplicate-retries`` avoids
  identical individuals in the first generation);
* parents are picked by binary tournaments on (Pareto rank, crowding
  distance) -- ECJ's ``TournamentSelection`` with NSGA-II fitness;
* ``ListCrossoverPipeline`` two-point crossover: each parent gets its own
  pair of cut points and the middle segments are swapped, so lengths change;
* ``VectorMutationPipeline``: every gene is replaced by a random instruction
  with probability ``mutation-prob``;
* (mu + lambda) survival: parents and children are merged and the best
  ``pop.subpop.0.size`` by rank and crowding distance survive.

Usage (from the project root)::

    python -m biocode learn --cfg examples/PrefAttach/PrefAttachExp.params
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from deap import base, creator, tools

from . import marshall
from .ecj_params import ParameterDatabase
from .operations import fmt_f32, random_operation
from .problem import TargetStats, evaluate_program
from .program import Program

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    target_graph: str
    output_file: str
    num_grow_iter: int
    generations: int = 10
    pop_size: int = 100
    genome_size: int = 10
    duplicate_retries: int = 0
    num_objectives: int = 3
    crossover: str = "list"            # "list" (variable length) or "vector"
    crossover_type: str = "two"        # "one" / "two" (/ "any" for vector)
    crossover_prob: float = 0.5        # per-gene prob. for vector "any" crossover
    cx_likelihood: float = 1.0         # prob. that a parent pair is crossed
    mut_likelihood: float = 1.0        # prob. that a child goes through mutation
    mutation_prob: float = 0.3         # per-gene mutation probability
    min_child_size: int = 0
    tries: int = 1
    tournament_size: int = 2
    seed: Optional[int] = None
    jobs: int = 1
    stat_file: Optional[str] = "out.stat"
    replicates: int = 5
    grow_timeout: float = 120.0
    notes: list = field(default_factory=list)

    @classmethod
    def from_params(cls, p: ParameterDatabase):
        sub = "pop.subpop.0"
        sp = sub + ".species"
        pipe = sp + ".pipe"
        xo = pipe + ".source.0"
        notes = []

        def req(key):
            v = p.get(key)
            if v is None:
                raise SystemExit("Missing required parameter: %s" % key)
            return v

        seed_s = p.get("seed.0", "time")
        seed = None if seed_s.lower() == "time" else int(seed_s)

        xo_class = p.get(xo, "ec.vector.breed.ListCrossoverPipeline")
        if "ListCrossover" in xo_class:
            crossover = "list"
            ctype = p.first([xo + ".crossover-type", "vector.list-xover.crossover-type"], "two")
            likelihood = p.first([xo + ".likelihood", "vector.list-xover.likelihood"], "1.0")
            min_child = p.first([xo + ".min-child-size", "vector.list-xover.min-child-size"], "0")
            tries = p.first([xo + ".tries", "vector.list-xover.tries"], "1")
        elif "VectorCrossover" in xo_class:
            crossover = "vector"
            ctype = p.get(sp + ".crossover-type", "one")
            likelihood = p.first([xo + ".likelihood", "vector.xover.likelihood"], "1.0")
            min_child, tries = "0", "1"
        else:
            raise SystemExit("Unsupported crossover pipeline: %s" % xo_class)

        if p.exists(sp + ".likelihood"):
            notes.append("'%s.likelihood' is not read by ECJ (crossover likelihood is "
                         "'%s.likelihood', default 1.0); it is ignored here too."
                         % (sp, xo))
        for key, want in (("breed", "NSGA2"), ("eval", "NSGA2")):
            if want not in p.get(key, want):
                notes.append("'%s = %s' requested; this port always uses NSGA-II." % (key, p.get(key)))

        tsize = p.first([xo + ".source.0.size", "select.tournament.size"], "2")
        evalthreads = p.get_int("evalthreads", 1)

        return cls(
            target_graph=p.resolve_path(req("targetGraph")),
            output_file=p.resolve_path(req("outputFile")),
            num_grow_iter=int(req("numGrowIter")),
            generations=p.get_int("generations", 10),
            pop_size=p.get_int(sub + ".size", 100),
            genome_size=p.get_int(sp + ".genome-size", 10),
            duplicate_retries=p.get_int(sub + ".duplicate-retries", 0),
            num_objectives=p.get_int(sp + ".fitness.num-objectives", 3),
            crossover=crossover,
            crossover_type=ctype.lower(),
            crossover_prob=p.get_float(sp + ".crossover-prob", 0.5),
            cx_likelihood=float(likelihood),
            mut_likelihood=float(p.first([pipe + ".likelihood", "vector.mutate.likelihood"], "1.0")),
            mutation_prob=p.get_float(sp + ".mutation-prob", 0.0),
            min_child_size=int(min_child),
            tries=int(tries),
            tournament_size=int(tsize),
            seed=seed,
            jobs=max(1, evalthreads),
            stat_file=p.resolve_path(p.get("stat.file", "out.stat")),
            replicates=p.get_int("numReplicates", 5),
            grow_timeout=p.get_float("growTimeout", 120.0),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# DEAP types and genetic operators
# ---------------------------------------------------------------------------

def deap_types(num_objectives):
    fname = "FitnessGrowCode%d" % num_objectives
    iname = "GrowCodeIndividual%d" % num_objectives
    if not hasattr(creator, fname):
        creator.create(fname, base.Fitness, weights=(1.0,) * num_objectives)
    if not hasattr(creator, iname):
        creator.create(iname, list, fitness=getattr(creator, fname))
    return getattr(creator, iname)


def init_population(ind_cls, n, genome_size, retries, rng):
    """Random individuals; retry up to ``retries`` times to avoid duplicates."""
    pop, seen = [], set()
    for _ in range(n):
        for attempt in range(retries + 1):
            genome = tuple(random_operation(rng) for _ in range(genome_size))
            if genome not in seen:
                break
        seen.add(genome)
        pop.append(ind_cls(genome))
    return pop


def cx_list(ind1, ind2, ctype, rng, min_child_size=0, tries=1):
    """ECJ ``ListCrossoverPipeline`` (one- or two-point, independent cut points
    for each parent).  Modifies both individuals in place; returns True if a
    crossover happened."""
    s0, s1 = len(ind1), len(ind2)
    if s0 == 0 or s1 == 0:
        return False
    for _ in range(tries):
        cuts = []
        for s in (s0, s1):
            a = rng.randrange(s)
            b = s if ctype == "one" else a + rng.randrange(s - a)
            cuts.append((a, b))
        (a0, b0), (a1, b1) = cuts
        c0 = ind1[:a0] + ind2[a1:b1] + ind1[b0:]
        c1 = ind2[:a1] + ind1[a0:b0] + ind2[b1:]
        if len(c0) > min_child_size and len(c1) > min_child_size:
            ind1[:] = c0
            ind2[:] = c1
            return True
    return False


def cx_vector(ind1, ind2, ctype, rng, crossover_prob=0.5):
    """ECJ ``VectorCrossoverPipeline`` on the common prefix length."""
    n = min(len(ind1), len(ind2))
    if n < 2:
        return False
    if ctype == "one":
        a, b = 1 + rng.randrange(n - 1), n
    elif ctype == "two":
        a, b = sorted((rng.randrange(n), rng.randrange(n)))
    else:   # "any": uniform crossover
        for i in range(n):
            if rng.random() < crossover_prob:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return True
    ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
    return True


def mutate_genes(ind, prob, rng):
    """Replace each gene by a random instruction with probability ``prob``."""
    for i in range(len(ind)):
        if rng.random() < prob:
            ind[i] = random_operation(rng)
    return (ind,)


def assign_rank_and_crowding(pop):
    """Pareto rank (``ind.rank``) and crowding distance for every individual."""
    fronts = tools.sortNondominated(pop, len(pop))
    for r, front in enumerate(fronts):
        tools.emo.assignCrowdingDist(front)
        for ind in front:
            ind.rank = r
    return fronts


def select_nsga2(pop, k):
    """Keep ``k`` individuals: whole fronts first, then by crowding distance."""
    fronts = assign_rank_and_crowding(pop)
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            rest = sorted(front, key=lambda i: i.fitness.crowding_dist, reverse=True)
            chosen.extend(rest[:k - len(chosen)])
            break
    return chosen


def better(a, b):
    """NSGA-II crowded comparison (ECJ ``NSGA2MultiObjectiveFitness.betterThan``)."""
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.fitness.crowding_dist > b.fitness.crowding_dist


def tournament(pop, size, rng):
    best = pop[rng.randrange(len(pop))]
    for _ in range(size - 1):
        c = pop[rng.randrange(len(pop))]
        if better(c, best):
            best = c
    return best


def best_individual(pop):
    best = pop[0]
    for ind in pop[1:]:
        if better(ind, best):
            best = ind
    return best


# ---------------------------------------------------------------------------
# Evaluation (optionally in worker processes)
# ---------------------------------------------------------------------------

_WORKER = {}


def _worker_init(target, num_iter, replicates, timeout, seed):
    _WORKER.update(target=target, num_iter=num_iter, replicates=replicates, timeout=timeout)
    if seed is None:
        random.seed()
    else:
        random.seed(seed * 1000003 + os.getpid())


def _worker_eval(genome):
    prog = Program(genome)
    return evaluate_program(prog, _WORKER["target"], _WORKER["num_iter"],
                            replicates=_WORKER["replicates"], timeout=_WORKER["timeout"],
                            log=lambda m: print(m, file=sys.stderr, flush=True))


def evaluate_all(individuals, pool, progress=True):
    """Evaluate individuals without a valid fitness; prints a ``*`` per individual."""
    todo = [ind for ind in individuals if not ind.fitness.valid]
    genomes = [list(ind) for ind in todo]
    results = pool.imap(_worker_eval, genomes) if pool is not None else map(_worker_eval, genomes)
    for ind, fit in zip(todo, results):
        ind.fitness.values = fit
        if progress:
            sys.stdout.write("*")
            sys.stdout.flush()
    if todo and progress:
        sys.stdout.write("\n")
    return len(todo)


# ---------------------------------------------------------------------------
# Statistics output (ECJ MultiObjectiveStatistics look-alike)
# ---------------------------------------------------------------------------

def _fmt_double(x):
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return repr(float(x))


def individual_for_humans(ind):
    s = "Evaluated: T\n"
    s += "Fitness: [" + " ".join(fmt_f32(v) for v in ind.fitness.values) + " max]\n"
    s += "R=%d S=%s\n" % (getattr(ind, "rank", 0), _fmt_double(getattr(ind.fitness, "crowding_dist", 0.0)))
    s += "".join(" " + op.Str + "\n" for op in ind)
    return s


def to_programs(pop):
    """``GCFinisher``: the final population as Programs with a fitness property."""
    progs = []
    for ind in pop:
        p = Program(list(ind))
        p.properties["fitness"] = ",".join(fmt_f32(v) for v in ind.fitness.values)
        progs.append(p)
    return progs


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: EvolutionConfig, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    for n in cfg.notes:
        say("Note: " + n)
    if cfg.num_objectives != 3:
        raise SystemExit("The BioCode fitness has 3 objectives; num-objectives is %d" % cfg.num_objectives)

    rng = random.Random(cfg.seed)
    random.seed(None if cfg.seed is None else cfg.seed + 1)
    ind_cls = deap_types(cfg.num_objectives)

    target = TargetStats.from_file(cfg.target_graph, verbose=verbose)
    init_args = (target, cfg.num_grow_iter, cfg.replicates, cfg.grow_timeout, cfg.seed)
    pool = None
    if cfg.jobs > 1:
        pool = mp.get_context().Pool(cfg.jobs, initializer=_worker_init, initargs=init_args)
    else:
        _worker_init(*init_args)
        random.seed(None if cfg.seed is None else cfg.seed + 1)

    stat = None
    if cfg.stat_file:
        if os.path.dirname(cfg.stat_file):
            os.makedirs(os.path.dirname(cfg.stat_file), exist_ok=True)
        stat = open(cfg.stat_file, "w")
    t0 = time.time()
    try:
        pop = init_population(ind_cls, cfg.pop_size, cfg.genome_size, cfg.duplicate_retries, rng)
        for gen in range(cfg.generations):
            if gen > 0:
                # --- breed lambda = mu children from the current archive ----
                children = []
                while len(children) < cfg.pop_size:
                    p1 = ind_cls(tournament(pop, cfg.tournament_size, rng))
                    p2 = ind_cls(tournament(pop, cfg.tournament_size, rng))
                    if rng.random() < cfg.cx_likelihood:
                        if cfg.crossover == "list":
                            cx_list(p1, p2, cfg.crossover_type, rng, cfg.min_child_size, cfg.tries)
                        else:
                            cx_vector(p1, p2, cfg.crossover_type, rng, cfg.crossover_prob)
                    for c in (p1, p2):
                        if len(children) < cfg.pop_size:
                            if rng.random() < cfg.mut_likelihood:
                                mutate_genes(c, cfg.mutation_prob, rng)
                            children.append(c)
                pop = pop + children

            say("Generation %d: evaluating %d individuals" % (gen, sum(1 for i in pop if not i.fitness.valid)))
            evaluate_all(pop, pool, progress=verbose)
            pop = select_nsga2(pop, cfg.pop_size)

            best = best_individual(pop)
            front = [i for i in pop if i.rank == 0]
            say("Generation %d: best [%s]  front size %d  (%.1fs)" % (
                gen, " ".join(fmt_f32(v) for v in best.fitness.values), len(front), time.time() - t0))
            if stat:
                stat.write("\nGeneration: %d\nBest Individual:\nSubpopulation 0:\n%s\n"
                           % (gen, individual_for_humans(best)))
                stat.flush()

        if stat:
            stat.write("\n\n\n PARETO FRONTS\n\n\nPareto Front of Subpopulation 0\n")
            for ind in pop:
                if ind.rank == 0:
                    stat.write(individual_for_humans(ind) + "\n")
    finally:
        if stat:
            stat.close()
        if pool is not None:
            pool.close()
            pool.join()

    # --- GCFinisher -----------------------------------------------------------
    progs = to_programs(pop)
    d = os.path.dirname(cfg.output_file)
    if d:
        os.makedirs(d, exist_ok=True)
    marshall.write_to_file(cfg.output_file, progs)
    say("Wrote %d programs to %s (%.1fs)" % (len(progs), cfg.output_file, time.time() - t0))
    return pop


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="LearnGraph",
        description="Learn a graph: evolve GrowCode programs for a target network.")
    ap.add_argument("--cfg", required=True, help="file containing the configuration (ECJ .params)")
    ap.add_argument("-p", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a parameter (repeatable), e.g. -p generations=3")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="worker processes for fitness evaluation (default: evalthreads)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    params = ParameterDatabase.from_file(args.cfg)
    for ov in args.overrides:
        if "=" not in ov:
            ap.error("override must look like KEY=VALUE: %r" % ov)
        k, v = ov.split("=", 1)
        params.set(k.strip(), v.strip())
    cfg = EvolutionConfig.from_params(params)
    if args.jobs is not None:
        cfg.jobs = max(1, args.jobs)
    run(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
