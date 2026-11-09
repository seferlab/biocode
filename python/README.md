BioCode (Python port)
=====================

A Python port of BioCode / GrowCode, the non-parametric framework for learning
how biological networks grow. Small "GrowCode" programs run on a virtual
machine that grows a network. A multi-objective genetic search (NSGA-II, built
on DEAP) looks for programs whose networks match a target network on three
measures: degree-distribution shape, assortativity and average clustering.

The Scala sources in `../src/main/scala` are left untouched. This folder is a
self-contained replacement that reads the same `examples/*.params`,
`*.edg` and `*.grw` files.

Requirements
------------

* Python 3.8+
* `numpy`, `deap` (required); `matplotlib` (optional, for plots)

```bash
pip install -r python/requirements.txt      # or: pip install -e python
```

Running
-------

Run the commands from the **project root**, because the paths inside the
`.params` files are relative to it. `python/runexample.sh` does the same steps
as the original `runexample.sh`:

```bash
export PYTHONPATH=$PWD/python                # not needed after `pip install -e python`

# Evolve an ensemble of programs for a Barabasi-Albert network (k=4)
python -m biocode learn --cfg examples/PrefAttach/PrefAttachExp.params --jobs 4

# Analyse the ensemble and report the 'best' program
python -m biocode grow -g examples/PrefAttach/PrefAttach.edg -i 498 \
    -o examples/PrefAttach/PrefAttach.grw examples/PrefAttach/PrefAttach.grw
```

| Command | Scala original | What it does |
|---|---|---|
| `learn --cfg F.params [-p key=value] [-j N]` | `LearnGraph` (+ ECJ) | Evolves programs and writes the final population to `outputFile` and the per-generation report to `stat.file` |
| `grow [-g graph] [-i iters] [-o out] [-p idx] FILE` | `GrowCode.GrowGraph` | With a population file, picks a program, writes it as text and compares its graph with the original. With a text program, grows one graph, saves degree plots and can run a power-law test (`-y`) |
| `ensemble --input P --numGraphs N --iterations I --outDir D` | `GrowCode.GrowEnsemble` | Grows N graphs from one program (`D/<i>.edg`) |
| `analyze-ensemble -i D -g graph -o out.txt` | `GrowCode.AnalyzeEnsemble` | Assortativity, clustering and shape distance of every graph in D |
| `analyze-population POP [-o plot.png]` | `GrowCode.AnalyzePopulation` | MDS map of a population from program alignments |
| `edge-names-to-id edges.txt` | `EdgeNamesToID` | Turns named edges into integer IDs |

`-p` overrides any parameter, as in ECJ. For example, a quick smoke test:
`python -m biocode learn --cfg examples/PrefAttach/PrefAttachExp.params -p generations=3 -p pop.subpop.0.size=20`.

### Parameters read from the `.params` file

`targetGraph`, `outputFile`, `numGrowIter`, `generations`, `seed.0` (`time`
or an integer), `evalthreads` (worker processes, overridden by `--jobs`),
`stat.file`, `pop.subpop.0.size`, `pop.subpop.0.duplicate-retries`,
`pop.subpop.0.species.genome-size`, `...species.mutation-prob`,
`...species.fitness.num-objectives` (must be 3), `...species.pipe.source.0`
(`ListCrossoverPipeline` or `VectorCrossoverPipeline`, plus its
`crossover-type`, `likelihood`, `min-child-size`, `tries`) and
`select.tournament.size`.

Two optional keys are new: `growTimeout` (seconds per grown graph, default
120, as hard-coded in Scala) and `numReplicates` (graphs grown per fitness
evaluation, default 5).

Note: `pop.subpop.0.species.likelihood = 0.9` in the example files is not a
parameter ECJ reads. The crossover likelihood lives at
`pop.subpop.0.species.pipe.source.0.likelihood` and defaults to 1.0, so the
original runs always applied crossover. The port behaves the same way and
prints a note when it sees the unused key.

Module map
----------

| Python module | Scala source |
|---|---|
| `slimgraph.py` | `SlimGraph.scala` (graphs, edges, union-find, triangles, MST, observable graph) |
| `graph_readers.py` | `FileReaders.scala`, `EdgeListReader` in `GCInitializer.scala` |
| `metrics.py` | `Metrics.scala` |
| `graph_algorithms.py` | `SlimGraphAlgorithms.scala`, `GraphAlgorithms.scala`, `LaplacianCalculator.scala`, `PowerIteration.scala` |
| `histogram.py` | `Histogram.scala` |
| `operations.py`, `program.py`, `machine.py` | `Operation.scala`, `Program.scala`, `Machine.scala` |
| `problem.py` | `GCInitializer.scala`, `BestProgramProblem.scala` |
| `learn_graph.py` | `LearnGraph.scala`, `GCBit.scala`, `GCFinisher.scala` + ECJ NSGA-II |
| `grow_graph.py`, `grow_ensemble.py`, `analyze_ensemble.py`, `analyze_population.py`, `edge_names_to_id.py` | the tools of the same names |
| `growth_models.py` | `GrowthModels.scala` (forest fire, DMC) |
| `global_alignment.py`, `mds.py`, `marshall.py` | `GlobalAlignment.scala`, `MDS.scala`, `Marshall.scala` + `FileUtils.scala` |
| `plfit.py` | the missing `src/main/python/EvaluatePLFit.py` helper |

What is kept exactly
--------------------

* The machine semantics, including some quirks: `create` calls `nodeIn`
  four times, `rewind` counters are keyed by program position, and
  `Program.from_file` wraps code in `step 1 … loop`.
* The fitness function: 5 replicates, the 50% size gate, the three objectives
  in the order (assortativity, shape, avg clustering), and float32 rounding.
* How the metrics treat self loops (a self loop adds 2 to the degree and is
  ignored by clustering). The padded assortativity used in the fitness is also
  kept.
* The breeding scheme of the ECJ version that actually ran. The project's
  classpath loads `lib/ecj-old.jar` (ECJ 20) before `lib/ecj.jar`, and this
  port follows ECJ 20: list crossover with independent cut points per parent
  (so genome lengths change), per-gene mutation with probability
  `mutation-prob`, binary tournaments on (rank, crowding), (μ+λ) survival, and
  duplicate retries in generation 0.
* GrowGraph's program-selection rule: maximise sum / stddev of the normalised
  objectives. Be aware that this rule can favour programs whose objectives are
  all mediocre but equal.

Deliberate differences
----------------------

* **DEAP crowding distance.** DEAP normalises each objective by its range in
  the front. ECJ summed raw differences, which lets the unbounded shape
  objective dominate. Only tie-breaking inside the last surviving front is
  affected.
* **Population files** are Python pickles, not Java-serialised objects, so
  binary `.grw` files from the Scala version cannot be read (text `.grw`
  programs can). Only open population files you made yourself.
* `ronpaul` is accepted as an alias of `clearinf`. `YeastPPIBest.grw` uses
  it, and the Scala parser rejected it.
* **Early stop.** Growth halts once a graph reaches 1.5× the target's vertex
  count, because such a replicate can only score zero. Fitness values do not
  change, but runs are much faster.
* **Power-law test.** It is built in (`plfit.py`: discrete MLE, KS-chosen
  `xmin`, bootstrap p-value) instead of calling the missing external script.
* **Plots and prompts.** Plots are saved as PNG files, and the Swing
  "continue?" dialog is now a terminal prompt (or `-y`).
* **AnalyzePopulation.** It accepts multi-objective fitness strings and uses
  `min(n, 50)` programs. `classical_mds` now uses sorted eigenvector columns,
  where the Scala code indexed rows.
* Small robustness fixes: histogram values past the last bin go into the last
  bin, readers skip blank lines, and `EdgeNamesToID` numbers nodes by first
  appearance.

Speed
-----

On two cores, the PrefAttach example (100 individuals × 10 generations) takes
about 20 seconds. One YeastPPI evaluation (5 graphs × 2645 iterations) takes
about 1 second per core, so the full YeastPPI run (200 × 25) needs roughly
15–30 minutes with `--jobs 4`.

Results are reproducible with an integer `seed.0` and `--jobs 1`.

Tests
-----

```bash
cd python && pytest            # metric checks against networkx run if it is installed
```
