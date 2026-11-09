"""Checks for the Python port.  Run from the ``python/`` folder with ``pytest``.

Comparisons against networkx are skipped when networkx is not installed.
"""

import math
import os
import random

import numpy as np
import pytest

from biocode import marshall
from biocode.ecj_params import ParameterDatabase
from biocode.global_alignment import GlobalAlignment, alignment_score
from biocode.graph_algorithms import (assortativity, avg_clustering, characteristic_path_length,
                                      diameter, eigenvector_centrality, laplacian, shape)
from biocode.graph_readers import EdgeListReader
from biocode.growth_models import dmc, forest_fire
from biocode.learn_graph import EvolutionConfig, cx_list, deap_types, select_nsga2
from biocode.machine import Machine
from biocode.mds import classical_mds
from biocode import metrics
from biocode.operations import RonPaulOp, from_token_list, random_operation
from biocode.plfit import hurwitz_zeta, plfit
from biocode.problem import TargetStats, evaluate_program, score_graph
from biocode.program import Program
from biocode.grow_graph import select_program
from biocode.slimgraph import EdgeGraphAlgorithms, SlimGraphUnDirected, UndirectedEdge, WeightedUndirectedEdge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX = os.path.join(ROOT, "examples")
PA_EDG = os.path.join(EX, "PrefAttach", "PrefAttach.edg")
YE_EDG = os.path.join(EX, "YeastPPI", "YeastPPI.edg")

try:
    import networkx as nx
except ImportError:      # pragma: no cover
    nx = None
needs_nx = pytest.mark.skipif(nx is None, reason="networkx not installed")


def to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(g.vertices)
    G.add_edges_from((e.source, e.target) for e in g.edges)
    return G


def grown_graph(seed=3, it=300):
    random.seed(seed)
    m = Machine()
    m.grow(Program.from_file(os.path.join(EX, "PrefAttach", "PrefAttach.grw")), it)
    return m.graph


# --------------------------------------------------------------------------- graph

def test_undirected_edge_normalisation_and_degree():
    g = SlimGraphUnDirected()
    assert g.insert_edge(UndirectedEdge(3, 1))
    assert not g.insert_edge(UndirectedEdge(1, 3))
    g.insert_edge(UndirectedEdge(2, 2))          # self loop counts twice
    assert g.degree(2) == 2 and g.degree(1) == 1
    assert g.has_edge(3, 1) and g.has_edge(1, 3)
    assert g.edge_option(1, 3) == (1, 3)
    assert g.remove_edge(3, 1) and g.size() == 1
    assert g.remove_vertex(2) and g.size() == 0


@needs_nx
@pytest.mark.parametrize("path", [PA_EDG, YE_EDG])
def test_metrics_match_networkx(path):
    g = EdgeListReader.from_file_sg(path)
    G = to_nx(g)
    assert g.order() == G.number_of_nodes() and g.size() == G.number_of_edges()
    assert avg_clustering(g) == pytest.approx(nx.average_clustering(G), abs=1e-12)
    assert metrics.assortativity(g) == pytest.approx(nx.degree_assortativity_coefficient(G), abs=1e-9)
    # the fitness version adds one padding observation -> close but not equal
    assert assortativity(g) == pytest.approx(metrics.assortativity(g), abs=5e-3)


@needs_nx
def test_metrics_on_grown_graph_with_self_loops():
    g = grown_graph()
    G = to_nx(g)
    assert avg_clustering(g) == pytest.approx(nx.average_clustering(G), abs=1e-12)
    cc = max(nx.connected_components(G), key=len)
    sub = SlimGraphUnDirected()
    for e in g.edges:
        if e.source in cc:
            sub.insert_edge(e)
    assert diameter(sub) == nx.diameter(G.subgraph(cc))


@needs_nx
def test_triangles_and_mst():
    g = EdgeListReader.from_file_sg(PA_EDG)
    G = to_nx(g)
    n_tri = sum(nx.triangles(G).values()) // 3
    assert len(EdgeGraphAlgorithms.triangles(g)) == n_tri
    assert len(EdgeGraphAlgorithms.fast_triangles(g)) == n_tri
    assert len(EdgeGraphAlgorithms.triangles_by_edge(g)) == n_tri
    w = SlimGraphUnDirected()
    rnd = random.Random(0)
    for e in g.edges:
        w.insert_edge(WeightedUndirectedEdge(e.source, e.target, rnd.random()))
    t = EdgeGraphAlgorithms.minimum_spanning_tree(w, lambda e: e.weight)
    W = nx.Graph()
    W.add_weighted_edges_from((e.source, e.target, e.weight) for e in w.edges)
    ours = sum(e.weight for e in t.edges)
    theirs = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(W, data=True))
    assert ours == pytest.approx(theirs)


@needs_nx
def test_spectral():
    g = EdgeListReader.from_file_sg(PA_EDG)
    G = to_nx(g)
    ev = eigenvector_centrality(g)
    ref = nx.eigenvector_centrality_numpy(G)
    ref = np.array([ref[v] for v in g.vertices])
    assert np.corrcoef(ev, ref)[0, 1] > 0.99
    lap = laplacian(g)
    ref = np.sort(np.linalg.eigvalsh(nx.normalized_laplacian_matrix(G, nodelist=list(g.vertices)).toarray()))
    assert np.allclose(lap, ref, atol=1e-8)
    assert characteristic_path_length(g) > 1


def test_shape_is_cumulative():
    s = shape(EdgeListReader.from_file_sg(PA_EDG))
    assert len(s) == 20 and s[-1] == pytest.approx(20.0)
    assert np.all(np.diff(s) >= 0)


# --------------------------------------------------------------------------- programs

def test_program_parsing_and_roundtrip(tmp_path):
    p = Program.from_file(os.path.join(EX, "PrefAttach", "PrefAttach.grw"))
    assert p.pcode[0].Str == "step 1" and p.pcode[-1].Str == "loop"
    assert p.pcode[4].Str == "scoin 0.12510872"
    q = Program(p.pcode[1:-1])
    f = tmp_path / "x.grw"
    q.to_file(str(f))
    r = Program.from_file(str(f))
    assert [o.Str for o in r.pcode[1:-1]] == [o.Str for o in q.pcode]
    # the Yeast example uses the old "ronpaul" spelling
    y = Program.from_file(os.path.join(EX, "YeastPPI", "YeastPPIBest.grw"))
    assert any(isinstance(o, RonPaulOp) for o in y.pcode)
    with pytest.raises(ValueError):
        from_token_list(["bogus"])


def test_operation_value_semantics():
    rng = random.Random(1)
    for _ in range(200):
        op = random_operation(rng)
        assert from_token_list(op.Str.split()) == op
        assert hash(from_token_list(op.Str.split())) == hash(op)


def test_machine_is_reproducible():
    a = grown_graph(seed=11, it=200)
    b = grown_graph(seed=11, it=200)
    assert sorted(a.edges) == sorted(b.edges)
    assert a.order() == 202          # one "new" per run


def test_machine_guards():
    p = Program.parse_lines(["new", "rewind 1,1000"], wrap=False)
    m = Machine()
    m.grow(p, 10_000, max_order=500)
    assert m.halt and m.graph.order() >= 500
    m = Machine()
    m.grow(p, 10_000_000, timeout=0.2)
    assert m.timed_out


def test_fitness_function():
    target = TargetStats.from_file(PA_EDG, verbose=False)
    assert score_graph(EdgeListReader.from_file_sg(PA_EDG), target) == (1.0, pytest.approx(10000.0), 1.0)
    random.seed(5)
    prog = Program(Program.from_file(os.path.join(EX, "PrefAttach", "PrefAttach.grw")).pcode[1:-1])
    fit = evaluate_program(prog, target, 498)
    assert len(fit) == 3 and all(v > 0 for v in fit)
    bad = Program.parse_lines(["new", "new", "new"], wrap=False)
    assert evaluate_program(bad, target, 498) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- GA

def test_list_crossover_changes_lengths():
    cls = deap_types(3)
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        a = cls(random_operation(rng) for _ in range(10))
        b = cls(random_operation(rng) for _ in range(10))
        before = sorted(map(repr, a + b))
        assert cx_list(a, b, "two", rng)
        assert sorted(map(repr, a + b)) == before      # genes are only moved
        assert len(a) >= 1 and len(b) >= 1 and len(a) + len(b) == 20
        seen.add(len(a))
    assert len(seen) > 5


def test_nsga2_selection():
    cls = deap_types(3)
    pop = []
    for v in [(1, 1, 1), (0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0), (0.2, 0.2, 0.2)]:
        ind = cls([])
        ind.fitness.values = v
        pop.append(ind)
    chosen = select_nsga2(pop, 2)
    assert chosen[0].fitness.values == (1, 1, 1) and chosen[0].rank == 0


def test_params_files():
    for name in ["PrefAttach/PrefAttachExp.params", "YeastPPI/YeastPPIExp.params",
                 "GeneRegulatory/ASRouteViewsExp.params"]:
        cfg = EvolutionConfig.from_params(ParameterDatabase.from_file(os.path.join(EX, name)))
        assert cfg.num_objectives == 3 and cfg.crossover == "list" and cfg.crossover_type == "two"
        assert cfg.cx_likelihood == 1.0 and cfg.tournament_size == 2
    cfg = EvolutionConfig.from_params(ParameterDatabase.from_file(os.path.join(EX, "PrefAttach/PrefAttachExp.params")))
    assert (cfg.pop_size, cfg.genome_size, cfg.generations, cfg.num_grow_iter) == (100, 10, 10, 498)
    assert cfg.mutation_prob == pytest.approx(0.3)


def test_select_program():
    progs = []
    for fit in ["0.9,0.1,0.9", "0.95,0.3,0.97", "0.5,0.5,0.5"]:
        p = Program([random_operation()])
        p.properties["fitness"] = fit
        progs.append(p)
    sorted_progs, idx = select_program(progs, out=lambda *a: None)
    assert sorted_progs[0].properties["fitness"].startswith("0.95")
    assert idx == 0                              # best sum/stddev trade-off
    assert select_program(progs, prog_ind=2, out=lambda *a: None)[1] == 2
    # a row whose normalised objectives are all equal has stddev 0 -> score -inf -> wins
    p = Program([random_operation()])
    p.properties["fitness"] = "0.475,0.25,0.485"
    sorted_progs, idx = select_program(progs + [p], out=lambda *a: None)
    assert sorted_progs[idx] is p


def test_marshall_roundtrip(tmp_path):
    p = Program([random_operation() for _ in range(5)])
    p.properties["fitness"] = "1,2,3"
    f = str(tmp_path / "pop.grw")
    marshall.write_to_file(f, [p])
    assert not marshall.is_ascii(f)
    q = marshall.read(f)[0]
    assert q.shortStr == p.shortStr and q.properties == p.properties


# --------------------------------------------------------------------------- misc

def test_alignment():
    ga = GlobalAlignment("kitten", "sitting").align()
    assert ga.score == alignment_score("kitten", "sitting")
    assert alignment_score("abc", "abc") == -3
    assert alignment_score("", "abc") == 3


def test_mds_recovers_distances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Y, err = classical_mds(D2, 2)
    D2y = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    assert np.allclose(D2, D2y, atol=1e-8)


def test_hurwitz_zeta_and_plfit():
    assert hurwitz_zeta(2.0, 1.0) == pytest.approx(math.pi ** 2 / 6, rel=1e-10)
    assert hurwitz_zeta(3.0, 2.0) == pytest.approx(1.2020569031595942 - 1, rel=1e-10)
    rng = random.Random(0)
    data = [math.floor(0.5 * (1 - rng.random()) ** (-1 / 1.5) + 0.5) for _ in range(3000)]
    alpha, xmin, ll, p, gof = plfit(data, n_boot=0)
    assert alpha == pytest.approx(2.5, abs=0.15)


def test_growth_models():
    g = forest_fire(300, 0.3, rng=random.Random(1))
    assert g.order() == 300
    h = dmc(300, 0.4, 0.3, rng=random.Random(1))
    assert h.order() >= 2
