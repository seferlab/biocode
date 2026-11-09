"""BioCode / GrowCode in Python.

A non-parametric framework for learning how biological networks grow: small
"GrowCode" programs are executed on a virtual machine to grow a network, and a
multi-objective genetic search (NSGA-II via DEAP) looks for programs whose
networks match a target network's degree distribution shape, assortativity and
average clustering coefficient.

Python port of the Scala project (Patro, Duggal, Sefer, Wang, Filippova,
Kingsford, KDD 2012).
"""

__version__ = "1.0.0"
