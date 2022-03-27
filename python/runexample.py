import networkx as nx
import os

from growGraph import *

population = 50
#Yeast PPI
graphfile = "examples/YeastPPI/YeastPPI.edg"
G = nx.read_edgelist(graphfile)
itercount = G.number_of_nodes() - 2
outputdir = "output/yeastPPI"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
biocodeoutputfile = "{0}/biocode.program".format(outputdir)
runner(graphfile, itercount, population, biocodeoutputfile)

#GWAS Collaboration Network, Barabasi-Albert Network
graphfile = "examples/BiologicalCollaboration/BiologicalCollaboration.edg"
G = nx.read_edgelist(graphfile)
itercount = G.number_of_nodes() - 2
outputdir = "output/BiologicalCollaboration"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
biocodeoutputfile = "{0}/biocode.program".format(outputdir)
runner(graphfile, itercount, population, biocodeoutputfile)

#Gene Regulatory Network, Forest Fire Network
graphfile = "examples/GeneRegulatory/GeneRegulatory.edg"
G = nx.read_edgelist(graphfile)
itercount = G.number_of_nodes() - 2
outputdir = "output/GeneRegulatory"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
biocodeoutputfile = "{0}/biocode.program".format(outputdir)
runner(graphfile, itercount, population, biocodeoutputfile)
