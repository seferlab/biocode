import os
import sys
import networkx as nx
import argparse
import random
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from operations import *
from evolution import *

operation_map = {
    "Clear": clear_op,
    "Swap":  swap_op,
    "Save":  save_op,
    "Load":  load_op,
    "Set":   set_op,    
    "Skipinstruction": skip_op,
    "Rewind":  rewind_op,
    "Createedge": createedge_op,
    "Newnode":  newnode_op,
    "Randomedge": randomedge_op,
    "Randomnode": randomnode_op,
    "Clearinfluenced": clearinfluenced_op,
    "Detachfrominfluenced": detachfrominfluenced_op,
    "Attachtoinfluenced": attachtoinfluenced_op,
    "Influenceneighbors": influenceneighbors_op
}

def runner(graphfile, itercount, biocodeoutputfile):
    G = nx.read_edgelist(graphfile)

    pop, logbook = evolution(itercount, seed=None)

    paramsfile = biocodeoutputfile.replace(".program", ".details")
    with open(paramsfile, "w") as outfile:
        outfile.write(str(logbook))

    program = program_from_evolution(logbook)
    with open(biocodeoutputfile, "w") as outfile:
        for ins in program:
            outfile.write(ins+"\n")
        

def makeParser():
    """
    """
    parser = argparse.ArgumentParser(description='Grow Graphs via Biocode')
    parser.add_argument('-g', dest='graphfile', type=str, action='store', default="examples/YeastPPI/YeastPPI.edg", help='Graph filepath(default: examples/YeastPPI/YeastPPI.edg)')
    parser.add_argument('-i', dest='itercount', type=int, action='store', default=100, help='iteration count(default: 1000)')
    parser.add_argument('-o', dest='biocodeoutputfile', type=str, action='store', default="examples/YeastPPI/YeastPPIBest.grw", help="Biocode output filename(default: examples/YeastPPI/YeastPPIBest.grw)")

    return parser


if  __name__ =='__main__':
    """runs Biocode
    """
    import argparse
    parser = makeParser()
    args = parser.parse_args(sys.argv[1:])
    globals().update(vars(args))

    runner(graphfile, itercount, biocodeoutputfile)
