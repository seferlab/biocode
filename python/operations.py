import os
import sys
import networkx as nx
import argparse

def clear_op(tag):
    return "CLEAN {0}".format(tag)

def swap_op():
    return "SWAP"

def save_op():
    return "SAVE"

def load_op():
    return "LOAD"

def set_op(r1):
    return "SET({0})".format(r1)

def skip_op(p):
    return "SKIP INSTRUCTION({0})".format(p)

def rewind_op(n,tag):
    return "REWIND({0}, {1})".format(n, tag)

def createedge_op():
    return "GENERATE EDGE"

def newnode_op():
    return "NEW NODE"

def randomedge_op():
    return "RANDOM EDGE"

def randomnode_op():
    return "RANDOM NODE"

def clearinfluenced_op():
    return "CLEAR INFLUENCED"

def detachfrominfluenced_op():
    return "DISCONNECT FROM INFLUENCED"

def attachtoinfluenced_op():
    return "CONNECT TO INFLUENCED"

def influenceneighbors_op(p):
    return "INFLUENCE NEIGHBORS({0})".format(p)
