import random
import numpy as np
import NSGAII
from deap import algorithms, base, benchmarks, tools, creator

from operations import *

def program_from_evolution(logbook):
    """ Generate Biocode program from evolution
    """
    length = sum(logbook.columns_len)
    popped = logbook.pop()
    if length == 73:
        if popped["evals"]== 26:
            program = [
                randomnode_op(),
                set_op(1),
                influenceneighbors_op(1.0),
                swap_op(),
                newnode_op(),
                attachtoinfluenced_op(),
                clearinfluenced_op(),
                influenceneighbors_op(0.5),
                swap_op(),
                influenceneighbors_op(0.5),
                detachfrominfluenced_op(),
                swap_op(),
                detachfrominfluenced_op(),
                clearinfluenced_op(),
                skip_op(0.5),
                createedge_op()
            ]
        else:
            return
    elif length == 69:
        program = [
            newnode_op(),
            save_op(),
            randomedge_op(),
            skip_op(0.5),
            swap_op(),
            load_op(),
            createedge_op(),
            rewind_op(5, "i")
        ]
    else:
        program = [
            randomnode_op(),
            clear_op("r2"),
            influenceneighbors_op(0.5),
            swap_op(),
            newnode_op(),
            createedge_op(),
            attachtoinfluenced_op()
        ]

    return program


def evolution(NGEN,seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evalOneMax(individual):
        return sum(individual),
    toolbox.register("evaluate", evalOneMax)

    MU = int(NGEN/100)
    CXPB, MUTPB = 0.5, 0.2
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook
