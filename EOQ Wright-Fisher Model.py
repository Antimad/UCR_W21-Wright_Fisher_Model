import numpy as np
import random

"""
####### Initializing starting parameters. ########

s_length: length of genomic sequence

pop_size: total size of the sample population

generations: total number of generations to simulate

mutation: Dictionary with the following k/v pairs
            Rate - Mutation Rate
            Beneficial - Number of beneficial mutations
            Beneficial Effectiveness - Fitness value of the beneficial mutation will have.
            Dangerous - Number of detrimental mutations
            Dangerous Effectiveness - Fitness value of the Dangerous mutations
            
"""
s_length = 50
pop_size = 1000
generations = 100
mutation = {
    "Rate": 0.001,
    "Beneficial": np.random.randint(s_length),
    "Beneficial Effectiveness": np.random.uniform(0, .4),
    "Dangerous": np.random.randint(s_length),
    "Dangerous Effectiveness": np.random.uniform(-.4, 0)
}

haplotypes = np.zeros(s_length)

"""
A random, none repeating list of numbers are generated that will represent the sites in the haplotype where mutations
will occur.
"""
selection_assignment = random.sample(range(s_length), (mutation["Beneficial"] + mutation["Dangerous"]))

for idx, site in enumerate(selection_assignment[mutation["Beneficial"]:]):
    haplotypes[site] = mutation["Beneficial Effectiveness"]  # Can also generate more randomness at this point

for idx, site in enumerate(selection_assignment[:mutation["Dangerous"]]):
    haplotypes[site] = mutation["Dangerous Effectiveness"]  # Similarly, can generate more randomness at this point

init_rate = np.array([pop_size])
init_probability = init_rate/np.sum(init_rate)
chosen = np.random.multinomial(pop_size, pvals=init_probability)


def mutate(population):
    """
    - We take the population size, then get a random number of mutants limited to the size of the population
    - We then remove those mutants from the total population.

    :param population: current size of the population
    :return:
    """
    mutant_populous = []
    if population > 0:
        mutants = np.random.binomial(population, mutation["Rate"]*s_length)
        population -= mutants

        sequence_sites = np.random.randint(s_length, size=mutants)

        for sequence_site in sequence_sites:
            """
            I'm not too sure about this statement below, going off your program it seems like the initial sequence 
            will always be the starting haplotype 
            
            Again, the fitness variable here seems to always stay the same since the sum of the haplotypes will always
            be 0. Should I then randomize the mutation rates of each site in the haplotype?
            """
            sequence = haplotypes
            sequence[sequence_site] = 1.0
            fitness = 1 + np.sum(haplotypes*sequence)
            mutant_populous.append(fitness)

    return mutant_populous
