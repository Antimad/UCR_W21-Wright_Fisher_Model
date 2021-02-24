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


def select_mutant():
    return np.random.binomial(pop_size, mutation["Rate"] * s_length)


def mutate(selected_to_mutate):
    """
    :param selected_to_mutate: list of sites, where the len() is the number of individuals selected to be mutated in
                               the population.
    :return: A dictionary of the new mutant sequences.
    """

    def new_sequence():
        """
        Magic!

        On a serious note. It seems that your clone function in your Species class creates a new sequence for each
        mutation however, I can't figure out where or how it creates this new sequence. To me, I would start with a
        blank slate, but your program creates a unique sequence per mutation.

        :return: Magic!
        """
        return ["I'm really not sure yet..."]

    mutant_populous = {
        "Sequence": [],
        "Fitness": []
    }
    for individual, site_location in enumerate(selected_to_mutate):
        mutation_sequence = new_sequence()
        mutation_sequence[site_location] += (1 - int(mutation_sequence[site_location]))
        fitness = 1 + np.sum(haplotypes * mutation_sequence)

        mutant_populous["Sequence"].append(mutation_sequence)
        mutant_populous["Fitness"].append(fitness)

    return mutant_populous
