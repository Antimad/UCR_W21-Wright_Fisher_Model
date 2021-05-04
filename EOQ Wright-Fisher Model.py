import numpy as np
import random
import pandas as pd

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

base_population: Dictionary with the following k/v pairs
    The starting population may be randomize, but for simplicity, a population sequence array of just zero will be used.
    **If randomize, be sure to remove diversity from the dictionary and calculate separately
    
            Sequence - sequence of zeros at the start of the population
            Fitness - starting fitness of the population
"""
s_length = 50
pop_size = 1000
generations = 100
mutation = {
    "Rate": 0.001,
    "Beneficial": np.random.randint(s_length/2),
    "Beneficial Effectiveness": np.random.uniform(0, .4),
    "Dangerous": np.random.randint(s_length/2),
    "Dangerous Effectiveness": np.random.uniform(-.4, 0)
}

base_population = {
    "Sequence": np.zeros(s_length),
    "Fitness": 1,
    "Diversity": np.array([pop_size])
}

growth_monitor = {
    "Generational Sequences": [np.array([base_population["Sequence"]])],
    "Count of Unique Generational Sequences": [base_population["Diversity"]],
    "Population": [pop_size]
}

Data = []

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
    """
    This function simply selects the number of individuals to mutate and nothing more

    :return: The number of individuals to mutate.
    """
    return np.random.binomial(pop_size, mutation["Rate"] * s_length)


def select_site_to_mutate(mutated):
    """
    This function takes the number of mutants that will mutate an returns a numpy array of the sites that will mutate,
    and randomly selects one of the possible s_length sites per mutant.

    :param mutated: The number of people that will mutate
    :return: Site that will mutate.
    """
    return np.random.randint(s_length, size=mutated)


def mutate(selected_to_mutate):
    """
    :param selected_to_mutate: list of sites, where the len() is the number of individuals selected to be mutated in
                               the population.
    :return: A dictionary of the new mutant sequences.
    """

    mutant_populous = {
        "Sequence": [],
        "Fitness": []
    }
    for individual_sequence, site_location in enumerate(selected_to_mutate):
        # Must explicitly call the .copy() function to assign a array to the variable
        mutation_sequence = base_population["Sequence"].copy()
        mutation_sequence[site_location] += (1 - int(mutation_sequence[site_location]))
        fitness = 1 + np.sum(haplotypes * mutation_sequence)

        mutant_populous["Sequence"].append(mutation_sequence)
        mutant_populous["Fitness"].append(fitness)

    return mutant_populous


for generation in range(generations):
    """
    At first the entire population will have the same sequence.
    Based on the mutation rate, a few will be selected and mutations will take place.
    
    Those who will mutate will be chosen 
    """
    mutated_population = []
    generational_monitor = {"Sequence": [],
                            "Size": []}

    for population in range(len(growth_monitor["Population"])):
        mutants = select_mutant()
        mutation_sites = select_site_to_mutate(mutants)
        mutation_seq_fit = mutate(mutation_sites)
        # growth_monitor["Population"][population] -= len(mutation_sites)

        for n_mutants, mutant_sequence in enumerate(mutation_seq_fit["Sequence"]):
            unique = True
            for previous, added_m_sequences in enumerate(mutated_population):
                if np.array_equal(mutant_sequence, added_m_sequences):
                    generational_monitor["Size"][previous] += 1
                    unique = False
                    continue
            if unique:
                generational_monitor["Sequence"].append(mutant_sequence)
                generational_monitor["Size"].append(1)
                mutated_population.append(mutant_sequence)

    # growth_monitor["Generational Sequences"].append(np.array(generational_monitor.keys()))
    # growth_monitor["Count of Unique Generational Sequences"].append(np.array(generational_monitor.values()))
    Data.append(generational_monitor)

info = pd.DataFrame(Data)
info = info.T
info.to_json("Results.json")
