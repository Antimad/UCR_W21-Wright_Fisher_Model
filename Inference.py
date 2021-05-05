import numpy as np
import pandas as pd

Data = pd.read_json("Results.json")
mutation_rate = 0.001

Pop_Size = 1000  # Population size


def fokker_planck(x_t, x_ij, s_i, mu, i, seq):
    """

    :param x_ij:
    :param x_t: Site
    :param s_i: Selection Coefficient
    :param mu: Mutation Rate
    :param i: Must be a numerical index of site
    :param seq: Entire Sequence
    :return: Results of Fokker-Planck equation
    """
    first = mu * (1 - (2 * x_t))
    second = x_t * ((1 - x_t) * s_i)

    third = 0
    for idx, x_j in enumerate(seq["Mutant Allele"]):
        if i != idx:
            third += (x_ij - (x_t[i] * x_j)) * seq["Selection Coefficient"][idx]

    return first + second + third


def c_ij(x_t, x_ij,  i, j):
    """

    :param x_ij:
    :param x_t: Results of the Fokker-Planck Equation
    :param i: index, can be numerical or dictionary type. If dict, should be built dynamically
    :param j: index, can be numerical or dictionary type. If dict, should be built dynamically
    :return: Results of Covariance Matrix equation
    """
    if i == j:
        i_eq_j = x_t[i] * (1 - x_t[i])
        return i_eq_j

    if i != j:
        i_not_j = x_ij - (x_t[i] * x_t[j])
        return i_not_j


def identity_matrix(rows): # TODO: How do I determine the number of rows?
    # rows = possible mutants, so sequence length.
    return np.identity(n=rows)


gamma = 1  # Works just as fine


def s_inference(seq, generations):

    #TODO: Switch site and generation
    for i, generation in enumerate(generations):
        pass

    for idx, site in enumerate(seq):
        site_selection_coefficient = 0 # extract based on datatype
        mutation_rate = 0 # Most likely given and not stored here.
        # Denominator
        for i, generation in enumerate(generations):
            delta_t = generation - generations[0]  # The distance between known generations
            fo_pl = fokker_planck(x_t=site, s_i=site_selection_coefficient, mu=mutation_rate, i=idx, seq=seq, x_ij=0)
            c_xt_k = c_ij(x_t=fo_pl, i=idx, j=idx, x_ij=0) # TODO: Is this being used correctly? Doubt...

            part_1_a = delta_t * c_xt_k

            part_1_b = gamma * identity_matrix(2)  # This could be a constant before starting

            # Can use numpy module to compute x_i and x_ij


def inference(generation, site_s_coefficient):
    for idx, sequence in enumerate(generation["Sequence"]):
        freq = np.multiply(sequence, generation["Size"][idx])
        complete_gen_sum = np.matmul(generation["Size"], generation["Sequence"])
        delta_t = idx
        for i, site in enumerate(complete_gen_sum):
            x_i = site/Pop_Size
            for j, site_j in enumerate(complete_gen_sum):
                x_j = site_j / Pop_Size
                if i == j:
                    covariance = x_i * (1-x_i)
                else:
                    x_ij = 0
                    covariance = x_ij - (x_i*x_j)

                part_1a = delta_t * covariance
                part_1b = gamma * identity_matrix(2)

                part_2a = 0
                part_2b = mutation_rate*(delta_t*(1-(2*x_j)))

