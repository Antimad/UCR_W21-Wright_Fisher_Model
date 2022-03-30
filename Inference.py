import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

mutation_rate = 0.001

# Data = np.load("data/trajectory.npz", allow_pickle=True)

Pop_Size = 1000  # Population size
N = 1000  # TODO: Change later!!!

gamma = 1  # Works just as fine


def sum_mutant_allele_sites(generation: np.array, size: np.array, empty_array: np.array):
    results = empty_array
    for i, seq in enumerate(generation):
        results += seq * size[i]
    return results


def part_2b_func(x_j):
    """

    :param x_j: site
    :return: frequency * mutation calculation
    """
    two_b = mutation_rate * (1 - (2 * (x_j/N)))
    return two_b


def calc_xij(generation, i, j):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for total in column_sum:
        if total > 1:
            x_ij_count += 1

    return x_ij_count/N


def covariance_builder(generation: np.array, size: np.array, dim: int):
    covariance_matrix = np.zeros((dim, dim))
    # generation_with_size = (generation.T * size).T
    # generation_with_size = sum(generation_with_size)
    generation_with_size = sum_mutant_allele_sites(generation=generation, size=size, empty_array=np.zeros(dim))
    # generation_with_size = generation_with_size  # / Pop_Size
    covariance = []
    for i_idx, x_i_sum in enumerate(generation_with_size):
        covariance_list = []
        for j_idx, x_j_sum in enumerate(generation_with_size):
            x_i_freq = x_i_sum/N
            x_j_freq = x_j_sum/N
            if i_idx == j_idx:
                covariance_diagonal = (x_i_freq * (1 - x_i_freq))
                covariance_list.append(covariance_diagonal)
            else:
                x_ij = calc_xij(generation, i=i_idx, j=j_idx)
                off_diagonal_covariance = (x_ij - (x_i_freq * x_j_freq))
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    covariance_matrix += covariance

    return covariance_matrix


def inference(generations):
    generations = np.load(generations, allow_pickle=True)
    part_2b_summation = 0
    covariance_matrix = np.zeros((50, 50))
    for gen_num, sequences in enumerate(generations["Sequence"]):
        """
        complete_gen_sum preserves the sequence information, and adds the number of individuals in a population that
                        that have the same genetic sequence.

        Using this information we will know the number of individuals where i=j. Where the sites of each individual is 
        the same and possess a mutation. Then we take it and multiply it by (1 - itself).
        """

        sizes = generations["Size"][gen_num]
        # complete_gen_sum = (sequences.T * sizes).T
        # complete_gen_sum = sum(complete_gen_sum)
        complete_gen_sum = sum_mutant_allele_sites(generation=sequences, size=sizes, empty_array=np.zeros(50))

        generational_covariance = covariance_builder(generation=sequences, size=sizes, dim=50)
        covariance_matrix += generational_covariance

        part_2b_summation += np.array(list(map(part_2b_func, complete_gen_sum)))

    seq_0 = generations["Sequence"][0]
    size_0 = generations["Size"][0]
    x_0 = sum((seq_0.T * size_0).T)  # First generation info

    seq_k = generations["Sequence"][-1]
    size_k = generations["Size"][-1]
    # x_k = sum((seq_k.T * size_k).T)  # Last generation info
    x_k = sum_mutant_allele_sites(generation=seq_k, size=size_k, empty_array=np.zeros(50))

    regularized_covariance = covariance_matrix + np.identity(50)
    inverted_covariance = np.linalg.inv(regularized_covariance)
    inverted_covariance = np.einsum('ij->j', inverted_covariance)
    p2 = ((x_k/N)-x_0) - part_2b_summation

    inferred_selections = inverted_covariance*p2

    selection_coefficients = sum(inferred_selections)

    print("Done!")

    return selection_coefficients


inputs = [os.path.join("Data", file) for root, dirs, files in os.walk("Data", topdown=False) for file in files]

answer = inference(inputs[3])

"""
if __name__ == '__main__':
    pool = Pool(11)
    outputs = pool.map(inference, inputs)

plt.hist(x=outputs)
plt.ylabel("Population")
plt.xlabel("Selection Coefficients")
plt.title("Inferred Selection Distribution")

plt.savefig("Selection Inference Distribution")

"""