import numpy as np
import os
import matplotlib.pyplot as plt

mutation_rate = 0.0001

# Pop_Size = 1000  # Population size
N = 1000 
# selection_coefficient  = 0.03

gamma = 1  # Works just as fine


def sum_mutant_allele_sites(generation: np.array, size: np.array, empty_array: np.array):
    results = empty_array
    for i, seq in enumerate(generation):
        results += seq * size[i]
    return results


def mutation_time_summation_func(x_j):
    """

    :param x_j: site
    :return: frequency * mutation calculation
    """
    two_b = mutation_rate * (1 - (2 * (x_j / N)))
    return two_b


def calc_xij(generation, i, j):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for total in column_sum:
        if total > 1:
            x_ij_count += 1

    return x_ij_count / N


def covariance_builder(generation: np.array, size: np.array, dim: int):
    temp_cov = np.zeros((dim, dim))
    generation_with_size = sum_mutant_allele_sites(generation=generation, size=size, empty_array=np.zeros(dim))
    covariance = []
    for i_idx, x_i_sum in enumerate(generation_with_size):
        covariance_list = []
        for j_idx, x_j_sum in enumerate(generation_with_size):
            x_i_freq = x_i_sum / N
            x_j_freq = x_j_sum / N
            if i_idx == j_idx:
                covariance_diagonal = (x_i_freq * (1 - x_i_freq))
                covariance_list.append(covariance_diagonal)
            else:
                x_ij = calc_xij(generation, i=i_idx, j=j_idx)
                off_diagonal_covariance = (x_ij - (x_i_freq * x_j_freq))
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    temp_cov += covariance

    return temp_cov


def inference(generations, seq_l):
    generations = np.load(generations, allow_pickle=True)
    # generations.files = ["Size", "Sequence"]
    mutation_time_summation = np.zeros(seq_l)
    covariance_matrix = np.zeros((seq_l, seq_l))
    for gen_num, sequences in enumerate(generations["Sequence"]):
        """
        complete_gen_sum preserves the sequence information, and adds the number of individuals in a population that
                        that have the same genetic sequence.

        Using this information we will know the number of individuals where i=j. Where the sites of each individual is 
        the same and possess a mutation. Then we take it and multiply it by (1 - itself).
        """

        sizes = generations["Size"][gen_num]
        summed_mutant_alleles = sum_mutant_allele_sites(generation=sequences, size=sizes, empty_array=np.zeros(seq_l))

        generational_covariance = covariance_builder(generation=sequences, size=sizes, dim=seq_l)
        covariance_matrix += generational_covariance

        mutation_time_summation += np.array(list(map(mutation_time_summation_func, summed_mutant_alleles)))

    seq_0 = generations["Sequence"][0]
    size_0 = generations["Size"][0]
    x_0 = sum((seq_0.T * size_0).T)  # First generation info

    seq_k = generations["Sequence"][-1]
    size_k = generations["Size"][-1]
    x_k = sum_mutant_allele_sites(generation=seq_k, size=size_k, empty_array=np.zeros(seq_l))/N

    regularized_covariance = covariance_matrix + (np.identity(seq_l)/N)
    inverted_covariance = np.linalg.inv(regularized_covariance)
    # inverted_covariance = np.einsum('ij->j', inverted_covariance)
    p2 = (x_k - x_0) - mutation_time_summation

    inferred_selections = inverted_covariance.dot(p2)

    print(inferred_selections)

    return inferred_selections


gen = np.array([[1, 0, 0, 0, 1],
               [1, 1, 0, 0, 0],
               [1, 0, 1, 0, 1]])
test_size = np.array([1, 1, 1])

test_file = "Data/Test_Results_LowMutation.npz"

test_results = np.load(test_file, allow_pickle=True)


def check_allele_frequency():
    test_N = 1000
    time = list(range(101))
    seq = test_results["Sequence"]
    sz = test_results["Size"]
    complete_gen_sum = []
    for gen_num, sequences in enumerate(seq):
        sizes = sz[gen_num]
        complete_gen_sum.append(sum_mutant_allele_sites(generation=sequences, size=sizes,
                                                        empty_array=np.zeros(5))/test_N)
    plt.xlabel("Generation")
    plt.ylabel("Allele")
    plt.plot(time, complete_gen_sum)
    plt.savefig("5 sites.png")


def scatter():
    x = answer
    y = [0.3]*5
    plt.close()
    plt.scatter(x, y)
    plt.savefig("scatter.png")


# cov_test = covariance_builder(generation=gen, size=test_size, dim=5)

# inputs = [os.path.join("Data", file) for root, dirs, files in os.walk("Data", topdown=False) for file in files]

answer = inference(test_file, seq_l=5)

