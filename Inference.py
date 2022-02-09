import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

mutation_rate = 0.001

# Data = np.load("data/trajectory.npz", allow_pickle=True)


Pop_Size = 1000  # Population size

gamma = 1  # Works just as fine


def part_2b_func(site):
    answer = mutation_rate * (1 - (2 * site))
    return answer


def calc_xij(generation, i, j):
    ith_column = generation[:, i]
    jth_column = generation[:, j]

    column_sum = ith_column + jth_column
    x_ij_count = 0
    for total in column_sum:
        if total > 1:
            x_ij_count += 1

    return x_ij_count


def covariance_builder(generation: np.array, size: np.array):
    covariance_matrix = np.zeros((50, 50))
    generation_with_size = (generation.T * size).T
    generation_with_size = sum(generation_with_size)
    generation_with_size = generation_with_size / Pop_Size
    covariance = []
    for i_idx, x_i_sum in enumerate(generation_with_size):
        covariance_list = []
        for j_idx, x_j_sum in enumerate(generation_with_size):
            if i_idx == j_idx:
                covariance_diagonal = (x_i_sum * (1 - x_i_sum))
                covariance_list.append(covariance_diagonal)
            else:
                x_ij = calc_xij(generation, i=i_idx, j=j_idx)
                off_diagonal_covariance = (x_ij - (x_i_sum * x_j_sum))  # / sample_properties["pop_size"]
                covariance_list.append(off_diagonal_covariance)
        covariance.append(np.array(covariance_list))
    covariance = np.array(covariance)
    covariance_matrix += covariance

    return covariance_matrix


def inference(generations):
    inferred_selections = []
    generations = np.load(generations, allow_pickle=True)
    part_2_helper = 0
    part_2b_summation = 0
    covariance_matrix = np.zeros((50, 50))
    for gen_num, sequences in enumerate(generations["Sequence"]):
        sizes = generations["Size"][gen_num]
        complete_gen_sum = (sequences.T * sizes).T
        complete_gen_sum = sum(complete_gen_sum)
        part_2_helper += complete_gen_sum

        """
        complete_gen_sum preserves the sequence information, and adds the number of individuals in a population that
                        that have the same genetic sequence.
                        
        Using this information we will know the number of individuals where i=j. Where the sites of each individual is 
        the same and possess a mutation. Then we take it and multiply it by (1 - itself).
        """

        generational_covariance = covariance_builder(generation=sequences, size=sizes)
        covariance_matrix += generational_covariance

        """
        Given that K tracks time. We must stay in the outermost loop for the rest of the equation
        """
        # covariance = covariance / Pop_Size
        # covariance_int += (covariance_matrix * delta_t).sum()
        part_2b_summation += np.array(list(map(part_2b_func, complete_gen_sum)))

    seq_0 = generations["Sequence"][0]
    size_0 = generations["Size"][0]
    t_0 = sum((seq_0.T * size_0).T)  # First generation info

    seq_k = generations["Sequence"][-1]
    size_k = generations["Size"][-1]
    t_k = sum((seq_k.T * size_k).T)  # Last generation info

    for j_index, x_i in enumerate(part_2b_summation):
        part_1a = covariance_matrix[j_index].sum()
        part_1b = 1
        part_2a = t_k[j_index] - t_0[j_index]
        part_2b = x_i

        s_i = ((part_1a + part_1b)**-1) * (part_2a - part_2b)
        inferred_selections.append(s_i)

    selection_coefficients = sum(inferred_selections)
    print("Done!")

    return selection_coefficients


inputs = [os.path.join("data", file) for root, dirs, files in os.walk("data", topdown=False) for file in files]

inference(inputs[45])

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