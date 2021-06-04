import numpy as np

# Data = pd.read_json("Results.json").T
mutation_rate = 0.001

Data = np.load("Simulation_Data_2.csv", allow_pickle=True)


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


def inference(generations):
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
        generational_covariance = []
        for abv_i_idx, x_i_sum in enumerate(complete_gen_sum):

            covariance_list = []
            for abv_j_idx, x_j_sum in enumerate(complete_gen_sum):
                if abv_i_idx == abv_j_idx:
                    covariance_diagonal = (x_i_sum * (1 - x_i_sum)) / Pop_Size
                    covariance_list.append(covariance_diagonal)
                else:
                    x_ij = calc_xij(generation=sequences, i=abv_i_idx, j=abv_j_idx)
                    covariance = (x_ij - (x_i_sum*x_j_sum)) / Pop_Size  # Second portion of the covariance
                    covariance_list.append(covariance)

            generational_covariance.append(np.array(covariance_list))
        generational_covariance = np.array(generational_covariance)
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
    t_k = sum((seq_k.T * size_k).T)  # Second generation info

    for j_index, x_i in enumerate(part_2b_summation):
        part_1a = covariance_matrix[j_index].sum()
        part_1b = 1
        part_2a = t_k[j_index] - t_0[j_index]
        part_2b = x_i

        s_i = ((part_1a + part_1b)**-1) * (part_2a - part_2b)
        inferred_selections.append(s_i)


inferred_selections = []
inference(generations=Data)

selection_coefficients = sum(inferred_selections)
