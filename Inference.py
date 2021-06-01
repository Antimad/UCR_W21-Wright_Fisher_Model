import numpy as np

# Data = pd.read_json("Results.json").T
mutation_rate = 0.001

Data = np.load("Simulation_Data_2.csv", allow_pickle=True)


Pop_Size = 1000  # Population size

gamma = 1  # Works just as fine


def calc_xij(generation):
    row_sum = generation.sum(axis=1)
    x_ij_count = 0
    for total in row_sum:
        if total > 1:
            x_ij_count += 1
    return x_ij_count


def inference(generations):
    for gen_num, sequences in enumerate(generations["Sequence"]):
        sizes = generations["Size"][gen_num]
        complete_gen_sum = (sequences.T * sizes).T
        complete_gen_sum = sum(complete_gen_sum)
        """
        complete_gen_sum preserves the sequence information, and adds the number of individuals in a population that
                        that have the same genetic sequence.
                        
        Using this information we will know the number of individuals where i=j. Where the sites of each individual is 
        the same and possess a mutation. Then we take it and multiply it by (1 - itself).
        """
        x_ij = calc_xij(generation=sequences)
        covariance = 0
        gen_si = []
        for abv_i_idx, x_i_sum in enumerate(complete_gen_sum):
            delta_t = abv_i_idx + 1
            covariance += x_i_sum * (1 - x_i_sum)  # First portion of the covariance
            for abv_j_idx, x_j_sum in enumerate(complete_gen_sum):
                covariance += x_ij - (x_i_sum*x_j_sum)  # Second portion of the covariance

            """
            The entire block needs to stay in this index to track x_i_sum as it is being used.
            """
            covariance = covariance / Pop_Size

            part_1a = delta_t * covariance
            part_1b = 1
            part_2a = x_i_sum - complete_gen_sum[0]
            part_2b = mutation_rate*(delta_t*(1-(2*x_i_sum)))

            s_i = (part_1a + part_1b) * (part_2a - part_2b)
            gen_si.append(s_i)
        inferred_selections.append(np.array(gen_si))


inferred_selections = []
inference(generations=Data)

selection_coefficients = sum(inferred_selections)
