from random import *

# return mean of given dataset
def calculate_mean(systemX_perm): # todo change to count 0s
    # number of positive classified instances / number of instances
    return (sum(systemX_perm) / len(systemX_perm))
    # return (sum(filter(lambda x: x > 0, systemX_perm)) / len(systemX_perm))

# return difference in mean values of two datasets
def calculate_mean_difference(systemA_perm, systemB_perm):
    return abs(calculate_mean(systemA_perm) - calculate_mean(systemB_perm))

# return a permutation of systemA and systemB
def generate_resamples(systemA, systemB):
    systemA_perm = []
    systemB_perm = []

    # iterate over all test instances
    for i in range(0, len(systemA)):
        # Pick a random number 0 or 1
        if randint(0, 1):
            # if 1, swap score for systemA and systemB
            systemA_perm.append(systemB[i])
            systemB_perm.append(systemA[i])
        else:
            # otherwise leave pair unchanged
            systemA_perm.append(systemA[i])
            systemB_perm.append(systemB[i])

    return systemA_perm, systemB_perm

# return p-value of permutation test
def permutation_test(systemA, systemB, R):
    assert len(systemA) == len(systemB), "system results aren't equal in length :(("

    # mean difference in original samples
    unpermuted_mean_difference = calculate_mean_difference(systemA, systemB)

    s = 0  # number of mean differences >= unpermuted mean difference

    # for <= 2^n permutations
    for x in range(0, R):
        # generate a permutation of the results
        systemA_perm, systemB_perm = generate_resamples(systemA, systemB)

        # calculate difference in mean between 2 systems
        difference = calculate_mean_difference(systemA_perm, systemB_perm)

        # if difference is >= mean difference in unpermuted systems increment s
        if difference >= unpermuted_mean_difference:
            s += 1

    # calculate p-value
    p = (s + 1) / (R + 1)
    print('p-value: ', p)



