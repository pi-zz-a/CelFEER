import os
import pickle as pkl

import numpy as np
from scipy.special import gammaln

np.seterr(divide="ignore", invalid="ignore")


## Functions


def generate_alpha(pkl_file):
    """
    reads in pickle file dictating tissue proportions
    :param str pickle file of the tissue proportions (people X tissues)
    :return: values loaded as a numpy array

    """

    t = pkl.load(open(pkl_file, "rb"))
    return t.T


def generate_gamma(i, j):
    """
    randomly generates a matrix of methylation values between 0 and 1
    :param int tissue: number of tissues
    :param int cpg: number of cpgs
    :return: cpg x tissue matrix of random methylation values
    """
    gamma = np.random.uniform(size=(i, j, 5))
    gamma /= np.sum(gamma, axis=2)[:, :, np.newaxis]
    # beta = np.zeros((i, j, 5))
    #
    # for n in range(j):
    #     beta[:, n] = np.random.uniform(
    #         0, 1, size=i
    #     )  # draws from a uniform distribution

    return gamma


def generate_beta(alpha, gamma, x_depths):
    """
    generates the cfDNA reads based on the generated tissue proportions, the true
    methylation values in the reference and the depths
    :param array alpha: tissue props
    :param array gamma: methylation values for the reference
    :param array x_depths: simulated read depths for each CpG in each individual in cfDNA input
    :return: methylation reads for the cfDNA input (number of sites X number of individuals)

    """

    total_indiv = alpha.shape[0]
    i, total_cpg = gamma.shape[0], gamma.shape[1]

    beta = np.zeros((total_indiv, total_cpg, 5))

    for n in range(total_indiv):
        for j in range(total_cpg):
            depth = x_depths[n, j]  # depth at a paticular cpg and person
            gamma_cpg = gamma[:, j]  # "true" methylation value in the reference

            mix = np.random.choice(
                i,
                depth,
                replace=True,
                p=alpha[
                    n,
                ],
            )  # assign reads based on the tissue proportions for that individual
            probability = gamma_cpg[mix]

            beta[n, j] = np.sum(
                np.random.binomial(1, probability, size=(depth, 5)), axis=0
            )  # the beta is the sum of all the individual reads coming from the tissues contributing to that cpg in that individual

    return beta


def generate_counts(count, probability):
    """
    generate the methylation read counts for the reference data
    :param array count: read depths
    :param array probability: probability of each cpg being methylated for each tissue
    :return: array of methylated reads

    """
    counts = np.zeros(probability.shape)
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            counts[i, j, :] = np.random.multinomial(count[i, j], probability[i, j, :])
    return counts


def generate_depths(depth, input_shape):
    """
    :param int depth: read depth
    :param tuple input_shape: number of tissues X number of CpGs
    :return: array of ints where each number represents the number of total reads in a tissue at a cpg
    """

    return np.random.poisson(depth, input_shape)


def rare_cell_type(i, n):
    alpha_int_1 = np.zeros(i)
    alpha_int_1[0] = 0.01
    alpha_int_1[1:] = np.random.uniform(0.5, 1, size=i - 1)
    alpha_int_1[1:] = alpha_int_1[1:] * 0.99 / alpha_int_1[1:].sum()
    alpha_int_1 = alpha_int_1.reshape(1, i)
    alpha_int_1 = np.repeat(alpha_int_1, n / 2, axis=0)  # repeat for the number of individuals

    alpha_int_2 = np.zeros(i)
    alpha_int_2[1:] = np.random.uniform(0.5, 1, size=i - 1)
    alpha_int_2[1:] = alpha_int_2[1:] / alpha_int_2[1:].sum()
    alpha_int_2 = alpha_int_2.reshape(1, i)
    alpha_int_2 = np.repeat(alpha_int_2, n / 2, axis=0)  # repeat for the number of individuals
    alpha_int = np.vstack((alpha_int_1, alpha_int_2))
    return alpha_int


def missing_cell_type(i, n):
    alpha_int = np.zeros(i)
    alpha_int[-1] = np.clip(np.random.normal(0.2, 0.1), 0, 1)
    alpha_int[:-1] = np.random.uniform(0, 1, size=i - 1)
    alpha_int = alpha_int / alpha_int.sum()
    alpha_int = alpha_int.reshape(1, i)
    alpha_int = np.repeat(alpha_int, n, axis=0)  # repeat for the number of individuals
    return alpha_int


def generate_em_replicate(i, j, n, depth, gamma_depth, pkl_file):
    # let alpha be a simple increasing vector of proportions
    alpha_int = np.array(list(range(1, i + 1)))
    alpha_int = (alpha_int / alpha_int.sum()).reshape(1, len(alpha_int))
    alpha_int = np.repeat(alpha_int, n, axis=0)  # repeat for the number of individuals
    alpha_int = missing_cell_type(i, n)

    gamma_int = generate_gamma(i, j)

    Y_int_depths = generate_depths(gamma_depth, (i, j))
    Y_int = generate_counts(Y_int_depths, gamma_int)

    X_int_depths = generate_depths(depth, (n, j))
    X_int = generate_beta(alpha_int, gamma_int, X_int_depths)

    # set one unknown
    Y_int[-1] = 0

    return Y_int, X_int, alpha_int, gamma_int


def add_pseudocounts(values, array, meth):
    """finds values of beta where logll cannot be computed, adds pseudo-counts to make
    computation possible

    :param int value: checks for a value that will prevent computation; either 0 or 1
    :param array array: beta array to check for inproper value
    :param array meth: np array of methylation counts
    :param array meth_depths: np array of total number of reads (meth counts + unmethylated counts)

    """

    idx1 = np.where(
        (array == 0) | (array == 1)  # find indices where value isn't able to be computed
    )
    idx2 = np.where(
        (array != 0) & (array != 1)  # find indices where value isn't able to be computed
    )
    meth[idx1] += 0.5
    meth[idx2] += 1
    return meth  # add one read to methylated counts


def check_gamma(array):
    """checks for values of beta where log likelihood cannot be computed, returns
    true if can be computed

    :param array array: np array to check
    :return: whether there is a 0 or 1 in the array
    """

    return (0 in array) or (1 in array)


########  expectation-maximization algorithm  ########


def expectation(gamma, alpha):
    """calculates the components needed for loglikelihood for each iteration of beta and alpha

    beta: np matrix of the estimated 'true' methylation proportions
    alpha: np matrix of estimated mixing proportions"""

    alpha = alpha.T[:, np.newaxis, :, np.newaxis]
    gamma = gamma[:, :, np.newaxis, :]

    p = gamma * alpha

    p /= np.nansum(p, axis=0)[np.newaxis, ...]

    p = np.nan_to_num(p)
    return p


def log_likelihood(p, x, y, gamma, alpha):
    """calculates the log likelihood P(X, Z, Y | alpha, beta)

    p0: probability that read is methylated
    p1: probability read is unmethylated
    x_depths: input read depths
    x: input methylated reads
    y_depths: reference matrix read depths
    y: reference methylated counts
    beta: estimated true methylation proportions
    alpha: estimated mixing proportions
    """

    # ll_gamma = add_pseudocounts(1, beta, beta)
    # ll_y = add_pseudocounts(1, y, y)
    # ll_gamma = add_pseudocounts(0, ll_gamma, ll_gamma)
    # ll_y = add_pseudocounts(0, ll_y, ll_y)

    # Reshape arrays for faster computation
    ll_alpha = alpha.T[:, np.newaxis, :]
    ll_gamma = gamma[:, :, np.newaxis, :]
    ll_y = y[:, :, np.newaxis, :]
    ll_x = np.transpose(x, (1, 0, 2))[np.newaxis, ...]

    ll = 0
    ll += np.sum((ll_y + p * ll_x) * np.log(ll_gamma))
    ll += np.sum(np.sum(p * ll_x, axis=3) * np.log(ll_alpha))
    ll += np.sum(gammaln(np.sum(ll_y, axis=3) + 1) - np.sum(gammaln(ll_y + 1), axis=3))

    return ll


def maximization(p, x, y):
    """maximizes log-likelihood, calculated in the expectation step
    calculates new alpha and beta given these new parameters

    p0: probability that read is methylated
    p1: probability read is unmethylated
    x_depths: input read depths
    x: input methylated reads
    y_depths: reference matrix read depths
    y: reference methylated counts
    """

    # in case of overflow or error, transform nans to 0 and inf to large float
    x = np.nan_to_num(x)

    # in p: first index: tissue, second index: sites, third index: individuals
    term1 = p * np.transpose(x, (1, 0, 2))[np.newaxis, ...]
    new_alpha = np.sum(term1, axis=(1, 3)).T
    new_gamma = np.sum(term1, axis=2) + y

    #     check if beta goes out of bounds, if so add psuedocounts to misbehaving y values
    if check_gamma(new_gamma):
        add_pseudocounts(1, new_gamma, y)
        add_pseudocounts(0, new_gamma, y)
        new_gamma = np.sum(term1, axis=2) + y

    # return alpha to be normalized to sum to 1
    new_alpha = np.array([row / row.sum() for row in new_alpha])
    new_gamma /= np.sum(new_gamma, axis=2)[:, :, np.newaxis]
    new_gamma = np.nan_to_num(new_gamma)
    return new_alpha, new_gamma


########################  run em  ########################


def em(x, y, num_iterations):
    """
    run EM
    x: the methylation counts for cfdna input
    x_depths: the total read depths for cfdna input
    y: the number of methylation counts for the reference
    y_depths: the number of reads for the reference
    num_iterations: number of iterations to complete
    """
    # randomly intialize alpha for each iteration
    alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
    alpha /= np.sum(alpha, axis=1)[:, np.newaxis]  # make alpha sum to 1

    # begin by checking for instances where there are no counts for y or y_depths
    # not necessary anymore?
    add_pseudocounts(1, y, y)
    add_pseudocounts(0, y, y)

    # intialize beta to reference values
    gamma = y / np.sum(y, axis=2)[:, :, np.newaxis]

    # perform EM for a given number of iterations
    for i in range(num_iterations):

        p = expectation(gamma, alpha)
        a, g = maximization(p, x, y)
        if np.isnan(g).any():
            print(i)

        # check convergence of alpha and beta
        alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
        gamma_diff = np.mean(abs(g - gamma)) / np.mean(abs(gamma))

        if (
                alpha_diff + gamma_diff < 0.001
        ):  # if convergence criteria, break
            break

        else:  # set current evaluation of alpha and beta
            alpha = a
            gamma = g

    ll = log_likelihood(
        p, x, y, gamma, alpha
    )  # print ll for random restarts

    return alpha, gamma, ll


# maak correlatie 100%
# doe met Y en DY
# maak eerst read counts die bij een tissue horen en trek dan read counts volgens die ratios.
if __name__ == "__main__":

    output_dir = "output/read_based_2"  # sys.argv[1]  # output directory    output_dir = "output/read_based"  # sys.argv[1]  # output directory
    rep_num = 1  # what rep number this is (i ran this in parallel with all my reps running simulataneously)
    i = 10  # number of tissues
    j = 6000  # number of cpgs (should be divisible by 3 to work with grouping of 3 cpg sites)
    n = 10  # number of individuals
    c = 0  # nr of CpG sites
    depth = 10  # input depth
    gamma_depth = 10  # reference read depth
    pkl_file = "unknown_sim_0201_10people.pkl"  # str(sys.argv[7])  # pickle file of numpy array tissue proportions
    np.random.seed(rep_num)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Y, X, alpha_true, gamma_int = generate_em_replicate(
        i, j, n, depth, gamma_depth, pkl_file
    )

    random_restarts = []
    # perform for 10 random restarts
    for i in range(10):
        alpha, gamma, ll = em(X, Y, 1000)
        random_restarts.append((ll, alpha, gamma))
        print(ll)

    ll_max, alpha_max, gamma_max = max(random_restarts)

    with open(output_dir + "/" + str(rep_num) + "_alpha_est.pkl", "wb") as f:
        pkl.dump(alpha_max, f)

    with open(output_dir + "/" + str(rep_num) + "_alpha_true.pkl", "wb") as f:
        pkl.dump(alpha_true, f)

    with open(output_dir + "/" + str(rep_num) + "_gamma_est.pkl", "wb") as f:
        pkl.dump(gamma_max, f)

    with open(output_dir + "/" + str(rep_num) + "_gamma_true.pkl", "wb") as f:
        pkl.dump(gamma_int, f)
