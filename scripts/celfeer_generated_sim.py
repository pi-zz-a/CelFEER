import os
import pickle as pkl

import numpy as np
from scipy.special import gammaln

np.seterr(divide="ignore", invalid="ignore")


## Functions


def generate_beta(i, j):
    """
    randomly generates a matrix of methylation values between 0 and 1
    tissue: number of tissues
    cpg: number of cpgs
    """
    beta = np.random.uniform(size=(i, j, 5))
    beta /= np.sum(beta, axis=2)[:, :, np.newaxis]
    return beta


def generate_reads(alpha, beta, x_depths):
    """
    generates the cfDNA reads based on the generated tissue proportions, the true
    methylation values in the reference and the depths

    alpha: tissue props
    beta: methylation values for the reference
    x_depths: simulated read depths for each CpG in each individual in cfDNA input
    """

    total_indiv = alpha.shape[0]
    i, total_cpg = beta.shape[0], beta.shape[1]

    beta = np.zeros((total_indiv, total_cpg, 5))

    for n in range(total_indiv):
        for j in range(total_cpg):
            depth = x_depths[n, j]  # depth at a paticular cpg and person
            beta_cpg = beta[:, j]  # "true" methylation value in the reference

            mix = np.random.choice(
                i,
                depth,
                replace=True,
                p=alpha[
                    n,
                ],
            )  # assign reads based on the tissue proportions for that individual
            probability = beta_cpg[mix]

            beta[n, j] = np.sum(
                np.random.binomial(1, probability, size=(depth, 5)), axis=0
            )  # the beta is the sum of all the individual reads coming from the tissues contributing to that cpg in that individual

    return beta


def generate_counts(count, probability):
    """
    generate the methylation read counts for the reference data

    count: read depths
    probability: probability of each cpg being methylated for each tissue
    """
    counts = np.zeros(probability.shape)
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            counts[i, j, :] = np.random.multinomial(count[i, j], probability[i, j, :])
    return counts


def generate_depths(depth, input_shape):
    """
    creates array of ints where each number represents the number of total reads in a tissue at a cpg

    depth: read depth
    input_shape: number of tissues X number of CpGs
    """

    return np.random.poisson(depth, input_shape)


def rare_cell_type(tissues, individuals):
    """
    creates true cell type proportions for a group of individuals where half has cfDNA containing a rare cell type
    and the other half does not

    tissues: number of tissues
    individuals: number of individuals
    """
    alpha_int_1 = np.zeros(tissues)
    alpha_int_1[0] = 0.01
    alpha_int_1[1:] = np.random.uniform(0.5, 1, size=tissues - 1)
    alpha_int_1[1:] = alpha_int_1[1:] * 0.99 / alpha_int_1[1:].sum()
    alpha_int_1 = alpha_int_1.reshape(1, tissues)
    alpha_int_1 = np.repeat(alpha_int_1, individuals / 2, axis=0)  # repeat for the number of individuals

    alpha_int_2 = np.zeros(tissues)
    alpha_int_2[1:] = np.random.uniform(0.5, 1, size=tissues - 1)
    alpha_int_2[1:] = alpha_int_2[1:] / alpha_int_2[1:].sum()
    alpha_int_2 = alpha_int_2.reshape(1, tissues)
    alpha_int_2 = np.repeat(alpha_int_2, individuals / 2, axis=0)  # repeat for the number of individuals
    alpha_int = np.vstack((alpha_int_1, alpha_int_2))
    return alpha_int


def missing_cell_type(tissues, individuals):
    """
    creates true cell type proportions drawn from a normal distribution, where one cell type in the reference data
    does not appear in the cfdna of any individual

    tissues: number of tissues
    individuals: number of individuals
    """
    alpha_int = np.zeros(tissues)
    alpha_int[-1] = np.clip(np.random.normal(0.2, 0.1), 0, 1)
    alpha_int[:-1] = np.random.uniform(0, 1, size=tissues - 1)
    alpha_int = alpha_int / alpha_int.sum()
    alpha_int = alpha_int.reshape(1, tissues)
    alpha_int = np.repeat(alpha_int, individuals, axis=0)  # repeat for the number of individuals
    return alpha_int


def generate_em_replicate(tissues, cpgs, individuals, depth, beta_depth, unknowns):
    """
    generates the input data for the simulation experiments according to the distributions described in paper

    tissues: number of tissues
    cpgs: number of cpgs
    individuals: number of individuals
    depth: expected depth of cfdna data
    beta_depth: expected depth of reference data
    unknown: number of unknown cell types
    """
    # let alpha be a simple increasing vector of proportions
    alpha_int = np.array(list(range(1, tissues + 1)))
    alpha_int = (alpha_int / alpha_int.sum()).reshape(1, len(alpha_int))
    alpha_int = np.repeat(alpha_int, individuals, axis=0)  # repeat for the number of individuals

    # comment out for replicating experiments with missing cell types
    # alpha_int = missing_cell_type(tissues, individuals)

    # comment out for replicating experiments with rare cell type
    # alpha_int = rare_cell_type(tissues, individuals)

    beta_int = generate_beta(tissues, cpgs)

    Y_int_depths = generate_depths(beta_depth, (tissues, cpgs))
    Y_int = generate_counts(Y_int_depths, beta_int)

    X_int_depths = generate_depths(depth, (individuals, cpgs))
    X_int = generate_reads(alpha_int, beta_int, X_int_depths)

    # set unknowns
    if unknowns > 0:
        Y_int[-unknowns:] = 0

    return Y_int, X_int, alpha_int, beta_int

def add_pseudocounts(array, meth):
    """
    finds values of beta where logll cannot be computed, adds pseudo-counts to make
    computation possible

    array: beta array to check for inproper value
    meth: np array of methylation counts
    """

    idx1 = np.where(
        (array == 0) | (array == 1)
    )
    meth[idx1[0], idx1[1]] += 0.01
    return meth

def check_gamma(array):
    """
    checks for values of beta where log likelihood cannot be computed, returns
    true if can be computed

    array: np array to check
    whether there is a 0 or 1 in the array
    """

    return (0 in array) or (1 in array)


########  expectation-maximization algorithm  ########


def expectation(beta, alpha):
    """
    calculates the components needed for log likelihood for each iteration of beta and alpha

    beta: np matrix of the estimated 'true' methylation proportions
    alpha: np matrix of estimated mixing proportions
    """

    e_alpha = alpha.T[:, np.newaxis, :, np.newaxis]
    e_beta = beta[:, :, np.newaxis, :]

    p = e_beta * e_alpha

    p /= np.nansum(p, axis=0)[np.newaxis, ...]

    return p


def log_likelihood(p, x, y, beta, alpha):
    """
    calculates the log likelihood P(X, Z, Y | alpha, beta)

    p: probability that read has certain read average
    x: input reads
    y: reference reads
    beta: estimated true methylation proportions
    alpha: estimated mixing proportions
    """

    ll_alpha = alpha.T[:, np.newaxis, :]
    ll_beta = beta[:, :, np.newaxis, :]
    ll_y = y[:, :, np.newaxis, :]
    ll_x = np.transpose(x, (1, 0, 2))[np.newaxis, ...]

    ll = 0
    ll += np.sum((ll_y + p * ll_x) * np.log(ll_beta))
    ll += np.sum(np.sum(p * ll_x, axis=3) * np.log(ll_alpha))
    ll += np.sum(gammaln(np.sum(ll_y, axis=3) + 1) - np.sum(gammaln(np.sum(ll_y, axis=3) + 1)))

    return ll


def maximization(p, x, y):
    """
    maximizes log-likelihood, calculated in the expectation step
    calculates new alpha and beta given these new parameters

    p: probability that read has certain read average
    x: input reads
    y: reference reads
    """

    # in case of overflow or error, transform nans to 0 and inf to large float
    p = np.nan_to_num(p)
    x = np.nan_to_num(x)

    # in p: first index: tissue, second index: sites, third index: individuals
    term1 = p * np.transpose(x, (1, 0, 2))[np.newaxis, ...]
    new_alpha = np.sum(term1, axis=(1, 3)).T
    new_beta = np.sum(term1, axis=2) + y * p.shape[2]

    # check if beta goes out of bounds, if so add pseudocounts to misbehaving y values
    if check_beta(new_beta):
        add_pseudocounts(new_beta, y)
        new_beta = np.sum(term1, axis=2) + y * p.shape[2]

    # return alpha to be normalized to sum to 1
    new_alpha = np.array([row / row.sum() for row in new_alpha])
    new_beta /= np.sum(new_beta, axis=2)[:, :, np.newaxis]
    return new_alpha, new_beta


def em(x, y, num_iterations, convergence_criteria):
    """
    take in the input cfdna matrices and the reference data and
    runs the EM for the specified number of iterations, or stops once the
    convergence_criteria is reached

    x: input reads
    y: reference reads
    convergence_criteria: difference between alpha + beta before stopping
    """

    # randomly intialize alpha for each iteration
    alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
    alpha /= np.sum(alpha, axis=1)[:, np.newaxis]  # make alpha sum to 1

    # begin by checking for instances where there are no counts for y
    add_pseudocounts(y, y)

    # intialize beta to reference values
    beta = y / np.sum(y, axis=2)[:, :, np.newaxis]

    # perform EM for a given number of iterations
    for i in range(num_iterations):

        p = expectation(beta, alpha)
        a, b = maximization(p, x, y)

        # check convergence of alpha and beta
        alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
        beta_diff = np.mean(abs(b - beta)) / np.mean(abs(beta))

        if alpha_diff + beta_diff < convergence_criteria:  # if convergence criteria, break
            break

        else:  # set current evaluation of alpha and beta
            alpha = a
            beta = b

    ll = log_likelihood(
        p, x, y, beta, alpha
    )

    return alpha, beta, ll

################## run #######################


if __name__ == "__main__":

    # read command line input parameters
    parser = argparse.ArgumentParser(
        description="CelFEER - Code to perform simulations on generated data."
    )
    parser.add_argument("input_path", help="the path to the input file")
    parser.add_argument("output_directory", help="the path to the output directory")
    parser.add_argument("num_samples", type=int, help="Number of cfdna samples")
    parser.add_argument("num_tissues", type=int, help="Number of cell types")
    parser.add_argument("num_cpgs", type=int, help="Number of cpg sites")
    parser.add_argument("depth", type=int, help="expected depth of cfdna reads")
    parser.add_argument("beta_depth", type=int, help="expected depth of reference reads")
    parser.add_argument(
        "-m",
        "--max_iterations",
        default=1000,
        type=int,
        help="How long the EM should iterate before stopping, unless convergence criteria is met. Default 1000.",
    )
    parser.add_argument(
        "-u",
        "--unknowns",
        default="",
        type=str,
        help="Number of tissues in the reference data that should be treated as unknown.",
    )
    parser.add_argument(
        "-p",
        "--parallel_job_id",
        default=1,
        type=int,
        help="Replicate number in a simulation experiment. Default 1. ",
    )
    parser.add_argument(
        "-c",
        "--convergence",
        default=0.0001,
        type=float,
        help="Convergence criteria for EM. Default 0.0001.",
    )
    parser.add_argument(
        "-r",
        "--random_restarts",
        default=10,
        type=int,
        help="CelFEER will perform several random restarts and select the one with the highest log-likelihood. Default 10.",
    )

    args = parser.parse_args()
    np.random.seed(args.parallel_job_id)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    Y, X, alpha_true, beta_int = generate_em_replicate(
        args.num_tissues, args.num_cpgs, args.num_samples, args.depth, args.beta_depth, args.unknowns
    )

    random_restarts = []
    # perform for 10 random restarts
    for i in range(args.random_restarts):
        alpha, beta, ll = em(X, Y, args.max_iterations)
        random_restarts.append((ll, alpha, beta))
        print(ll)

    ll_max, alpha_max, beta_max = max(random_restarts)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_est.pkl", "wb") as f:
        pkl.dump(alpha_max, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_true.pkl", "wb") as f:
        pkl.dump(alpha_true, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_est.pkl", "wb") as f:
        pkl.dump(beta_max, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_true.pkl", "wb") as f:
        pkl.dump(beta_int, f)
