#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import os
from scipy.special import gammaln
import pickle as pkl

np.seterr(divide="ignore", invalid="ignore")

################  support functions   ################

def add_pseudocounts(array, meth):
    """finds values of beta where logll cannot be computed, adds pseudo-counts to make
    computation possible

    array: beta array to check for inproper value
    meth: np array of methylation counts
    """

    idx1 = np.where(
        (array == 0) | (array == 1)
    )
    meth[idx1[0], idx1[1]] += 0.01
    return meth


def check_beta(array):
    """checks for values of beta where log likelihood cannot be computed, returns
    true if can be computed

    array: np array to check
    """

    return (0 in array) or (1 in array)


def complex_mix(sample, proportions, unknown_cols, num_tissues):
    """normalizes the read coverage of the input tissue data and multiplies this data with the input proportions
    sets unknown tissues to 0 in the reference and initializes beta

    sample: pandas dataframe of data (samples and reference), assumes there are 3 columns (chrom, start, end)
    before the samples and before the reference
    proportions: desired proportions for each tissue
    unknown_cols: tissues to treat as unknowns
    num_tissues: amount of tissues
    """
    test = sample.iloc[:, 3: (num_tissues) * 5 + 3].values
    train = sample.iloc[:, (num_tissues) * 5 + 3 + 3:].values
    x = np.array(np.split(test, num_tissues, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    tissue_totaldepths = np.sum(x, axis=(1, 2))
    x = (x.T / tissue_totaldepths).T * np.average(tissue_totaldepths)

    x_percents = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    mix_x_percents = np.dot(proportions, x_percents)

    mix_x = mix_x_percents.reshape(proportions.shape[0], x.shape[1], 5)

    true_beta = y / np.sum(y, axis=2)[:, :, np.newaxis]

    if unknown_cols:
        for col in unknown_cols:
            y[col] = 0

    return np.nan_to_num(mix_x), np.nan_to_num(y), np.nan_to_num(true_beta)

########  expectation-maximization algorithm  ########

def expectation(beta, alpha):
    """calculates the components needed for log likelihood for each iteration of beta and alpha

    beta: np matrix of the estimated 'true' methylation proportions
    alpha: np matrix of estimated mixing proportions
    """

    e_alpha = alpha.T[:, np.newaxis, :, np.newaxis]
    e_beta = beta[:, :, np.newaxis, :]

    p = e_beta * e_alpha

    p /= np.nansum(p, axis=0)[np.newaxis, ...]

    return p


def log_likelihood(p, x, y, beta, alpha):
    """calculates the log likelihood P(X, Z, Y | alpha, beta)

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
    """maximizes log-likelihood, calculated in the expectation step
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


########################  run em  ########################


def em(x, y, num_iterations, convergence_criteria):
    """take in the input cfdna matrices and the reference data and
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
        description="CelFEER - Code to perform simulations using WGBS data."
    )
    parser.add_argument("input_path", help="the path to the input file")
    parser.add_argument("output_directory", help="the path to the output directory")
    parser.add_argument("num_samples", type=int, help="Number of cfdna samples")
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
        default=None,
        type=str,
        help="Tissues in the reference data that should be treated as unknown. Give tissue columns separated by " +
             "comma, e.g. 0,3,6. Default is none.",
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
    parser.add_argument(
        "-f",
        "--proportions",
        default=None,
        type=str,
        help="Pickle file containing the tissue mixture proportions. Put None if the proportions are hard coded.",
    )
    args = parser.parse_args()
    np.random.seed(args.parallel_job_id)
    unknowns = [int(x) for x in args.unknowns.split(",")] if args.unknowns else None
    num_tissues = args.num_samples

    # make output directory if it does not exist
    if not os.path.exists(args.output_directory) and args.parallel_job_id == 1:
        os.makedirs(args.output_directory)
        print("made " + args.output_directory + "/")
        print()
    else:
        print("writing to " + args.output_directory + "/")

    data_df = pd.read_csv(args.input_path, header=None, delimiter="\t")

    print(f"finished reading {args.input_path}")
    print()

    # If desired, hardcode the proportions here. The current array creates ascending proportions.
    props = np.array([i / sum(range(1, num_tissues + 1)) for i in range(1, num_tissues + 1)]) \
        .reshape(1, num_tissues).repeat(args.num_samples, axis=0)

    # Load file containing proportions
    if args.proportions:
        props = pkl.load(open(args.proportions, 'rb'))

    x, y, true_beta = complex_mix(data_df, props, unknowns, args.num_samples)

    # Run EM with the specified iterations and convergence criteria
    random_restarts = []

    for i in range(args.random_restarts):
        alpha, beta, ll = em(
            x, y, args.max_iterations, args.convergence
        )
        random_restarts.append((ll, alpha, beta))

    ll_max, alpha_max, beta_max = max(random_restarts)  # pick best random restart per replicate

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_est.pkl", "wb") as f:
        pkl.dump(alpha_max, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_true.pkl", "wb") as f:
        pkl.dump(props, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_est.pkl", "wb") as f:
        pkl.dump(beta_max, f)

    with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_true.pkl", "wb") as f:
        pkl.dump(true_beta, f)
