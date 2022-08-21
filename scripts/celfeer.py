#!/usr/bin/env python

import argparse
import time
import os
import numpy as np
import pandas as pd
from scipy.special import gammaln

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

################## read in data #######################


def define_arrays(sample, num_samples, num_unk):
    """
    takes input data matrix- cfDNA and reference, and creates the arrays to run in EM. Adds
    specified number of unknowns to estimate


    sample: pandas dataframe of data (samples and reference). Assumes there is 3 columns (chrom, start, end)
    before the samples and before the reference
    num_samples: number of samples to deconvolve
    num_unk: number of unknowns to estimate
    """
    test = sample.iloc[:, 3: (num_samples) * 5 + 3].values
    train = sample.iloc[:, (num_samples) * 5 + 3 + 3:].values
    num_tissues = train.shape[1] // 5
    # DEL
    print(num_tissues)
    # DEL

    x = np.array(np.split(test, num_samples, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, y.shape[1], 5))
    y_unknown = np.append(y, unknown, axis=0)

    return (
        np.nan_to_num(x),
        np.nan_to_num(y_unknown),
    )


def parse_header_names(header, step):
    parsed_header = []

    for i in range(0, len(header), step):
        parsed_header.append(header[i].rsplit("_", 1)[0])

    return parsed_header


def get_header(sample, num_samples, num_unk):
    """
    gets the tissue and sample names to be used in generating an interpretable output file

    sample: dataframe of input data- with header
    num_samples: number of cfDNA samples
    num_unk: number of unknowns to be estimated
    """

    header = list(sample)

    samples = parse_header_names(
        header[3: (num_samples + 3)], 1
    )  # samples are first part of header
    tissues = parse_header_names(
        header[(num_samples) + 3 + 3:], 1
    )  # tissues are second part of header

    unknowns = ["unknown" + str(i) for i in range(1, num_unk + 1)]

    return samples, tissues + unknowns


def write_output(output_file, output_matrix, header, index, unk):
    """
    write estimated methylation proportions and tissue proportions as txt file

    output_file: outputfile name
    output_matrix: celfeer estimate
    header: tissue names
    index: either number of cpgs or number of samples, depending on type of output
    written
    """
    if len(output_matrix.shape) == 3:
        markers, tissues, _ = output_matrix.shape
        output = pd.DataFrame(output_matrix.reshape(markers, tissues * 5))
        output.columns = pd.MultiIndex.from_product([header, [0, 0.25, 0.5, 0.75, 1]])
    else:
        output = pd.DataFrame(output_matrix)
        output.columns = header
    output.insert(
        0, "", index
    )  # insert either the sample names or cpg numbers as first col

    output.to_csv(output_file, sep="\t", index=False)  # save as text file


################## run #######################

if __name__ == "__main__":
    startTime=time.time()
    # read command line input parameters
    parser = argparse.ArgumentParser(
        description="CelFEER - Read resolution adaptation of CelFiE: Cell-free DNA decomposition."
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
        default=0,
        type=int,
        help="Number of unknown categories to be estimated along with the reference data. Default 0.",
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

    # make output directory if it does not exist
    if not os.path.exists(args.output_directory) and args.parallel_job_id == 1:
        os.makedirs(args.output_directory)
        print("made " + args.output_directory + "/")
        print()
    else:
        print("writing to " + args.output_directory + "/")

    data_df = pd.read_csv(
        args.input_path, delimiter="\t")  # read input samples/reference data

    print(f"finished reading {args.input_path}")
    print()

    output_alpha_file = f"{args.output_directory}_tissue_proportions.txt"
    output_beta_file = f"{args.output_directory}_methylation_proportions.txt"

    print(f"beginning generation of {args.output_directory}")
    print()

    # make input arrays and add the specified number of unknowns
    x, y = define_arrays(data_df, int(args.num_samples), int(args.unknowns))

    # get header for output files
    samples, tissues = get_header(data_df, args.num_samples, args.unknowns)

    # Run EM with the specified iterations and convergence criteria
    random_restarts = []

    for i in range(args.random_restarts):
        alpha, beta, ll = em(
            x, y, args.max_iterations, args.convergence
        )
        random_restarts.append((ll, alpha, beta))

    ll_max, alpha_max, beta_max = max(
        random_restarts
    )  # pick best random restart per replicate

    # write estimates as text files
    write_output(output_alpha_file, alpha_max, tissues, samples, args.unknowns)
    write_output(
        output_beta_file, np.transpose(beta_max, (1, 0, 2)), tissues, list(range(len(beta_max[1]))), args.unknowns
    )
