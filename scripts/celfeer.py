#!/usr/bin/env python

import argparse
import pickle as pkl
import time

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
        a, g = maximization(p, x, y)

        # check convergence of alpha and beta
        alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
        beta_diff = np.mean(abs(g - beta)) / np.mean(abs(beta))

        if alpha_diff + beta_diff < convergence_criteria:  # if convergence criteria, break
            break

        else:  # set current evaluation of alpha and beta
            alpha = a
            beta = g

    ll = log_likelihood(
        p, x, y, beta, alpha
    )

    return alpha, beta, ll

################## read in data #######################


def define_arrays(sample, num_samples, num_unk, num_tissues):
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

    x = np.array(np.split(test, num_samples, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    # add one unknown component
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
    output_matrix: celfie estimate
    header: tissue names
    index: either number of cpgs or number of samples, depending on type of output
    written
    """
    if len(output_matrix.shape) == 3:
        markers, tissues, _ = output_matrix.shape
        output = pd.DataFrame(output_matrix.reshape(markers, tissues * 5))
        output.columns = pd.MultiIndex.from_product([header[:num_tissues+unk], [0, 0.25, 0.5, 0.75, 1]])
    else:
        output = pd.DataFrame(output_matrix)
        output.columns = header[:num_tissues + unk]
    output.insert(
        0, "", index
    )  # insert either the sample names or cpg numbers as first col

    output.to_csv(output_file, sep="\t", index=False)  # save as text file


################## run #######################

if __name__ == "__main__":
    startTime=time.time()
    # read command line input parameters
    parser = argparse.ArgumentParser(
        description="CelFiE - Cell-free DNA decomposition. CelFie estimated the cell type of origin proportions of a cell-free DNA sample."
    )
    parser.add_argument("--input_path", help="the path to the input file",
                        default="../data/rb_input/ALS_reads.txt")
    parser.add_argument("--output_directory", help="the path to the output directory",
                        default="../output/rb_output/ALS")
    parser.add_argument("--num_samples", type=int, help="Number of cfdna samples", default=8)
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
        help="Number of unknown categories to be estimated along with the reference data. Default 1.",
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
        help="CelFiE will perform several random restarts and select the one with the highest log-likelihood. Default 10.",
    )
    args = parser.parse_args()
    num_tissues = 19
    np.random.seed(42)

    # make output directory if it does not exist
    # if not os.path.exists(args.output_directory) and args.parallel_job_id == 1:
    #     os.makedirs(args.output_directory)
    #     print("made " + args.output_directory + "/")
    #     print()
    # else:
    #     print("writing to " + args.output_directory + "/")

    data_df = pd.read_csv(
        args.input_path, delimiter="\t", header=None, skiprows=1
    )  # read input samples/reference data

    print(f"finished reading {args.input_path}")
    print()

    output_alpha_file = f"{args.output_directory}_tissue_proportions.txt"
    output_gamma_file = f"{args.output_directory}_methylation_proportions.txt"

    print(f"beginning generation of {args.output_directory}")
    print()

    # make input arrays and add the specified number of unknowns
    x, y = define_arrays(data_df, int(args.num_samples), int(args.unknowns), num_tissues)

    #props = np.array([i / sum(range(1, num_tissues + 1)) for i in range(1, num_tissues + 1)])[::-1].reshape(1,
    #                                                                                                 num_tissues).repeat(
    #   args.num_samples, axis=0)
    # props = np.array([1 / 3 for i in range(1, num_tissues + 1)]).reshape(1, num_tissues).repeat(
    #     args.num_samples, axis=0)
    #props = pkl.load(open("../data/true_alpha_100n7t.pkl", 'rb'))
    # props = np.array([0.05, 0.05, 0.9]).reshape(1,num_tissues).repeat(args.num_samples, axis=0)
    # x, x_depths, y, y_depths = define_arrays(data_df, int(args.num_samples), int(args.unknowns))
    #props = np.array([0.05, 0.9, 0.05]).reshape(1,num_tissues).repeat(args.num_samples, axis=0)
    #x, y = my_complex_mix(data_df, props, int(args.unknowns), num_tissues)  # int(args.num_samples))

    data_df_header = pd.read_csv(
       args.input_path, delimiter="\t", nrows=1
    )

    # get header for output files
    samples, tissues = get_header(data_df_header, args.num_samples, args.unknowns)

    # Run EM with the specified iterations and convergence criteria
    random_restarts = []

    for i in range(args.random_restarts):
        alpha, gamma, ll = em(
            x, y, args.max_iterations, args.convergence
        )
        random_restarts.append((ll, alpha, gamma))

    ll_max, alpha_max, gamma_max = max(
        random_restarts
    )  # pick best random restart per replicate

    # write estimates as text files
    # write_output(output_alpha_file, alpha_max, tissues, samples, args.unknowns)
    # write_output(
    #     output_gamma_file, np.transpose(beta_max, (1, 0, 2)), tissues, list(range(len(beta_max[1]))), args.unknowns
    # )
    print("execution time: " + str(time.time() - startTime))
