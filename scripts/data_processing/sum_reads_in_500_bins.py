# imports
import csv
import numpy as np
import pandas as pd
import sys
import math
import collections


def get_region_dict(file):
    """
    retrieve the 500 bp bins from text file
    """

    regions_dict = collections.defaultdict(list)  

    with open(file, "r") as input_file:
        regions_file = csv.reader(input_file, delimiter="\t")

        for line in regions_file:
            chrom, start, end = line[0], int(line[1]), line[2]

            regions_dict[(chrom, start)] = np.zeros(5)

    return regions_dict


def get_methylation_counts(file, regions_dict):
    """
    add together the methylation values for all CpGs in the selected region
    """

    # file of CpGs
    with open(file, "r") as input_file:
        cpg_file = csv.reader(input_file, delimiter="\t")

        # get methylation read counts for each position
        for line in cpg_file:
            chrom, start, end = line[0], int(line[1]), line[2]
            meth = np.array(line[3:], dtype=np.float64)
            binstart = 500*math.floor(start/500)
            if (chrom, binstart) in regions_dict:
                regions_dict[(chrom, binstart)] += meth

    return regions_dict


def write_bed_file(output_file, regions_dict):
    """
    write bed file of summed counts for all tissues
    """
    with open(output_file, "w") as output:
        bed_file = csv.writer(output, delimiter="\t",  lineterminator="\n")

        for chrom, start in regions_dict:
            values = regions_dict[(chrom, start)]
            bed_file.writerow(
                [chrom] + [start] + [start+499] + list(values)
            )


if __name__ == "__main__":

    regions_file = sys.argv[1]
    tissue_cpg_file = sys.argv[2] # the input file
    output_file_name = sys.argv[3]

    regions = get_region_dict(
        regions_file
    )  # get dictionary of regions to sum within
    get_methylation_counts(
        tissue_cpg_file, regions
    )  # get methylation read counts of Cpgs within region
    write_bed_file(output_file_name, regions)  # write output
