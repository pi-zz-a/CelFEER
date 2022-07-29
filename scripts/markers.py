import csv
import numpy as np
import bottleneck as bn  # faster nan calcs
import heapq
from collections import defaultdict
import sys


def distance(values, variant):
    """
    calculates the absolute difference between the tissue methylation
    and the median for a CpG

    values: array of methylation proportions for all tissues

    return: distance between each tissue methylation prop and median, CpG
    median methylation
    """
    # original code:
    median = bn.nanmedian(values)
    metric = 0
    if variant == "original":
        dist = np.abs(values - median)
    elif variant == "hypomin" or variant == "hypo":
    # for hypo only:
        if variant == "hypomin":
            median = [min(np.concatenate([values[:i],values[i+1:]])) for i in range(len(values))]
        dist = median-values
        metric = 1
    # for hyper only:
    elif variant == "hypermax" or variant == "hyper":
        if variant == "hypermax": 
            median = [max(np.concatenate([values[:i],values[i+1:]])) for i in range(len(values))]
        dist = values - median
        metric = 2
    elif variant == "mixed":
        #median = bn.nanmedian(values)
        mn = np.array([min(np.concatenate([values[:i],values[i+1:]])) for i in range(len(values))])
        mx = np.array([max(np.concatenate([values[:i],values[i+1:]])) for i in range(len(values))])
        #dist_med = np.abs(values - median)
        dist_min = mn - values
        dist_max = values - mx
        #dist_options = np.array([dist_med,dist_min,dist_max])
        dist_options = np.array([dist_min,dist_max])
        #med_options = np.array([np.array(median).repeat(len(values)), mn, mx])
        med_options = np.array([mn, mx])
        idx = np.where(dist_options == np.amax(dist_options))[0][0]
        dist = dist_options[idx]
        median = med_options[idx]
        metric = idx
    else: 
        raise Exception("Not a valid variant: pick between original, hypo, hypomin, hyper, hypermax, mixed")
    return dist, median, metric


def add_to_heap(heap, n, dist, median, cpg, percent, num, metric):
    """
    keep track of the top CpGs using a max heap of size n

    heap: max heap keeping track of top tims for a specific tissues
    n: size of max heap
    dist: distance between tissue methylation and median methylation
    median: median methylation for CpG
    cpg: cpg positional information (tuple of chrom, start, end)
    percent: tissue methylation percent
    num: tissue number

    return: none
    """

    if len(heap) < 3/2*n:  # only keep track of the n CpGs with greatest distance
        heapq.heappush(
            heap, (dist, cpg, median, percent, num, metric)
        )  # note- works because it only checks value of dist, not the rest of the tuple
    else:
        heapq.heappushpop(heap, (dist, cpg, median, percent, num, metric))


def get_cpgs(heap_list):
    """
    converts heap to a default dict in order to print

    heap_list: list of max heaps- size is the number of tissues

    return: dictionary of CpGs and TIM values
    """

    cpgs = defaultdict(list)

    for heap in heap_list:
        for value in heap:  # iterate over list of heaps

            # dictionary is CpG position info: number of tissue, distance between
            # tissue meth and median, methylation percent for tissue, median CpG meth
            cpgs[tuple(value[1])].append((value[4], value[0], value[3], value[2], value[5]))

    return cpgs


if __name__ == "__main__":
    # modification on line 107: ensure min depth of tissue in question
    # modification on line 118: ensure TIM is not TIM in top 3/2 of other tissues

    # user parameters
    input_file = sys.argv[1]  # input bedfile of WGBS data
    output_file = sys.argv[2]  # path to output file
    num_values = int(sys.argv[3])  # number of values to keep as tims
    tissues = int(sys.argv[4])  # total number of tissues to calc tims for
    depth_filter = int(sys.argv[5])  # depth filter
    nan_filter = int(sys.argv[6])  # number of nans to allow in TIM calc
    extra_filter = sys.argv[7]
    variant = sys.argv[8]
    
    if extra_filter == "False":
        extra_filter = False
    else:
        extra_filter = True

    # list of heaps needed to count max distances for each tissue
    distance_heaps = [[] for i in range(tissues)]

    with open(input_file, "r") as f:
        bed = csv.reader(f, delimiter="\t")

        for line in bed:
            cpg = line[0:3]  # positional information
            meth = np.asarray(line[3:], dtype="float")
            # split into subarrays of length 5
            meth = np.array(np.split(meth, tissues))
            depth = meth.sum(axis=1)
            median_depth = bn.nanmedian(depth)

            np.seterr(
                divide="ignore", invalid="ignore"
            )  # ignore Nans when there are no counts
            percents = meth / depth.reshape(1,tissues).T
            percents *= [0,0.25,0.5,0.75,1]
            percents = percents.sum(axis=1)
            nan_count = np.count_nonzero(np.isnan(percents))

            if (
                median_depth >= depth_filter and nan_count <= nan_filter
            ):  # data must pass basic quality to be a tim
                dist, median, metric = distance(percents, variant)

                for i, col_dist in enumerate(dist):
                    if not np.isnan(col_dist): #and col_dist > 0.5:
                        if not extra_filter or (depth[i] >= depth_filter and extra_filter):
                            add_to_heap(
                                distance_heaps[i],
                                num_values,
                                col_dist,
                                median,
                                cpg,
                                percents[i],
                                i,
                                metric
                            )

    distance_heaps = [sorted(distance_heaps[x], reverse=True) for x in range(tissues)]
    new_heaps = [heap[:num_values] for heap in distance_heaps]
    all_cpgs = get_cpgs(distance_heaps)  # get TIM heap for printing
    cpgs = get_cpgs(new_heaps)
    if not extra_filter:
        all_cpgs = cpgs
    
    # write the output file as tab delimited file with columns being:
    # chrom, start, end, tissue # for tim, absolute difference, methylation prop for tissue, median methylation
    # for all other tissues
    with open(output_file, "w") as o:
        dist_out = csv.writer(o, delimiter="\t")

        # header
        dist_out.writerow(
            [
                "chrom",
                "start",
                "end",
                "tissue number",
                "difference",
                "tissue methylation",
                "other tissue methylation",
            ]
        )

        count = 0
        current_tissue = 0
        for c in cpgs:
            x = np.array(all_cpgs[c])
            if len(x) == 1:
                position = [c[0], c[1], c[2]]
                entry = [cpgs[c][0][0], cpgs[c][0][1], cpgs[c][0][2], cpgs[c][0][3]]
                dist_out.writerow(position + entry)
