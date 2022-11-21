import sys
import os


def add_to_list(out, chrom, start, end, meth_value):
    if 0.125 <= meth_value < 0.375:
        reads.append(chrom+"\t" + str(start) + "\t" + str(end) + "\t0\t1\t0\t0\t0\n")
    elif 0.375 <= meth_value < 0.625:
        reads.append(chrom+"\t" + str(start) + "\t" + str(end) + "\t0\t0\t1\t0\t0\n")
    elif 0.625 <= meth_value < 0.875:
        reads.append(chrom+"\t" + str(start) + "\t" + str(end) + "\t0\t0\t0\t1\t0\n")
    elif 0.875 <= meth_value <= 1:
        reads.append(chrom+"\t" + str(start) + "\t" + str(end) + "\t0\t0\t0\t0\t1\n")
    else:
        reads.append(chrom+"\t" + str(start) + "\t" + str(end) + "\t1\t0\t0\t0\t0\n")


if __name__ == "__main__":

    # output directory
    out_dir = sys.argv[1]
    # input file (should be sorted by read ID !)
    readfile = sys.argv[2]

    reads = []
    name = os.path.basename(readfile).split('.')[0].split('_')[2]
    outfile = out_dir + '/' + name + '.bed'
    with open(readfile, 'r') as f:
        meth_count = 0
        total_count = 0
        prev = ""
        chrom = ""
        start = -1
        end = -1
        for line in f:
            line = line.strip().split()
            id = line[0]
            meth = line[1]
            pos = line[3]
            if prev != id:
                # Only use reads that cover at least 3 CpG sites
                if total_count >= 3:
                    add_to_list(reads, chrom, start, end, meth_count / total_count)
                meth_count = 0
                total_count = 0
                prev = id
                start = pos
                end = pos
            chrom = line[2]
            total_count += 1
            if end < pos:
                end = pos
            if meth == "+":
                meth_count += 1
    with open(outfile, 'w') as out:
        out.writelines(reads)