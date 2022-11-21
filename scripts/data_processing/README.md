# Recommended CelFEER data processing guide

This is a very rough guide on how to process WGBS reads as input for CelFEER.

1. Use [Bismark](https://www.bioinformatics.babraham.ac.uk/projects/bismark/) to map WGBS reads and perform methylation calls.
2. After Bismark's methylation call, you should get a CpG_context file. Sort this file on the read identifiers, then run bismark_meth_to_input.py to get the percentage of methylated CpG sites at each read. 
3. Remove the reads overlapping SNPs from e.g. dbSNP to prevent ambiguity.
4. Sum the reads in 500 bp regions using sum_reads_in_500_bins.py with the file read_bins.txt (provided in data) as a reference region file. 
5. Use Unix commands ``paste`` and ``cut`` to create a single file of all cell types, and to retain only the chromosomal position information in front of the first cell type.
6. Get the reads overlapping the marker regions using [bedtools intersect](https://bedtools.readthedocs.io/en/latest/content/tools/intersect.html).


