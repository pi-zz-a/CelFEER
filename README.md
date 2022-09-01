# CelFEER

## Overview
Adaptation of CelFiE by [Caggiano et al. (2021)](https://www.nature.com/articles/s41467-021-22901-x).
The code for CelFiE can be found [here](https://github.com/christacaggiano/celfie).  

CelFiE is a cell type deconvolution method that takes the cell-free DNA (cfDNA) methylation of one or more individuals and outputs the cell type proportions within these cfDNA samples. This is done using a reference panel consisting of multiple cell types. Both the reference and input data are the number of methylated and unmethylated reads that cover each cell type informative marker.

CelFEER, on the other hand, models the cell type proportions using the percentage of methylated CpG sites per individual read, thus shifting the resolution from global read count per marker to CpG site per read.
CelFEER aims to solve the same problem as CelFiE, but with an increased sensitivity to rare cell types and a decreased sensitivity to noise.

## How to run CelFEER

Three different scripts are made available; to run on real cfDNA data, to simulate mixtures of cfDNA using cell type data and to generate cfDNA data.
For each of the individual scripts, run ```python scripts/<input_script>.py -h``` for a detailed explanation of each parameter.

CfDNA data:
```sh
python scripts/celfeer.py <input_path> <output_directory> <num_samples> <--max_iterations> <--unknowns> <--parallel_job_id <--convergence> <--random_restarts>
```

Simulated data:
```sh
python scripts/celfeer_WGBS_sim.py <input_path> <output_directory> <num_samples> <--max_iterations> <--unknowns> <--parallel_job_id <--convergence> <--random_restarts> <--proportions>
```

Generated data:
```sh
python scripts/celfeer_generated_sim.py <output_directory> <num_samples> <num_tissues> <num_cpgs> <depth> <beta_depth> <--max_iterations> <--unknowns> <--parallel_job_id> <--convergence> <--random_restarts>
```

## Expected input

The model expects the input and reference data in a single file, with each line in the following format (tab separated):  
``chrom chrom_start chrom_end <cfDNA data> chrom chrom_start chrom_end <reference data>.``

The cfDNA data of a single individual consists of the number of reads that are 0, 25, 50, 75 and 100 percent methylated respectively (where the methylation percentage is rounded to the closest of these five values).
Only reads that cover at least three CpG sites are used.
The different samples are tab separated.
The same goes for the reference data, which instead contains the number of reads methylated at each percentage for all different cell types in the reference data.  

A single input line may look as follows:  
``chr1 <tab> 1 <tab> 500 <tab> 0 1 0 2 34 <tab> 12 8 0 0 0 <tab> chr1 <tab> 1 <tab> 500 <tab>  12 5 1 1 1 <tab> 0 1 1 5 41  <tab> 2 2 5 2 3 ``

This line contains the cfDNA of two different individuals, and the reference data of three different cell types. The first individual has 37 reads that cover position 1 to position 500 on chromosome 1, of which 34 are 100% methylated.

## Output

The model outputs two text files; one containing the estimated cell type proportions, and one containing the estimated methylation percentages of each cell type.
Both simulation scripts output four pickle files for easier processing of the results.

## Finding marker regions

We provide the code for finding marker regions that mark the identity of a specific cell type by its methylation values over the region.
At a marker region, one cell type should have a reference methylation value distinctly different from the methylation values of all other cell types.
The code for finding the markers is in ``markers.py``. This code is an adaptation of the script provided by Caggiano et al., which can be found [here](https://github.com/christacaggiano/celfie/blob/master/TIMs/tim.py).

The script should be run as follows:
```sh
python markers.py <input_file> <output_file> <num_values> <tissues> <depth_filter> <nan_filter> <extra_filter> <variant>
```  

We recommend to set the ``num_values``, i.e. the number of markers for each cell type, to 100.  The variable ``tissues`` should be set to the number of cell types in the reference. ``depth_filter`` should be the minimum median coverage of all cell types at a marker region. We recommend setting this threshold to 20. The ``nan_filter`` sets how many cell types are allowed to have missing values at a marker region, which we recommend to set to 0. The ``extra_filter`` is further explained in the paper, and turns on the two modifications on the original code, making the requirements for markers more "strict". We recommend turning this filter on, i.e. setting ``extra_filter`` to True. Finally, the ``variant`` allows the user to measure the distance between cell types in six different ways:
-  "original" measures from median
-  "hypo" finds smallest methylation from median
-   "hyper" finds largest methylation from median
-   "hypomin" finds smallest methylation from min methylation of other tissues
-   "hypermax" finds largest methylation from max methylation of other tissues
-   "mixed" uses hypomin if this distance is larger than hypermax and vice versa.

The script outputs a file describing the markers by location, number identifying the cell type, difference in methylation from that cell type to all others and the methylation of that cell type.
Use [bedtools intersect](https://bedtools.readthedocs.io/en/latest/content/tools/intersect.html) to overlap these marker regions with your input data for CelFEER.
