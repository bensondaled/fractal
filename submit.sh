#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 2      # tasks requested
#SBATCH -c 8      # cores requested
#SBATCH --mem=1000  # memory in Mb
#SBATCH -o log.log  # send stdout to outfile
#SBATCH -e log.log  # send stderr to errfile
#SBATCH -t 1:30:00  # time requested in hour:minute:second

module load anaconda3/2019.10
source activate py39
python mandelbrot.py
