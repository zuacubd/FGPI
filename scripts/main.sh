#!/bin/bash

#SBATCH --job-name=program_name-tp
#SBATCH --output=logs/program_name.log
#SBATCH --error=logs/program_name.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


PYTHON_PROGRAM='/logiciels/Python-3.7.3/bin/python3.7'


data_dir="/projets/sig/mullah/nlp/fgpi/"

print("launching ...")
$PYTHON_PROGRAM app/fgpi.py $data_dir
print("done.")

