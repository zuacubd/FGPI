#!/bin/bash

#SBATCH --job-name=detect_community
#SBATCH --output=logs/detect_community.log
#SBATCH --error=logs/detect_community.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=48CPUNodes
#SBATCH --mem-per-cpu=8000M


#PYTHON_PROGRAM='/logiciels/Python-3.7.3/bin/python3.7'

data_dir="/projets/sig/mullah/nlp/fgpi/"

echo "launching ..."
python3 app/ml/community_detection.py $data_dir
echo "done."

