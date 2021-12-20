#!/bin/bash

#SBATCH --job-name=extract_tweets
#SBATCH --output=logs/extract_tweets.log
#SBATCH --error=logs/extract_tweets.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


#PYTHON_PROGRAM='/logiciels/Python-3.7.3/bin/python3.7'

data_dir="/projets/sig/mullah/nlp/prevision_fgpi/"

echo "launching ..."
python3 app/data_handler/extracting_tweets.py $data_dir
echo "done."

