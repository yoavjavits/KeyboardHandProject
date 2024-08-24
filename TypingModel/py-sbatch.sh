#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#
# Now supports setting NUM_CORES and NUM_GPUS via command-line arguments.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job with specific cores and GPUs
# ./py-sbatch.sh --cores 8 --gpus 2 main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission, specifying cores and GPUs
# ./py-sbatch.sh --cores 4 --gpus 1 main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments, with default resource allocations
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Default Parameters for sbatch
#
NUM_CORES=16
NUM_GPUS=1
JOB_NAME="hsc"
MAIL_USER="yoavjavits@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=HSC

# Parse command-line options
while getopts c:g: flag
do
    case "${flag}" in
        c) NUM_CORES=${OPTARG};;
        g) NUM_GPUS=${OPTARG};;
    esac
done

# Shift parsed options away
shift $((OPTIND -1))

sbatch \
    -c $NUM_CORES \
    --gres=gpu:$NUM_GPUS \
    --job-name $JOB_NAME \
    --mail-user $MAIL_USER \
    --mail-type $MAIL_TYPE \
    -o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF
