#!/bin/bash
###
# CS236605: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=1
NUM_GPUS=1
QUEUE=236605
JOB_NAME="submit_job"
MAIL_USER="nevoagmon@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
NUM_TASKS=21

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=ES_deep

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
        --ntasks-per-node=$NUM_TASKS \
	--gres=gpu:$NUM_GPUS \
	-p $QUEUE \
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
srun -n $NUM_TASKS --mpi=pmi2 python $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF


