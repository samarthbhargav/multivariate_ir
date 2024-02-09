#!/bin/sh
#SBATCH -o /ivi/ilps/projects/multivariate_ir/logs/%A_%a.out
#SBATCH -e /ivi/ilps/projects/multivariate_ir/logs/%A_%a.err
#SBATCH -n1
#SBATCH --partition={partition}
#SBATCH -c{n_cores}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=1-{n_jobs}%{n_parallel_jobs}
#SBATCH --gres=gpu:a6000:{n_gpus}
#SBATCH --exclude=ilps-cn108

# Set-up the environment. double curly brace necessary: https://stackoverflow.com/a/5466478
source ${{HOME}}\/.bashrc
conda activate multivariate_ir

# load param file
HPARAMS_FILE={hyperparams_file}

# echo run info
echo "SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR
echo "SLURM_JOB_ID"=$SLURM_JOB_ID
echo "SLURM_JOB_NAME"=$SLURM_JOB_NAME
echo $SLURM_SUBMIT_DIR

# Start the experiment.
{cmd}
