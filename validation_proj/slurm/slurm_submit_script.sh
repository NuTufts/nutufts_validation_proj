#!/bin/bash

#SBATCH --job-name=mlreco_iference_validation
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-9
#SBATCH --partition=wongjiradlab
#SBATCH --gres=gpu:p100:1

CONTAINER=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
RUN_SCRIPT=/cluster/home/nstieg01/nutufts_validation_proj/validation_proj/slurm/slurm_run_script.sh

module load singularity/3.5.3

singularity exec --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp ${CONTAINER} bash -c "source $RUN_SCRIPT"
