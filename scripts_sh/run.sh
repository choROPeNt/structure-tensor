#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=4:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=16                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem=256G                             # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "structure-tensor"                           # job name
#SBATCH --output=structure-tensor-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_autoshear
##################################################################
##################################################################


module load release/23.04  GCC/12.2.0 Python/3.10.8 OpenMPI/4.1.4 CUDA/11.8.0


## Display GPUs
nvidia-smi

source .venv_struct/bin/activate
echo $1
python ./scripts/analyse.py --config $1

exit 0