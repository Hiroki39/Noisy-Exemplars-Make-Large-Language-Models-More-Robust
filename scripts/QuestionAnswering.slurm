#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 08:00:00
#SBATCH --job-name QuestionAnswering
#SBATCH --output "/gpfs/data/oermannlab/users/hz2212/DS-GA 3001-003/slurm-logs/QuestionAnswering-%J.log" 

# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 
# module add anaconda2/gpu/5.2.0

module load anaconda3/cpu/5.3.1
source activate ds3001-003

export PYTHONPATH=""

if [ "$4" = "None" ]
then
    shots=""
else
    shots="--shots $4"
fi

if [ "$5" = "None" ]
then
    perturb=""
else
    perturb="--perturb $5"
fi

if [ "$6" = "True" ]
then
    perturb_exemplar="--perturb_exemplar"
else
    perturb_exemplar="--no-perturb_exemplar"
fi

cd "$HOME/DS-GA 3001-003"

python evaluate.py --model $1 --dataset $2 --prompt $3 $shots $perturb $perturb_exemplar