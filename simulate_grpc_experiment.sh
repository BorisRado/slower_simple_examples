#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=3:00
#SBATCH --output=out1.log
#SBATCH --partition=debug


source ../.venv/old_torch/bin/activate
export PYTHONPATH=$PYTHONPATH:../slower

for configuration in stream plain u; do

    srun --cpus-per-task=8 --ntasks=1 --mem-per-cpu=1G python -u run_server.py configuration=$configuration &
    sleep 10
    srun --cpus-per-task=8 --ntasks=1 --mem-per-cpu=1G python -u run_client.py configuration=$configuration &
    wait

done
