#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=15:00

# appropriately activate the environment
source ../.venv/slower_test/bin/activate

for configuration in streaming u_shaped plain; do

    echo "CONFIGURATION ${configuration}"
    srun --cpus-per-task=8 --ntasks=1 --mem-per-cpu=1G python3 run_server.py configuration=$configuration &
    sleep 10  # give some time to the server to start up
    srun --cpus-per-task=8 --ntasks=1 --mem-per-cpu=1G python3 run_client.py configuration=$configuration &
    wait

done
