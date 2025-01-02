#!/bin/bash -l

#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=30:00
#SBATCH --out=logs/logs.txt

# appropriately activate the environment
mamba activate test


for configuration in streaming u_shaped plain future; do

    echo " ===== CONFIGURATION ${configuration} ===== "
    srun --ntasks=1 python run_server.py configuration=$configuration &
    sleep 10  # give some time to the server to start up
    srun --ntasks=$((SLURM_NTASKS-1)) python run_client.py configuration=$configuration &
    wait
    echo " ===== END OF CONFIGURATION ${configuration} ===== "
    echo ""
    echo ""
    echo ""

done
