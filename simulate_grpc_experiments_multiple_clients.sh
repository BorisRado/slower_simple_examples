#!/bin/bash -l

#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=30:00
#SBATCH --out=logs/multiple_clients.log

# appropriately activate the environment
mamba activate test


for configuration in streaming u_shaped plain future; do
for server_config in \
    "common_server_model=True process_clients_as_batch=True" \
    "common_server_model=True process_clients_as_batch=False" \
    "common_server_model=False process_clients_as_batch=False"; do
    if [[ "$configuration" == "u_shaped" && "$server_config" == "common_server_model=True process_clients_as_batch=False" ]]; then
        continue
    fi
    echo " ===== CONFIGURATION ${configuration} ===== "
    srun --ntasks=1 python run_server.py configuration=$configuration $server_config num_clients=$((SLURM_NTASKS-1)) &
    sleep 10  # give some time to the server to start up
    srun --ntasks=$((SLURM_NTASKS-1)) python run_client.py configuration=$configuration &
    wait
    echo " ===== END OF CONFIGURATION ${configuration} ===== "
    echo ""
    echo ""
    echo ""

done
done
