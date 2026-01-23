#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --nodes=1
#SBATCH --output=../slurm_out/slurm_%j.out
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#-------------------------------------------
echo "Running job in `pwd`"
echo "Starting run at: `date`"
#-------------------------------------------

module load julia/1.11.3
export PMIX_MCA_psec=native

params_file=$1
results_dir=$(yq -r '.sim_anneal.results_dir' $params_file)
file_prefix=$(yq -r '.sim_anneal.file_prefix' $params_file)

srun julia ~/classical-mc/sim_anneal_runner.jl $params_file

cd $results_dir
rm $file_prefix* #clean up
