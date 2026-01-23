#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --nodes=1
#SBATCH --output=/scratch/antony/slurm_out/slurm_%j.out
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#-------------------------------------------
echo "Running job $SLURM_JOBNAME in `pwd`"
echo "Starting run at: `date`"
#-------------------------------------------

module purge
module load StdEnv/2023
module load julia/1.11.3
export PMIX_MCA_psec=native

params_file=$1
results_dir=$(yq -r '.parallel_temper.results_dir' $params_file)
file_prefix=$(yq -r '.parallel_temper.file_prefix' $params_file)
N_h=$(yq '.N_h' $params_file)

for ((i=1; i<=N_h; i++));
do
    mpirun julia ~/classical-mc/parallel_tempering.jl $params_file $i
done

cd $results_dir
rm $file_prefix* #clean up
