#!/bin/sh
#SBATCH --account=rrg-ybkim
#SBATCH --ntasks=10
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name mpitest
#SBATCH -o ./slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------# Loading module

module load julia/1.11.3
julia mpi_runner.jl $SLURM_NTASKS
