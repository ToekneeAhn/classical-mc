#!/bin/sh
#SBATCH --account=rrg-ybkim
#SBATCH --ntasks=30
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name 111_pmfrg_xxz_Jzzpos
#SBATCH -o ./slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------# Loading module

module load julia/1.11.3

Js="8.0,-6.0,0.0,0.0" #comma separated Jzz,Jpm,Jpmpm,Jzpm
h_dir="1,1,1" #comma separated h field direction e.g. "1,1,1"
h_dir_nospace="${h_dir//,/}" #h_dir with commas removed
file_prefix="pmfrg_xxz_Jzzpos_${h_dir_nospace}_obs_h" 

mc_params="200000,10,100000,200,50" #comma separated N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate
N_uc="6"
S="0.5"

h_sweep_args="0.0,30.0" #comma separated h_min,h_max
N_h="60" 
Ts="0.06,0.4" #comma separated T_min,T_max
results_dir="pt_out" #save directory for the individual rank and h data
save_dir="pt_collect" #save directory for the collected data

julia mpi_runner.jl $SLURM_NTASKS $mc_params $N_uc $S $Js $h_dir $h_sweep_args $N_h $Ts $results_dir $file_prefix $save_dir
