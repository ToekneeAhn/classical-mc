#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --ntasks=30
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name newparams_100
#SBATCH -o ../slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#

module load julia/1.11.3

Js="2.963,-0.5534,-0.00396,0.0" #comma separated Jzz,Jpm,Jpmpm,Jzpm
h_dir="1,0,0" #comma separated h field direction e.g. "1,1,1"
h_dir_nospace="${h_dir//,/}" #h_dir with commas removed
file_prefix="csi_newparams_${h_dir_nospace}_h" 

mc_params="200000,10,100000,200,50" #comma separated N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate
N_uc="6"
S="0.5"

h_sweep_args="0.0,12.0" #comma separated h_min,h_max
N_h="60" 
Ts="0.06,0.4" #comma separated T_min,T_max
results_dir="../pt_out" #save directory for the individual rank and h data
save_dir="../pt_collect" #save directory for the collected data

julia mpi_runner.jl $SLURM_NTASKS $mc_params $N_uc $S $Js $h_dir $h_sweep_args $N_h $Ts $results_dir $file_prefix $save_dir
