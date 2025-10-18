#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --ntasks=40
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name sim_anneal_higherT
#SBATCH -o ../slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#-------------------------------------------
echo "Running job in `pwd`"
echo "Starting run at: `date`"
#-------------------------------------------

module load julia/1.11.3
export PMIX_MCA_psec=native

Js="1.0,0.02,0.05,0.0" #comma separated Jzz,Jpm,Jpmpm,Jzpm in K
h_direction="1,1,1" #comma separated h field direction e.g. "1,1,1"
h_direction_nospace="${h_direction//,/}" #h_dir with commas removed
file_prefix="simanneal_adarsh_params_L8_highT_${h_direction_nospace}_h" 
disorder_strength="0.0" #in K"
disorder_seed="123456"

mc_params="200000,10,1000000" #comma separated N_therm, overrelax_rate, N_det
N_uc="8"
S="0.5"

h_sweep_args="2.0,4.0" #comma separated h_min,h_max
N_h=$SLURM_NTASKS
Ts="0.0001,1.0" #comma separated T_min,T_max
results_dir="../pt_out" #save directory for the individual rank and h data
save_dir="../pt_collect" #save directory for the collected data

mpiexec -n $N_h julia ~/classical-mc/sim_anneal_runner.jl $mc_params $N_uc $S $Js $h_direction $h_sweep_args $N_h $Ts $results_dir $file_prefix $save_dir $disorder_strength $disorder_seed