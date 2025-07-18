#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --ntasks=30
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name newparams_111
#SBATCH -o ../slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#-------------------------------------------
echo "Running job in `pwd`"
echo "Starting run at: `date`"
#-------------------------------------------

module load julia/1.11.3
export PMIX_MCA_psec=native

Js="2.963,-0.5534,-0.00396,0.0" #comma separated Jzz,Jpm,Jpmpm,Jzpm
h_direction="1,1,1" #comma separated h field direction e.g. "1,1,1"
h_direction_nospace="${h_direction//,/}" #h_dir with commas removed
file_prefix="csi_newparams_N8_${h_direction_nospace}_h" 

mc_params="200000,10,100000,200,50" #comma separated N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate
N_uc="8"
S="0.5"

h_sweep_args="0.0,12.0" #comma separated h_min,h_max
N_h="40" 
Ts="0.06,0.4" #comma separated T_min,T_max
results_dir="../pt_out" #save directory for the individual rank and h data
save_dir="../pt_collect" #save directory for the collected data

for ((i=1; i<=N_h; i++));
do
    mpiexec -n $SLURM_NTASKS julia parallel_tempering.jl $mc_params $N_uc $S $Js $h_direction $h_sweep_args $N_h $Ts $results_dir $file_prefix $save_dir $i
done
