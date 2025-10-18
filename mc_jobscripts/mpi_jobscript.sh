#!/bin/bash
#SBATCH --account=rrg-ybkim
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem-per-cpu=1G
#SBATCH --time=8:00:00
#SBATCH --job-name L8_newfit_scalerun
#SBATCH -o ../slurm_out/slurm-%j.out 
#SBATCH --mail-user=t.an@mail.utoronto.ca
#SBATCH --mail-type=ALL
#-------------------------------------------
echo "Running job $SLURM_JOBNAME in `pwd`"
echo "Starting run at: `date`"
#-------------------------------------------

module load julia/1.11.3
export PMIX_MCA_psec=native

Js="6.37,-2.28,0.109,0.0,-0.0,-0.0" #comma separated Jzz,Jpm,Jpmpm,Jzpm in K and also delta_1, delta_2
h_direction="1,1,1" #comma separated h field direction e.g. "1,1,1"
h_direction_nospace="${h_direction//,/}" #h_dir with commas removed

mc_params="1000000,10,500000,500,50,10000000" #comma separated N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate
N_uc="8"
S="0.5"

disorder_strength="0.379" #in K"
disorder_seed="0"

h_sweep_args="5.55,11.1" #comma separated h_min,h_max
N_h="50" 
Ts="0.06,1.0" #comma separated T_min,T_max

file_prefix="L${N_uc}_Jzz6_disorderfit_scalerun_${h_direction_nospace}_h" 
results_dir="../pt_out" #save directory for the individual rank and h data
save_dir="../pt_collect" #save directory for the collected data

for ((i=1; i<=N_h; i++));
do
    mpiexec -n $SLURM_NTASKS julia ~/classical-mc/parallel_tempering.jl $mc_params $N_uc $S $Js $h_direction $h_sweep_args $N_h $Ts $results_dir $file_prefix $save_dir $disorder_strength $disorder_seed $i
done

cd $results_dir
rm $file_prefix* #clean up