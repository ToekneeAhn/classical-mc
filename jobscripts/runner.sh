#!/bin/bash

#copies stuff into scratch and then runs sbatch
dirname=${PWD##*/}
dirname=${dirname:-/}

rsync -av --progress $PWD $SCRATCH
cd $SCRATCH/$dirname

NOW=$( date '+%F_%H:%M:%S' )
params_file="$SCRATCH/param_files/params_$NOW.yaml"
#moves parameters file to its own folder
rsync -av --progress params.yaml $params_file

if [[ $1 = "pt" ]]
then
  ntasks=$(yq '.parallel_temper.N_Ts' $params_file)
  #runtime=$(yq -r '.parallel_temper.job.time' $params_file)
  #mem=$(yq -r '.parallel_temper.job.mem_per_cpu' $params_file)
  #name=$(yq -r '.parallel_temper.job.job_name' $params_file)
  #submit=$(sbatch --ntasks=$ntasks --time=$runtime --mem-per-cpu=$mem --job-name=$name mpi_jobscript.sh $params_file)

  job_params=( $(yq '.parallel_temper.job[]' $params_file) )
  submit=$(sbatch --ntasks=$ntasks --time=${job_params[0]} --mem-per-cpu=${job_params[1]} --job-name=${job_params[2]} mpi_jobscript.sh $params_file | awk '{print $NF}')
  submit=( $submit )
  job_id=${submit[-1]}

  echo "Running parallel tempering jobname=${job_params[2]}, id=$job_id in `pwd`"
elif [[ $1 = "sa" ]]
then
  ntasks=$(yq '.N_h' $params_file)
  job_params=( $(yq '.sim_anneal.job[]' $params_file) )
  submit=$(sbatch --ntasks=$ntasks --time=${job_params[0]} --mem-per-cpu=${job_params[1]} --job-name=${job_params[2]} sim_anneal_jobscript.sh $params_file | awk '{print $NF}')

  submit=( $submit )
  job_id=${submit[-1]}

  echo "Running simulated annealing jobname=${job_params[2]}, id=$job_id in `pwd`"
elif [[ $1 = "sa_load_measure" ]]
then
  yq -i '.sim_anneal.save_configs = true' $params_file
  ntasks=$(yq '.N_h' $params_file) #simulated annealing: N_h tasks in parallel
  job_params=( $(yq '.sim_anneal.job[]' $params_file) )
  submit=$(sbatch --ntasks=$ntasks --time=${job_params[0]} --mem-per-cpu=${job_params[1]} --job-name=${job_params[2]} sim_anneal_jobscript.sh $params_file | awk '{print $NF}')

  submit=( $submit ) #get the number from "submitted batch job xyz"
  job_id=${submit[-1]}

  echo "Running simulated annealing jobname=${job_params[2]}, id=$job_id in `pwd`"

  yq -i '
    .parallel_temper.load_configs = true |
    .parallel_temper.mc_params.overrelax_rate = 1 |
    .parallel_temper.mc_params.replica_exchange_rate = 100000000
  ' $params_file

  save_path=$(yq '.sim_anneal.save_configs_prefix' $params_file) yq -i '.parallel_temper.load_configs_prefix = strenv(save_path)' $params_file

  ntasks=$(yq '.sim_anneal.N_save' $params_file) #"parallel tempering": N_save replicas (no swapping, only metropolis updates)
  job_params=( $(yq '.parallel_temper.job[]' $params_file) )
  sbatch --dependency=afterok:$job_id --ntasks=$ntasks --time=${job_params[0]} --mem-per-cpu=${job_params[1]} --job-name=${job_params[2]} mpi_jobscript.sh $params_file

  echo "Running parallel tempering jobname=${job_params[2]} in `pwd`"
else
  echo "No job started" 
fi
