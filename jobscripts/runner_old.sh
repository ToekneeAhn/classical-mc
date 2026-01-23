#!/bin/bash

#copies stuff into scratch and then runs sbatch
dirname=${PWD##*/}
dirname=${dirname:-/} 

rsync -av --progress --exclude=.git $PWD $SCRATCH
cd $SCRATCH/$dirname

if [[ $1 = "pt" ]]
then
  echo "Running parallel tempering with new jobscript in `pwd`"
  sbatch mpi_jobscript_old.sh
elif [[ $1 = "sa" ]]
then
  echo "Running simulated annealing with new jobscript in `pwd`"
  sbatch sim_anneal_jobscript_old.sh
else
  echo "No job started" 
fi