#!/bin/bash

#copies stuff into scratch and then runs sbatch

dirname=${PWD##*/}
dirname=${dirname:-/} 

#cp -r $PWD $SCRATCH
rsync -av --progress --exclude=.git $PWD $SCRATCH
cd $SCRATCH/$dirname

echo "Running job in `pwd`"
echo "Starting run at: `date`"

sbatch mpi_jobscript.sh
