#!/bin/bash

params_file="params.yaml"
N_theta=181
theta_min=-90
theta_max=90

for i in $(seq 0 $((N_theta-1))); do
    theta=$(awk -v i="$i" -v min="$theta_min" -v max="$theta_max" -v N="$N_theta" 'BEGIN{printf "%.1f", min + i*(max-min)/(N-1)}')

    theta_prefix="Jzz=2.0_piflux_hhltheta=${theta}_h"
    export theta theta_prefix
    yq -i '
      .h_theta = env(theta) |
      .sim_anneal.job.job_name = "theta=" + strenv(theta) |
      .sim_anneal.save_dir = "../theta_sweep" |
      .sim_anneal.file_prefix = strenv(theta_prefix)
    ' $params_file

    bash runner.sh sa
    sleep 1
done
