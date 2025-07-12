using MPI

n = parse(Int64, ARGS[1]) #number of replicas
mc_params = ARGS[2] #comma separated N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate
N_uc = ARGS[3]
S = ARGS[4]
Js = ARGS[5] #comma separated Jzz,Jpm,Jpmpm,Jzpm
h_direction = ARGS[6] #comma separated h field direction e.g. "1,1,1"
h_sweep_args = ARGS[7] #comma separated h_min,h_max
N_h = parse(Int64,ARGS[8])
Ts = ARGS[9] #comma separated T_min,T_max
results_dir = ARGS[10] 
file_prefix = ARGS[11]

#h sweep
for j in 1:N_h
    #run(`$(mpiexec()) -n $n julia parallel_tempering_2.jl $save_folder $h_direction $j`)
    run(`$(mpiexec()) -n $n julia parallel_tempering.jl $mc_params $N_uc $S $Js $h_direction $h_sweep_args $N_h $Ts $results_dir $file_prefix $j`)
end
