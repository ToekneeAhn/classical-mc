using MPI

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")
include("input_file.jl")

if length(ARGS) == 1
    h_index = parse(Int64, ARGS[1])
    h = h_sweep[h_index]*[1,1,1]/sqrt(3)
else
    h_index = "single"
end

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
r = MPI.Comm_rank(comm)

# create equal logarithmically spaced temperatures
Ts = exp10.(range(log10(T_min), stop=log10(T_max), length=comm_size)) 

#initial spin configuration 
spins_r = spins_initial_pyro(N, S) 

energies_r, meas_r, err_r, accept_r = parallel_temper(r, replica_exchange_rate, N_therm, N_det, probe_rate, overrelax_rate, Ts, Js, h, N, S, spins_r)

gather_accepts = MPI.Gather(accept_r[1], comm, root=0)

#folder where results are saved
save_dir = replace(pwd(),"\\"=>"/")*"/pt_out/"

if r == 0
    #=
    fid = h5open("swap_history.h5", "w") #w for write, r for read
    fid["swaps"] = gather_accepts #reshape in python since it's kinda weird here
    close(fid)
    =#
    for rr in 1:comm_size
        println("rank ", rr, " swapped ", gather_accepts[rr], " times.")
    end
    
    #makes save directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    #writes parameters to a file
    params=Dict("Js"=>collect(Js), "spin_length"=>S, "h"=>h, 
    "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate,
    "overrelax_rate"=>overrelax_rate, "replica_exchange_rate"=>replica_exchange_rate)
    fname_params = "params_h$(h_index).h5"
    write_params(save_dir*fname_params, params)
end

MPI.Barrier(comm)

#writes measurements to a file
fname="obs_h$(h_index)_$(r).h5"
write_observables(save_dir*fname, Dict("avg_spin"=>meas_r, "avg_spin_err"=>err_r, "energy_per_site"=>energies_r[end]))