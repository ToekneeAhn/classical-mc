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

#initial spin configuration (optional)
#spins = spins_initial_pyro(N) 

spins_r, energies_r, meas_r, accept_r = parallel_temper(r, replica_exchange_rate, N_therm, N_det, probe_rate, overrelax_rate, Ts, Js, h, N, S)

#swap_rate = 2*replica_exchange_rate*accept_r/(N_therm+N_det)
print("rank: ", r, " swapped ", sum(accept_r), " times.")

#gather_accepts = MPI.Gather(accept_r, comm, root=0)
#gather_meas = MPI.Gather(meas_r, comm, root=0)

#writes measurements to a file
obs_path = replace(pwd(),"\\"=>"/")*"/pt_out/obs_h$(h_index)_$(r).h5"
write_observables(obs_path, Dict("avg_spin"=>meas_r, "energy_per_site"=>energies_r[end]))

if r == 0
    #=
    fid = h5open("swap_history.h5", "w") #w for write, r for read
    fid["swaps"] = gather_accepts #reshape in python since it's kinda weird here
    close(fid)
    =#
    
    #writes parameters to a file
    params=Dict("Js"=>Js, "spin_length"=>S, "h"=>h, 
    "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate,
    "overrelax_rate"=>overrelax_rate, "replica_exchange_rate"=>replica_exchange_rate)
    params_path = replace(pwd(),"\\"=>"/")*"/pt_out/params_h$(h_index).h5"
    write_params(params_path, params)
end


