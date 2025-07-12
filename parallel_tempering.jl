using MPI, LinearAlgebra

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")

#$mc_params $N_uc $S $Js $h_direction $h_sweep_args $N_h $Ts $save_folder $j

mc_params = ARGS[1]
N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate = [parse(Int64, num) for num in split(mc_params,",")]
N = parse(Int64, ARGS[2])
S = parse(Float64, ARGS[3])
Js = ARGS[4]
Js = [parse(Float64, num) for num in split(Js,",")]
h_direction = ARGS[5] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
save_prefix = string(h_direction...) 
h_direction /= norm(h_direction)
h_sweep_args = ARGS[6]
h_min, h_max = [parse(Float64, num) for num in split(h_sweep_args,",")]
N_h = parse(Int64, ARGS[7])
Ts = ARGS[8]
T_min, T_max = [parse(Float64, num) for num in split(Ts,",")]
save_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[9]*"/"
save_prefix = ARGS[10]
h_index = parse(Int64, ARGS[11])

h_sweep = range(h_min, h_max, N_h)
h = h_sweep[h_index]*h_direction

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
r = MPI.Comm_rank(comm)

# create equal logarithmically spaced temperatures
Ts = exp10.(range(log10(T_min), stop=log10(T_max), length=comm_size)) 

#initial spin configuration 
spins_r = spins_initial_pyro(N, S) 
system = SpinSystem(spins_r, S, N, 4*N^3, Js, h)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs) #T argument is kind of useless here
energies_r, accept_r = parallel_temper!(simulation, r, Ts)

gather_accepts = MPI.Gather(accept_r[1], comm, root=0)

if r == 0
    for rr in 1:comm_size
        println("rank ", rr, " swapped ", gather_accepts[rr], " times.")
    end
    
    #makes save directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    #writes everything except measurements to a file
    fname_params = save_prefix*"_simulation_params_h$(h_index).h5"
    write_all(save_dir*fname_params, simulation)

end


MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname=save_prefix*"$(h_index)_$(r).h5"
write_observables(save_dir*fname, simulation)