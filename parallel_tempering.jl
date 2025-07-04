using MPI

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")
include("input_file.jl")

#first command line argument is for save directory
save_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[1]*"/"

if length(ARGS) == 2
    h_index = parse(Int64, ARGS[2])
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
system = SpinSystem(spins_r, S, N, 4*N^3, Js, h)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs) #T argument is kind of useless here
energies_r, accept_r = parallel_temper!(simulation, r, Ts)

gather_accepts = MPI.Gather(accept_r[1], comm, root=0)

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

    #writes everything except measurements to a file
    fname_params = "simulation_params_h$(h_index).h5"
    write_all(save_dir*fname_params, simulation)

    #fname="energies_0.h5"
    #write_single(save_dir*fname, energies_r, "energies")
end

MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname="obs_h$(h_index)_$(r).h5"
write_observables(save_dir*fname, simulation)
