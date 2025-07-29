using MPI
include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")
include("input_file.jl")

#first command line argument is for save directory
save_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[1]*"/"
h_direction = ARGS[2] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
#h_dir as a string with no spaces
save_prefix = string(h_direction...) 
h_direction /= norm(h_direction)
seed=123

if length(ARGS) == 3
    #convert h_dir to a vector
    h_index = parse(Int64, ARGS[3])
    h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
    #h_dir as a string with no spaces
    save_prefix = string(h_direction...) 
    h_direction /= norm(h_direction)
    
    #h field
    h = h_sweep[h_index]*h_direction
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
N_sites = 4*N^3
disorder_strength = 3.132 #in K
H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, disorder_strength, N_sites, seed)
system = SpinSystem(spins_r, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs) #T argument is needed for calculating observables
energies_r, accept_r = parallel_temper!(simulation, r, Ts)

gather_accepts = MPI.Gather(accept_r[1], comm, root=0)
gather_z1 = MPI.Gather(zeeman[1], comm, root=0)

if r == 0
    #=
    fid = h5open("swap_history.h5", "w") #w for write, r for read
    fid["swaps"] = gather_accepts #reshape in python since it's kinda weird here
    close(fid)
    =#
    for rr in 1:comm_size
        println("rank ", rr, " swapped ", gather_accepts[rr], " times.")
    end
    println("rank 0 specific heat:", specific_heat(simulation))
    println("rank 0 susceptibility:", susceptibility(simulation))    
    println("random field at site 1: ", gather_z1)
    #makes save directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end

    #writes everything except measurements to a file
    fname_params = save_prefix*"_simulation_params_h$(h_index).h5"
    write_all(save_dir*fname_params, simulation)

    #fname="energies_0.h5"
    #write_single(save_dir*fname, energies_r, "energies")
end


MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname=save_prefix*"_obs_h$(h_index)_$(r).h5"
write_observables(save_dir*fname, simulation)