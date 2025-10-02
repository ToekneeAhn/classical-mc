using MPI, BinningAnalysis, Printf
using BinningAnalysis: unbinned_tau
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
disorder_strength = 0.0 #in K
quad_strength = [0.0,0.0]
H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, quad_strength, disorder_strength, N_sites, seed)
system = SpinSystem(spins_r, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs, r, "none") #T argument is needed for calculating observables
energies_r, accept_metropolis_r, accept_swap_r, flow_r = parallel_temper!(simulation, r, Ts)
autocorr_time = unbinned_tau(energies_r)

gather_accept_metropolis = MPI.Gather(accept_metropolis_r[1], comm, root=0)
gather_accept_swap = MPI.Gather(accept_swap_r[1], comm, root=0)
gather_tau = MPI.Gather(autocorr_time, comm, root=0)
gather_flow = MPI.Gather(flow_r, comm, root=0)

if r == 0
    total_swaps = 2*(N_therm + N_meas)/replica_exchange_rate * ones(comm_size)
    #edge temperature ranks only swap when swap_type=0, i.e. half the time
    total_swaps[1] /= 2
    total_swaps[end] /= 2
    total_metropolis = N_sites * (N_therm + N_meas)/overrelax_rate
    
    println("final temperatures: ", Ts)
    for rr in 1:comm_size
        @printf("rank %d swap rate: %.1f%% \t|\t metropolis acceptance rate: %.1f%%\n", rr, 100 * gather_accept_swap[rr]/total_swaps[rr], 100 * gather_accept_metropolis[rr]/total_metropolis)
        @printf("rank %d autocorr time: %.0f \t|\t flow: %.2f \t|\t ideal flow: %.2f\n", rr, gather_tau[rr], gather_flow[rr], 1-(rr-1)/(comm_size-1))
    end
    
    #=
    println("dsdt ", dSdT(simulation)[1])
    println("dsdt err ", dSdT(simulation)[2])
    println("Js=", simulation.spin_system.Js)
    println("h=", h)
    println("rank 0 specific heat:", specific_heat(simulation))
    println("rank 0 susceptibility:", susceptibility(simulation))  
    =#

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
fname=save_prefix*"_obs_h$(h_index)_$(r).h5"
write_observables(save_dir*fname, simulation)