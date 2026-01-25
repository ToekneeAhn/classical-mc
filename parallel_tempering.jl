using MPI, BinningAnalysis, Printf
using BinningAnalysis: unbinned_tau
include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")
include("input_file.jl")

#first command line argument is for save directory
save_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[1]*"/"
h_direction = ARGS[2] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
h_index = 1
N_h = 1
#h_dir as a string with no spaces
file_prefix = string(h_direction...)*"_h" 
h_direction /= norm(h_direction)
seed=123

if length(ARGS) == 3
    load_configs_path = ARGS[3]
else
    load_configs_path = "none"
end

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
r = MPI.Comm_rank(comm)

#initial spin configuration 
if load_configs_path != "none"
    #load from hdf5 file
    spins_r, Ts = read_configuration_hdf5(load_configs_path, r+1)
else
    spins_r = spins_initial_pyro(N, S) 
    # create equal logarithmically spaced temperatures
    Ts = exp10.(range(log10(T_min), stop=log10(T_max), length=comm_size)) 
end

N_sites = 4*N^3
@assert size(spins_r, 2) == N_sites "Loaded spin configuration has incorrect number of sites"
@assert length(Ts) == comm_size "Loaded temperature array has incorrect length"

neighbours = neighbours_all(N, N_sites)
H_bilinear = H_bilinear_all(Js, N, N_sites)

cubic_sites = cubic_sites_all(N, N_sites)
unique_triplets, unique_H_cubic_vals = unique_cubic_triplets(K, N, N_sites)
pairs_i, pairs_j, pairs_k = cubic_pairs_split_all(cubic_sites, N_sites)
H_cubic_sparse = cubic_tensors_sparse_all(K, N, N_sites)

zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, 0)

if include_cubic
    system = SpinSystem(spins_r, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, K, cubic_sites, H_cubic_sparse, unique_triplets, unique_H_cubic_vals, pairs_i, pairs_j, pairs_k, zeeman)
else
    system = SpinSystem(spins_r, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman)
end

params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs, r, "none") #T argument is needed for calculating observables
energies_r, accept_metropolis_r, accept_swap_r, flow_r = parallel_temper!(simulation, r, Ts, comm, comm_size)
autocorr_time = unbinned_tau(energies_r)

gather_accept_metropolis = MPI.Gather(accept_metropolis_r[1], comm, root=0)
gather_accept_swap = MPI.Gather(accept_swap_r[1], comm, root=0)
gather_tau = MPI.Gather(autocorr_time, comm, root=0)
gather_flow = MPI.Gather(flow_r, comm, root=0)

if r == 0
    total_swaps = (N_therm + N_meas)/replica_exchange_rate * ones(comm_size)
    #edge temperature ranks only swap when swap_type=0, i.e. half the time
    total_swaps[1] /= 2
    total_swaps[end] /= 2
    total_metropolis = N_sites * (N_therm + N_meas)/overrelax_rate
    
    for rr in 1:comm_size
        @printf("rank %d swap rate: %.1f%% \t|\t metropolis acceptance rate: %.1f%%\n", rr, 100 * gather_accept_swap[rr]/total_swaps[rr], 100 * gather_accept_metropolis[rr]/total_metropolis)
        @printf("rank %d autocorr time: %.0f \t|\t flow: %.2f \t|\t ideal flow: %.2f\n", rr, gather_tau[rr], gather_flow[rr], 1-(rr-1)/(comm_size-1))
    end
     
    println("\nLowest temperature rank data")
    println("Energy per site: ", round(E_pyro(system)/N_sites, digits=6))
    S_avg = spin_expec(spins_r, N)
    println("Average spin per sublattice:")
    println("Sublattice 0: ", round.(S_avg[:,1], digits=4))
    println("Sublattice 1: ", round.(S_avg[:,2], digits=4))
    println("Sublattice 2: ", round.(S_avg[:,3], digits=4))
    println("Sublattice 3: ", round.(S_avg[:,4], digits=4))

    S_avg_global = [local_to_global(S_avg[:,i], i) for i in 1:4]
    println("Average spin in global frame per sublattice:")
    println("Sublattice 0: ", round.(S_avg_global[1], digits=4))
    println("Sublattice 1: ", round.(S_avg_global[2], digits=4))
    println("Sublattice 2: ", round.(S_avg_global[3], digits=4))
    println("Sublattice 3: ", round.(S_avg_global[4], digits=4))

    #makes save directory if it doesn't exist
    if !isdir(save_dir)
        mkdir(save_dir)
    end
end


MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname=file_prefix*"$(h_index)_$(r).h5"
write_observables(save_dir*fname, simulation, spins_r)

#collect results when sweep finished
if h_index == N_h
    if r == 0
        collect_hsweep(save_dir, file_prefix, save_dir, system, params, Ts, h_direction, [norm(h)], seed)
    end
end