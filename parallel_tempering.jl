using MPI, LinearAlgebra, Printf
using BinningAnalysis: unbinned_tau

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")

mc_params = ARGS[1]
N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate = [parse(Int64, num) for num in split(mc_params,",")]
N = parse(Int64, ARGS[2])
S = parse(Float64, ARGS[3])
Js = ARGS[4]
Js = [parse(Float64, num) for num in split(Js,",")]
delta_12 = Js[5:6]
h_direction = ARGS[5] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
h_direction /= norm(h_direction)
h_sweep_args = ARGS[6]
h_min, h_max = [parse(Float64, num) for num in split(h_sweep_args,",")]
N_h = parse(Int64, ARGS[7])
Ts = ARGS[8]
T_min, T_max = [parse(Float64, num) for num in split(Ts,",")]
results_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[9]
file_prefix = ARGS[10]
save_dir = ARGS[11]
disorder_strength = parse(Float64, ARGS[12])
disorder_seed = [parse(Int64, ARGS[13])]
h_index = parse(Int64, ARGS[14])

h_sweep = range(h_min, h_max, N_h)
h = h_sweep[h_index]*h_direction

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
r = MPI.Comm_rank(comm)

#do a broadcast to ensure all replicas have the same disorder configuration
if disorder_seed[1] == 0
    disorder_seed = [rand(1:10^9)]
    MPI.Bcast!(disorder_seed, root=0, comm)
end
disorder_seed = disorder_seed[1]

# create geometric progression of temperatures ranks
Ts = exp10.(range(log10(T_min), stop=log10(T_max), length=comm_size)) 

#create inverse linear temperature ranks
#Ts = 1.0 ./ (1/T_max .+ (1/T_min - 1/T_max)/(comm_size-1) * Vector(range(comm_size-1,0,comm_size)))

N_sites = 4*N^3
spins_r = spins_initial_pyro(N, S) 
H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, disorder_seed)
system = SpinSystem(spins_r, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs, r, "none") 

energies_r, accept_metropolis_r, accept_swap_r, flow_r = parallel_temper!(simulation, r, Ts)
gather_accept_metropolis = MPI.Gather(accept_metropolis_r[1], comm, root=0)
gather_accept_swap = MPI.Gather(accept_swap_r[1], comm, root=0)
gather_tau = MPI.Gather(unbinned_tau(energies_r[N_therm:end]), comm, root=0)
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
    @printf("max autocorrelation time: %.0f \n\n", maximum(gather_tau))
    
    #makes save directory if it doesn't exist
    if !isdir(results_dir)
        mkdir(results_dir)
    end
end

MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname=file_prefix*"$(h_index)_$(r).h5"
write_observables(joinpath(results_dir,fname), simulation)

#collect results when sweep finished
if h_index == N_h
    MPI.Barrier(comm) #barrier in case 
    if r == 0
        if !isdir(save_dir)
            mkdir(save_dir)
        end
        collect_hsweep(results_dir, file_prefix, save_dir, system, params, Ts, h_direction, Vector(h_sweep), disorder_seed)
    end
end

