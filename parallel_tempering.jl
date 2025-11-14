using MPI, LinearAlgebra, Printf, YAML
using BinningAnalysis: unbinned_tau

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")

params = YAML.load_file(ARGS[1]) 
N = params["N_uc"]
S = params["S"]
Js = params["Js"]
h_theta = params["h_theta"]
h_sweep_args = params["h_sweep_args"]
N_h = params["N_h"]
delta_12 = params["delta_12"]
breaking_field = params["breaking_field"]
disorder_strength = params["disorder_strength"]
disorder_seed = params["disorder_seed"]

params_pt = params["parallel_temper"]
N_therm = params_pt["mc_params"]["N_therm"]
overrelax_rate = params_pt["mc_params"]["overrelax_rate"]
N_meas = params_pt["mc_params"]["N_meas"]
probe_rate = params_pt["mc_params"]["probe_rate"]
replica_exchange_rate = params_pt["mc_params"]["replica_exchange_rate"]
optimize_temperature_rate = params_pt["mc_params"]["optimize_temperature_rate"]
T_args = params_pt["T_args"]
load_configs = params_pt["load_configs"]
load_configs_prefix = params_pt["load_configs_prefix"]
results_dir = params_pt["results_dir"]
save_dir = params_pt["save_dir"]
file_prefix = params_pt["file_prefix"]

h_direction = [1.0,1.0,1.0]/sqrt(3) .* cos(h_theta * pi/180) .+ [1.0,-1.0,0.0]/sqrt(2) .* sin(h_theta * pi/180)
h_min, h_max = h_sweep_args
h_sweep = range(h_min, h_max, N_h)
h_index = parse(Int64, ARGS[2])
h = h_sweep[h_index]*h_direction

T_min, T_max = T_args

#=
mc_params = ARGS[1]
N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate = [parse(Int64, num) for num in split(mc_params,",")]
N = parse(Int64, ARGS[2])
S = parse(Float64, ARGS[3])
Js = ARGS[4]
Js = [parse(Float64, num) for num in split(Js,",")]
delta_12 = Js[5:6]
#=
h_direction = ARGS[5] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
h_direction /= norm(h_direction)
=#
h_theta = parse(Float64, ARGS[5]) #angle from [111] direction towards [1-10] in degrees
h_direction = [1.0,1.0,1.0]/sqrt(3) .* cos(h_theta * pi/180) .+ [1.0,-1.0,0.0]/sqrt(2) .* sin(h_theta * pi/180)
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

if length(ARGS) >= 15
    load_configs_path = ARGS[15]*"$(h_index).h5"
else
    load_configs_path = "none"
end
=#


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

#initial spin configuration 
if load_configs == true
    #load from hdf5 file
    spins_r, Ts = read_configuration_hdf5(load_configs_prefix*"$(h_index).h5", r+1)
else
    spins_r = spins_initial_pyro(N, S) 
    # create equal logarithmically spaced temperatures
    Ts = exp10.(range(log10(T_min), stop=log10(T_max), length=comm_size)) 
end

N_sites = 4*N^3
@assert size(spins_r, 2) == N_sites "Loaded spin configuration has incorrect number of sites"
@assert length(Ts) == comm_size "Loaded temperature array has incorrect length"

H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, disorder_seed, breaking_field)
system = SpinSystem(spins_r, S, N, N_sites, Js, h, delta_12, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, -1, overrelax_rate, N_meas, probe_rate, replica_exchange_rate, optimize_temperature_rate)
obs = Observables()
simulation = Simulation(system, Ts[r+1], params, obs, r, "none") 

energies_r, accept_metropolis_r, accept_swap_r, flow_r = parallel_temper!(simulation, r, Ts, comm, comm_size)
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

#writes measurements to a file
fname=file_prefix*"$(h_index)_$(r).h5"
write_observables(joinpath(results_dir,fname), simulation)
MPI.Barrier(comm) #barrier in case 

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

