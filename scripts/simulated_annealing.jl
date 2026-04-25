using MPI, LinearAlgebra, YAML, ArgParse
using PyroClassicalMC
include(joinpath(@__DIR__, "config_utils.jl"))

s = ArgParseSettings()
@add_arg_table s begin
    "--params_file"
        help = "Path to the YAML parameter file"
        arg_type = String
        required = true
end

parsed_args = parse_args(s)
cfg = load_config(parsed_args["params_file"], :sim_anneal)

N = cfg.N
S = cfg.S
Js = cfg.Js
include_cubic = cfg.include_cubic
K = cfg.K
h_sweep_args = cfg.h_sweep_args
N_h = cfg.N_h
delta_12 = cfg.delta_12
disorder_strength = cfg.disorder_strength
disorder_seed = cfg.disorder_seed
N_therm = cfg.N_therm
overrelax_rate = cfg.overrelax_rate
N_det = cfg.N_det
T_args = cfg.T_args
save_configs = cfg.save_configs
results_dir = cfg.results_dir
save_dir = cfg.save_dir
file_prefix = cfg.file_prefix
h_direction = cfg.h_direction

h_min, h_max = h_sweep_args
h_sweep = range(h_min, h_max, N_h)
#h_index defined below by MPI rank

T_f, T_i = T_args

if save_configs
    T_save_args = cfg.T_save_args
    T_save_min, T_save_max = T_save_args
    N_save = cfg.N_save
    temp_save = exp10.(range(log10(T_save_min), stop=log10(T_save_max), length=N_save)) 
    save_configs_prefix = cfg.save_configs_prefix
else
    temp_save = []
end

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
@assert comm_size == N_h "Number of ranks does not match the number of h points"
r = MPI.Comm_rank(comm)
h_index = r + 1

h_sweep = range(h_min, h_max, N_h)
h = h_sweep[h_index]*h_direction

#do a broadcast to ensure all replicas have the same disorder configuration
disorder_seed_buffer = normalize_disorder_seed(disorder_seed)
if disorder_seed_buffer[1] == 0
    if r == 0
        disorder_seed_buffer[1] = rand(1:10^9)
    end
    MPI.Bcast!(disorder_seed_buffer, root=0, comm)
end
disorder_seed = disorder_seed_buffer[1]

N_sites = 4*N^3
#random initial configuration
spins = spins_initial_pyro(N, S)

neighbours = neighbours_all(N, N_sites)
H_bilinear = H_bilinear_all(Js, N, N_sites)

cubic_sites = cubic_sites_all(N, N_sites)
unique_triplets, unique_H_cubic_vals = unique_cubic_triplets(K, N, N_sites)
pairs_i, pairs_j, pairs_k = cubic_pairs_split_all(cubic_sites, N_sites)
H_cubic_sparse = cubic_tensors_sparse_all(K, N, N_sites)

zeeman = zeeman_field_random(h, Z_LOCAL, LOCAL_INTERACTIONS, delta_12, disorder_strength, N_sites, disorder_seed)

if include_cubic
    system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, K, cubic_sites, H_cubic_sparse, unique_triplets, unique_H_cubic_vals, pairs_i, pairs_j, pairs_k, zeeman)
    if r == 0
        println("Including cubic interactions with K = $(K)")
    end
else
    system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman)
    if r == 0
        println("Not including cubic interactions.")
    end
end

mc_params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1, -1)
simulation = Simulation(system, T_f, mc_params, Observables(), 0, "none")
config = RunConfig([T_f], h_direction, Vector(h_sweep), disorder_seed)

if r == 0    
    #makes save directories if they doesn't exist
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    if !isdir(save_dir)
        mkdir(save_dir)
    end
end

#simulated annealing with annealing schedule T = T_i*0.9^t
if save_configs
    #saves at specified temperatures during annealing
    _, configurations_save = sim_anneal!(simulation, t-> T_i * 0.9^t, temp_save, false)
    write_parameters(joinpath(save_dir, save_configs_prefix)*"$(h_index).h5", system, mc_params, config)
    write_collection_sim_anneal(joinpath(save_dir, save_configs_prefix)*"$(h_index).h5", configurations_save, temp_save)
else
    sim_anneal!(simulation, t-> T_i * 0.9^t, Float64[], r == 0 ? true : false)
end

#writes measurements to a file
file_append = "_h$(h_index)_0.h5"
parameters_path = joinpath(results_dir, file_prefix*"_parameters.h5")

write_observables(joinpath(results_dir, file_prefix*file_append), simulation)
MPI.Barrier(comm) #barrier in case 

#collect results about T_f when sweep finished
if r == 0
    write_parameters(parameters_path, system, mc_params, config)
    collect_hsweep(results_dir, file_prefix*"_h", save_dir, parameters_path)
end

