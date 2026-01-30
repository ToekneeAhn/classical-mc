using MPI, LinearAlgebra, YAML, ArgParse

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--params_file"
        help = "Path to the YAML parameter file"
        arg_type = String
        required = true
    "--theta_index"
        help = "Index of the theta value to use"
        arg_type = Int
        required = false
end

parsed_args = parse_args(s)
params = YAML.load_file(parsed_args["params_file"]) 
if parsed_args["theta_index"] !== nothing #theta sweep job
    theta_index = parsed_args["theta_index"] + 1
    theta_sweep = range(params["theta_min"], params["theta_max"], params["N_theta"])
    h_theta = theta_sweep[theta_index]
    file_prefix = params["sim_anneal"]["file_prefix"] * "_theta=$(h_theta)_h"
else #normal simulated annealing job
    h_theta = params["h_theta"]
    file_prefix = params["sim_anneal"]["file_prefix"]
end

N = params["N_uc"]
S = params["S"]
Js = params["Js"]
include_cubic = params["include_cubic"]
K = params["K"][1] + im * params["K"][2] 
h_sweep_args = params["h_sweep_args"]
N_h = params["N_h"]
delta_12 = params["delta_12"]
breaking_field = params["breaking_field"]
disorder_strength = params["disorder_strength"]
disorder_seed = params["disorder_seed"]

params_sa = params["sim_anneal"]
N_therm = params_sa["mc_params"]["N_therm"]
overrelax_rate = params_sa["mc_params"]["overrelax_rate"]
N_det = params_sa["mc_params"]["N_det"]
T_args = params_sa["T_args"]
save_configs = params_sa["save_configs"]
results_dir = params_sa["results_dir"]
save_dir = params_sa["save_dir"]

h_direction = [1.0,1.0,1.0]/sqrt(3) .* cos(h_theta * pi/180) .+ [1.0,1.0,-2.0]/sqrt(6) .* sin(h_theta * pi/180)
h_min, h_max = h_sweep_args
h_sweep = range(h_min, h_max, N_h)
#h_index defined below by MPI rank

T_f, T_i = T_args

if save_configs
    T_save_args = params_sa["T_save_args"]
    T_save_min, T_save_max = T_save_args
    N_save = params_sa["N_save"]
    temp_save = exp10.(range(log10(T_save_min), stop=log10(T_save_max), length=N_save)) 
    save_configs_prefix = params_sa["save_configs_prefix"]
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
if disorder_seed[1] == 0
    disorder_seed = [rand(1:10^9)]
    MPI.Bcast!(disorder_seed, root=0, comm)
end
disorder_seed = disorder_seed[1]

N_sites = 4*N^3
#random initial configuration
spins = spins_initial_pyro(N, S)

neighbours = neighbours_all(N, N_sites)
H_bilinear = H_bilinear_all(Js, N, N_sites)

cubic_sites = cubic_sites_all(N, N_sites)
unique_triplets, unique_H_cubic_vals = unique_cubic_triplets(K, N, N_sites)
pairs_i, pairs_j, pairs_k = cubic_pairs_split_all(cubic_sites, N_sites)
H_cubic_sparse = cubic_tensors_sparse_all(K, N, N_sites)

zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, disorder_seed, breaking_field)

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

params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables(), 0, "none")

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
    #how should we do this...
    write_collection_sim_anneal(joinpath(save_dir, save_configs_prefix)*"$(h_index).h5", configurations_save, params, system, temp_save, h_direction, [norm(h)], disorder_seed)
else
    sim_anneal!(simulation, t-> T_i * 0.9^t, Float64[], r == 0 ? true : false)
end

#writes measurements to a file
fname=file_prefix*"$(h_index)_0.h5"
write_observables(joinpath(results_dir,fname), simulation)
MPI.Barrier(comm) #barrier in case 

#collect results about T_f when sweep finished
if r == 0
    collect_hsweep(results_dir, file_prefix, save_dir, system, params, [T_f], h_direction, Vector(h_sweep), disorder_seed)
    #make better
    #collect_theta_sweep("/scratch/antony/theta_sweep", "Jzz=1.0_topright_hhltheta=", "/scratch/antony/theta_sweep", -90.0, 90.0, 181)
end

