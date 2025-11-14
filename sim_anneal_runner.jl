using MPI, LinearAlgebra, YAML

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

params_sa = params["sim_anneal"]
N_therm = params_sa["mc_params"]["N_therm"]
overrelax_rate = params_sa["mc_params"]["overrelax_rate"]
N_det = params_sa["mc_params"]["N_det"]
T_args = params_sa["T_args"]
save_configs = params_sa["save_configs"]
results_dir = params_sa["results_dir"]
save_dir = params_sa["save_dir"]
file_prefix = params_sa["file_prefix"]

h_direction = [1.0,1.0,1.0]/sqrt(3) .* cos(h_theta * pi/180) .+ [-1.0,-1.0,2.0]/sqrt(6) .* sin(h_theta * pi/180)
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
#=
mc_params = ARGS[1]
N_therm, overrelax_rate, N_det = [parse(Int64, num) for num in split(mc_params,",")]
N = parse(Int64, ARGS[2])
S = parse(Float64, ARGS[3])
Js = ARGS[4]
Js = [parse(Float64, num) for num in split(Js,",")]
delta_12 = Js[5:6]
h_theta = parse(Float64, ARGS[5]) #angle from [111] direction towards [1-10] in degrees
h_direction = [1.0,1.0,1.0]/sqrt(3) .* cos(h_theta * pi/180) .+ [1.0,-1.0,0.0]/sqrt(2) .* sin(h_theta * pi/180)
#=
h_direction = ARGS[5] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
h_direction /= norm(h_direction)
=#
h_sweep_args = ARGS[6]
h_min, h_max = [parse(Float64, num) for num in split(h_sweep_args,",")]
N_h = parse(Int64, ARGS[7])
Ts = ARGS[8]
T_f, T_i = [parse(Float64, num) for num in split(Ts,",")]
results_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[9]
file_prefix = ARGS[10]
save_dir = ARGS[11]
disorder_strength = parse(Float64, ARGS[12])
disorder_seed = [parse(Int64, ARGS[13])]

if length(ARGS) >= 14
    temp_save_args = ARGS[14]
    temp_save_min, temp_save_max = [parse(Float64, num) for num in split(temp_save_args,",")]
    N_temp_save = parse(Int64, ARGS[15])
    #temp_save = Vector(range(temp_save_min, temp_save_max, N_temp_save))
    temp_save = exp10.(range(log10(temp_save_min), stop=log10(temp_save_max), length=N_temp_save)) 
    config_save_prefix = ARGS[16]
else
    temp_save = []
end
=#

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

H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, disorder_seed, breaking_field)

system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, H_ij, neighbours, zeeman)
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
    sim_anneal!(simulation, t-> T_i * 0.9^t, Float64[], false)
end

#writes measurements to a file
fname=file_prefix*"$(h_index)_0.h5"
write_observables(joinpath(results_dir,fname), simulation)
MPI.Barrier(comm) #barrier in case 

#collect results about T_f when sweep finished
if r == 0
    collect_hsweep(results_dir, file_prefix, save_dir, system, params, [T_f], h_direction, Vector(h_sweep), disorder_seed)
end

