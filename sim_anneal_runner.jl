using MPI, LinearAlgebra

include("metropolis_pyrochlore.jl")
include("write_hdf5.jl")

mc_params = ARGS[1]
N_therm, overrelax_rate, N_det = [parse(Int64, num) for num in split(mc_params,",")]
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
T_f, T_i = [parse(Float64, num) for num in split(Ts,",")]
results_dir = replace(pwd(),"\\"=>"/")*"/"*ARGS[9]
file_prefix = ARGS[10]
save_dir = ARGS[11]
disorder_strength = parse(Float64, ARGS[12])
disorder_seed = [parse(Int64, ARGS[13])]

MPI.Init()
comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
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
zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, disorder_seed)

system = SpinSystem(spins, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables())

#simulated annealing with annealing schedule T = T_i*0.9^t
sim_anneal!(simulation, T_i, t->0.9^t)

if r == 0    
    #makes save directory if it doesn't exist
    if !isdir(results_dir)
        mkdir(results_dir)
    end
end

MPI.Barrier(comm) #barrier in case 
#writes measurements to a file
fname=file_prefix*"$(h_index)_0.h5"
write_observables(joinpath(results_dir,fname), simulation)

#collect results when sweep finished
if r == 0
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    collect_hsweep(results_dir, file_prefix, save_dir, system, params, [T_f], h_direction, Vector(h_sweep), disorder_seed)
end

