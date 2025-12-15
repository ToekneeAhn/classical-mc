using Random, Plots, BenchmarkTools

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

N_sites = 4*N^3
#random initial configuration
spins = spins_initial_pyro(N, S)

neighbours = neighbours_all(N, N_sites)
H_bilinear = H_bilinear_all(Js, N, N_sites)

cubic_sites = cubic_sites_all(N, N_sites)
pairs_i, pairs_j, pairs_k = cubic_pairs_split_all(cubic_sites, N_sites)
unique_triplets, unique_H_cubic_vals = unique_cubic_triplets(K, N, N_sites)
H_cubic_sparse = cubic_tensors_sparse_all(K, N, N_sites)

zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, 0)

if include_cubic
    system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, K, cubic_sites, H_cubic_sparse, unique_triplets, unique_H_cubic_vals, pairs_i, pairs_j, pairs_k, zeeman)
else
    system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman)
end

params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables(), 0, "none") #set temperature to T_f to use in observables

#simulated annealing with annealing schedule T = T_i*0.9^t
#temp_save = [0.001, 0.01, 0.011, 0.012, 0.05, 0.1, 0.5]
temp_save = Float64[]
@time energies, configurations_save = sim_anneal!(simulation, t-> T_i * 0.9^t, temp_save, true)

#plot energy as a function of sweep
display(plot(energies, xlabel="Sweep", ylabel="E", legend=false, ms=2,title=Js))

#plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
ylims!(-0.55,0.55)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")
display(scatter!(spins[3,:], label="Sz", ms=4))

println("Final energy per site: ", round(E_pyro(system)/N_sites, digits=6))

S_avg = spin_expec(spins, N)
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

#save configurations to hdf5
save_dir = "sim_anneal_collect"
fname = "configurations.h5"

if !isdir(save_dir)
    mkdir(save_dir)
end

disorder_seed = 0
if length(configurations_save) > 0
    write_collection_sim_anneal(joinpath(save_dir, fname), configurations_save, params, system, temp_save, h_direction, [h_strength], disorder_seed)
end