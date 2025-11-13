using Random, Plots

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

#single run
T_i = 1.0 #initial temperatures
T_f = 1e-4 #target temperature

N_sites = 4*N^3
#random initial configuration
spins = spins_initial_pyro(N, S)

H_bilinear = H_bilinear_all(Js, N, N_sites)
neighbours = neighbours_all(N, N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, delta_12, disorder_strength, N_sites, 0)

system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman)
params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables(), 0, "none") #set temperature to T_f to use in observables

#simulated annealing with annealing schedule T = T_i*0.9^t
temp_save = [0.001, 0.01, 0.011, 0.012, 0.05, 0.1, 0.5]
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

println(spin_expec(spins, N))

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
