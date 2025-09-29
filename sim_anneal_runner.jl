using Random, Plots

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

#single run
T_i = 1.0 #initial temperature
T_f = 1e-4 #target temperature

N_sites = 4*N^3
#random initial configuration
spins = spins_initial_pyro(N, S)

disorder_strength = 0.0 #in K
H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, [0.0, 0.0], disorder_strength, N_sites)

system = SpinSystem(spins, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables(), 0, "none") #set temperature to T_f to use in observables

#simulated annealing with annealing schedule T = T_i*0.9^t
@time energies = sim_anneal!(simulation, T_i, t->0.9^t)

#write to file, create directory if it doesn't exist
#=
fname = "simulation_obs.h5"
save_dir = replace(pwd(),"\\"=>"/")*"/sim_anneal/"
if !isdir(save_dir)
    mkdir(save_dir)
end

write_observables(save_dir*fname, simulation)

#writes parameters to a file
fname_params = "simulation_params.h5"
write_all(save_dir*fname_params, simulation)
=#

#plot energy as a function of sweep
display(plot(energies, xlabel="Sweep", ylabel="E", legend=false, ms=2,title=Js))

#plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
display(scatter!(spins[3,:], label="Sz", ms=4))
ylims!(-1.1,1.1)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")

println(spin_expec(spins, N))
