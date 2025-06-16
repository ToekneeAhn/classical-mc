using BenchmarkTools, Plots

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

#simulation parameters
T_i = 1.0 #initial temperature
T_f = 1e-4 #target temperature

#simulated annealing with random initial spin configuration
@time spins, energies = sim_anneal(N_therm, N_det, overrelax_rate, T_i, T_f, Js, h, N, S, [])

#average spin on sublattice
measurements = zeros(3,4)
measurements[:,1] = sum(spins[:,1:N^3], dims=2)[:,1]
measurements[:,2] = sum(spins[:,(N^3+1):(2*N^3)], dims=2)[:,1]
measurements[:,3] = sum(spins[:,(2*N^3+1):(3*N^3)], dims=2)[:,1]
measurements[:,4] = sum(spins[:,(3*N^3+1):(4*N^3)], dims=2)[:,1]
measurements /= N^3

#write to file
obs_path = replace(pwd(),"\\"=>"/")*"/pt_out/sim_anneal_out.h5"
write_observables(obs_path, Dict("avg_spin"=>measurements, "energy_per_site"=>energies[end]))

#plot energy as a function of sweep
display(plot(energies, xlabel="Sweep", ylabel="E/|Jzz| per site", legend=false, ms=2,title=Js))

#plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
scatter!(spins[3,:], label="Sz", ms=4)
ylims!(-1.1,1.1)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")

