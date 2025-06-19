using BenchmarkTools, Plots

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

#simulation parameters
T_i = 1.0 #initial temperature
T_f = 1e-6 #target temperature

#simulated annealing with random initial spin configuration
@time spins, energies_all, measurements = sim_anneal(N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, Js, h, N, S, [])

#=
N_loop = 50

energies = zeros(N_loop)
avg_spin = zeros(N_loop, 3, 4)

for n in 1:N_loop
    measurements = sim_anneal(N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, Js, h, N, S, [])
    energies[n] = measurements[1]
    avg_spin[n,:,:] = measurements[2]
    if n%10 == 0
        println("sweep ", n, " of ", N_loop)
    end
end

out_path = replace(pwd(),"\\"=>"/")*"/pt_out/"
write_single(out_path*"sim_anneal_energy_$(h_strength).h5", energies, "energies")
write_single(out_path*"sim_anneal_avg_spin_$(h_strength).h5", avg_spin, "avg_spin")

params=Dict("Js"=>Js, "spin_length"=>S, "h"=>h, 
    "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate,
    "overrelax_rate"=>overrelax_rate)
write_params(out_path*"params_$(h_strength).h5", params)

print(N_loop, " loops done!")
=#

#write to file
obs_path = replace(pwd(),"\\"=>"/")*"/pt_out/sim_anneal_out.h5"
write_observables(obs_path, Dict("energy_per_site"=>measurements[1], "avg_spin"=>measurements[2]))

#plot energy as a function of sweep
display(plot(energies_all, xlabel="Sweep", ylabel="E/|Jzz| per site", legend=false, ms=2,title=Js))

#.plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
scatter!(spins[3,:], label="Sz", ms=4)
ylims!(-1.1,1.1)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")
