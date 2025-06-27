using Plots

include("metropolis_pyrochlore.jl") 
include("write_hdf5.jl")
include("input_file.jl")

#simulation parameters
T_i = 1.0 #initial temperature
T_f = 1e-4 #target temperature

#random initial configuration
spins = spins_initial_pyro(N, S)

#simulated annealing with annealing schedule T = T_i*0.9^t
@time energies, measurements = sim_anneal!(spins, S, N, Js, h, N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, t->0.9^t)

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

#write to file, create directory if it doesn't exist
fname = "out.h5"
save_dir = replace(pwd(),"\\"=>"/")*"/sim_anneal/"
if !isdir(save_dir)
    mkdir(save_dir)
end
write_observables(save_dir*fname, Dict("energy_per_site"=>measurements[1], "avg_spin"=>measurements[2]))

#writes parameters to a file
#use collect to turn tuple to array
params=Dict("Js"=>collect(Js), "spin_length"=>S, "h"=>h, "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate, "overrelax_rate"=>overrelax_rate)
params_fname = "params.h5"
write_params(save_dir*params_fname, params)

#plot energy as a function of sweep
display(plot(energies, xlabel="Sweep", ylabel="E/|Jzz| per site", legend=false, ms=2,title=Js))

#.plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
scatter!(spins[3,:], label="Sz", ms=4)
ylims!(-1.1,1.1)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")
