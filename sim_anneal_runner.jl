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

disorder_strength = 3.132 #in K
H_ij = H_matrix_all(Js)
neighbours = neighbours_all(N_sites)
zeeman = zeeman_field_random(h, z_local, local_interactions, disorder_strength, N_sites)

system = SpinSystem(spins, S, N, N_sites, Js, h, disorder_strength, H_ij, neighbours, zeeman)
params = MCParams(N_therm, N_det, overrelax_rate, -1, -1, -1)
simulation = Simulation(system, T_f, params, Observables()) #set temperature to T_f to use in observables

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
display(plot(energies, xlabel="Sweep", ylabel="E/|Jzz|", legend=false, ms=2,title=Js))

#.plot spin components, grouped by sublattice
scatter(spins[1,:], label="Sx", ms=4, plot_title=Js)
scatter!(spins[2,:], label="Sy", ms=4)
scatter!(spins[3,:], label="Sz", ms=4)
ylims!(-1.1,1.1)
xticks!([N^3, 2*N^3, 3*N^3, 4*N^3])
vline!([N^3, 2*N^3, 3*N^3, 4*N^3].+0.5, linestyle=:dash, linecolor=:red, label="")


#=
N_loop = 50

energies = zeros(N_loop)
avg_spin = zeros(N_loop, 3, 4)

for n in 1:N_loop
    #random initial configuration
    spins = spins_initial_pyro(N, S)

    filler, measurements = sim_anneal!(spins, S, N, Js, h, N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, t->0.9^t)
    energies[n] = measurements[1]
    avg_spin[n,:,:] = measurements[2]
    if n%10 == 0
        println("sweep ", n, " of ", N_loop)
    end
end

out_path = replace(pwd(),"\\"=>"/")*"/sim_anneal/"
write_single(out_path*"sim_anneal_energy_$(h_strength).h5", energies, "energies")
write_single(out_path*"sim_anneal_avg_spin_$(h_strength).h5", avg_spin, "avg_spin")

params=Dict("Js"=>collect(Js), "spin_length"=>S, "h"=>h, "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate, "overrelax_rate"=>overrelax_rate)
params_fname = "params_$(h_strength).h5"
write_params(out_path*"params_$(h_strength).h5", params)

print(N_loop, " loops done!")
=#

#=
#sweeping h to compute magnetization
T_i = 1.0 #initial temperature
T_f = 1e-4 #target temperature

N_loop = 50
h_sweep = range(0.0, 12.0, length=N_loop)

energies = zeros(N_loop)
avg_spin = zeros(N_loop, 3, 4)
out_path = replace(pwd(),"\\"=>"/")*"/sim_anneal/"

for n in 1:N_loop
    #random initial configuration
    spins = spins_initial_pyro(N, S)
    h_loop = h_sweep[n]*[1,1,1]/sqrt(3)
    filler, measurements = sim_anneal!(spins, S, N, Js, h_loop, N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, t->0.9^t)
    energies[n] = measurements[1]
    avg_spin[n,:,:] = measurements[2]
    
    if n%10 == 0
        println("sweep ", n, " of ", N_loop)
    end

    if n == 50
        params=Dict("Js"=>collect(Js), "spin_length"=>S, "h"=>h_loop, "uc_N"=>N, "N_therm"=>N_therm, "N_det"=>N_det, "probe_rate" =>probe_rate, "overrelax_rate"=>overrelax_rate)
        params_fname = "params_h_sweep.h5"
        write_params(out_path*params_fname, params)
    end
end

write_single(out_path*"energy_h_sweep.h5", energies, "energies")
write_single(out_path*"avg_spin_h_sweep.h5", avg_spin, "avg_spin")

print(N_loop, " loops done!")
=#