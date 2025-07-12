using HDF5, LinearAlgebra

#loops through ranks and creates a group for each rank
#also creates a group for the simulation parameters

N_ranks = parse(Int64, ARGS[1])
mc_params = ARGS[2]
N_therm, overrelax_rate, N_meas, probe_rate, replica_exchange_rate = [parse(Int64, num) for num in split(mc_params,",")]
N = parse(Int64, ARGS[3])
S = parse(Float64, ARGS[4])
Js = ARGS[5]
Js = [parse(Float64, num) for num in split(Js,",")]
h_direction = ARGS[6] #comma-separated e.g. "1,1,1" or "1,1,0"
h_direction = [parse(Int64, hh) for hh in split(h_direction,",")]
h_direction /= norm(h_direction)
h_sweep_args = ARGS[7]
h_min, h_max = [parse(Float64, num) for num in split(h_sweep_args,",")]
N_h = parse(Int64, ARGS[8])
Ts = ARGS[9]
T_min, T_max = [parse(Float64, num) for num in split(Ts,",")]
results_dir = ARGS[10]
file_prefix = ARGS[11]
save_dir = ARGS[12]

h_sweep = range(h_min, h_max, N_h)

raw_files = readdir(results_dir, join=false, sort=false)
file = h5open(save_dir, "w")

param_gr = create_group(file, "parameters")
param_gr["N_therm"] = N_therm
param_gr["overrelax_rate"] = overrelax_rate
param_gr["N_meas"] = N_meas
param_gr["probe_rate"] = probe_rate
param_gr["replica_exchange_rate"] = replica_exchange_rate
param_gr["N"] = N
param_gr["S"] = S
param_gr["Js"] = Js
param_gr["Ts"] = exp10.(range(log10(T_min), stop=log10(T_max), length=N_ranks))
param_gr["h_direction"] = h_direction
param_gr["h_sweep"] = Vector(h_sweep)

for rank in 0:(N_ranks-1)
    mag = zeros(N_h)
    mag_err = zeros(N_h)
    energy = zeros(N_h)
    energy_err = zeros(N_h)
    avg_spin = zeros(N_h, 3, 4)
    avg_spin_err = zeros(N_h, 3, 4)

    for n in 1:N_h
        fname = file_prefix*"$(n)_$(rank).h5"
        
        if fname in raw_files
            fid=h5open(joinpath(results_dir,fname),"r")

            mag[n] = read(fid["magnetization"])
            mag_err[n] = read(fid["magnetization_err"])
            energy[n] = read(fid["energy"])
            energy_err[n] = read(fid["energy_err"])
            avg_spin[n,:,:] = read(fid["avg_spin"])
            avg_spin_err[n,:,:] = read(fid["avg_spin_err"])
            
            close(fid)
        else
            println("file $(n) not found!")
        end 
    end

    #all as a function of magnetic field h
    gr = create_group(file, "rank_$(rank)")
    gr["magnetization"] = mag
    gr["magnetization_err"] = mag_err
    gr["energy"] = energy
    gr["energy_err"] = energy_err
    gr["avg_spin"] = avg_spin
    gr["avg_spin_err"] = avg_spin_err
end

close(file)




