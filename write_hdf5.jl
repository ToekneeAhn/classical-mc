using HDF5

#writes an array to an hdf5 file, with key keyname
function write_single(path, arr, keyname="spins")
    file = h5open(path, "w") 
    file[keyname] = arr 
    close(file)
end

#writes everything except measurements to a file
function write_all(path, mc::Simulation)
    file = h5open(path, "w") 

    for key in fieldnames(SpinSystem)
        file[String(key)] = getfield(mc.spin_system, key)
    end
    
    for key in fieldnames(MCParams)
        file[String(key)] = getfield(mc.parameters, key)
    end

    file["T"] = [mc.T]

    close(file)
end

#writes measurements to a file
function write_observables(path, mc::Simulation)
    obs = mc.observables
    file = h5open(path, "w")
    
    #compute observables
    file["avg_spin"] = mean(obs.avg_spin,1)
    file["avg_spin_err"] = std_error(obs.avg_spin,1)
    file["energy"] = mean(obs.energy, 1)
    file["energy_err"] = std_error(obs.energy,1)
    file["magnetization"] = mean(obs.magnetization,1)
    file["magnetization_err"] = std_error(obs.magnetization,1)

    close(file)
end

#collects h sweep data from all ranks and all h points into one file, as well as simulation parameters
function collect_hsweep(results_dir::String, file_prefix::String, save_dir::String, system::SpinSystem, params::MCParams, temps::Vector{Float64}, h_direction::Vector{Float64}, h_sweep::Vector{Float64})
    raw_files = readdir(results_dir, join=false, sort=false)
    file = h5open(joinpath(save_dir, file_prefix*"sweep.h5"), "w")

    param_gr = create_group(file, "parameters")
    param_gr["N_therm"] = params.N_therm
    param_gr["overrelax_rate"] = params.overrelax_rate
    param_gr["N_meas"] = params.N_meas
    param_gr["probe_rate"] = params.probe_rate
    param_gr["replica_exchange_rate"] = params.replica_exchange_rate
    param_gr["N"] = system.N
    param_gr["S"] = system.S
    param_gr["Js"] = system.Js
    param_gr["Ts"] = temps
    param_gr["h_direction"] = h_direction
    param_gr["h_sweep"] = h_sweep

    N_h = length(h_sweep)
    N_ranks = length(temps)

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
end