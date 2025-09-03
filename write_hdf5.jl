using HDF5, StaticArrays

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
        value = getfield(mc.spin_system, key)
        if typeof(value) <: SArray #only have to do this for H_bond because it's a StaticArray
            file[String(key)] = Array(value)
        else
            file[String(key)] = value
        end
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
    
    heat, dheat = specific_heat(mc)
    susc, dsusc = susceptibility(mc)
    binder, dbinder = binder_cumulant(mc)

    #compute observables
    file["avg_spin"] = mean(obs.avg_spin,1)
    file["avg_spin_err"] = std_error(obs.avg_spin,1)
    file["avg_spin_squared"] = mean(obs.avg_spin,2)
    file["avg_spin_squared_err"] = std_error(obs.avg_spin,2)
    file["energy"] = mean(obs.energy, 1)
    file["energy_err"] = std_error(obs.energy,1)
    file["magnetization"] = mean(obs.magnetization,1)
    file["magnetization_err"] = std_error(obs.magnetization,1)
    file["specific_heat"] = heat
    file["specific_heat_err"] = dheat
    file["susceptibility"] = susc
    file["susceptibility_err"] = dsusc
    file["binder"] = binder
    file["binder_err"] = dbinder

    close(file)
end

#collects h sweep data from all ranks and all h points into one file, as well as simulation parameters
function collect_hsweep(results_dir::String, file_prefix::String, save_dir::String, system::SpinSystem, params::MCParams, temps::Vector{Float64}, h_direction::Vector{Float64}, h_sweep::Vector{Float64}, seed::Int64)
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
    param_gr["disorder_strength"] = system.disorder_strength
    param_gr["disorder_seed"] = seed

    N_h = length(h_sweep)
    N_ranks = length(temps)

    for rank in 0:(N_ranks-1)
        #how to make this less bad 
        mag = zeros(N_h)
        mag_err = zeros(N_h)
        energy = zeros(N_h)
        energy_err = zeros(N_h)
        avg_spin = zeros(N_h, 3, 4)
        avg_spin_err = zeros(N_h, 3, 4)
        avg_spin_squared = zeros(N_h, 3, 4)
        avg_spin_squared_err = zeros(N_h, 3, 4)
        spec_heat = zeros(N_h)
        spec_heat_err = zeros(N_h)
        susc = zeros(N_h)
        susc_err = zeros(N_h)
        binder = zeros(N_h)
        binder_err = zeros(N_h)

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
                avg_spin_squared[n,:,:] = read(fid["avg_spin_squared"])
                avg_spin_squared_err[n,:,:] = read(fid["avg_spin_squared_err"])
                spec_heat[n] = read(fid["specific_heat"])
                spec_heat_err[n] = read(fid["specific_heat_err"])
                susc[n] = read(fid["susceptibility"])
                susc_err[n] = read(fid["susceptibility_err"])
                binder[n] = read(fid["binder"])
                binder_err[n] = read(fid["binder_err"])
                
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
        gr["avg_spin_squared"] = avg_spin_squared
        gr["avg_spin_squared_err"] = avg_spin_squared_err
        gr["specific_heat"] = spec_heat
        gr["specific_heat_err"] = spec_heat_err
        gr["susceptibility"] = susc
        gr["susceptibility_err"] = susc_err
        gr["binder"] = binder
        gr["binder_err"] = binder_err
    end

    close(file)
end