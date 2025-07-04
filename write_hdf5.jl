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
    file["energy_per_site"] = mean(obs.energy, 1)
    file["energy_per_site_err"] = std_error(obs.energy,1)
    file["magnetization"] = mean(obs.magnetization,1)
    file["magnetization_err"] = std_error(obs.magnetization,1)

    close(file)
end
