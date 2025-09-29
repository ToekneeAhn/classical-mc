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

    for key in fieldnames(typeof(mc.spin_system))
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

    #compute observables
    file["avg_spin"] = mean(obs.avg_spin,1)
    file["avg_spin_err"] = std_error(obs.avg_spin,1)
    file["energy"] = mean(obs.energy, 1)
    file["energy_err"] = std_error(obs.energy,1)
    file["magnetization"] = mean(obs.magnetization,1)
    file["magnetization_err"] = std_error(obs.magnetization,1)
    file["specific_heat"] = heat
    file["specific_heat_err"] = dheat
    file["susceptibility"] = susc
    file["susceptibility_err"] = dsusc

    close(file)
end
