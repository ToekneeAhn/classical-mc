using HDF5

#writes a single thing (arr) to an hdf5 file called keyname 
function write_single(path, arr, keyname="spins")
    file = h5open(path, "w") 
    file[keyname] = arr 
    close(file)
end

function write_params(path, param_dict)
    allowed_keys = ["Js", "spin_length", "h", "uc_N", "N_therm", "N_det", "overrelax_rate", "replica_exchange_rate", "probe_rate"]
    
    file = h5open(path, "w") 

    for key in allowed_keys
        if haskey(param_dict, key)
            file[key] = param_dict[key]        
        end
    end
    
    close(file)
end

function write_observables(path, obs_dict)
    allowed_keys = ["energy_per_site", "energy_per_site_err", "avg_spin", "avg_spin_err"]

    file = h5open(path, "w") 

    for key in allowed_keys
        if haskey(obs_dict, key)
            file[key] = obs_dict[key]        
        end
    end
    
    close(file)
end