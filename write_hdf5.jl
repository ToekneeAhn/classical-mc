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
function write_observables(path, mc::Simulation, spin_config::Matrix{Float64}=nothing)
    obs = mc.observables
    file = h5open(path, "w")
    
    heat, dheat = specific_heat(mc)
    susc, dsusc = susceptibility(mc)
    spinderiv, dspinderiv = dSdT(mc)

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
    file["dSdT"] = spinderiv
    file["dSdT_err"] = dspinderiv

    if spin_config !== nothing
        file["spins"] = spin_config
    end

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
    param_gr["K"] = system.K
    param_gr["Ts"] = temps
    param_gr["delta_12"] = system.delta_12
    param_gr["h_direction"] = h_direction
    param_gr["h_sweep"] = h_sweep
    param_gr["disorder_strength"] = system.disorder_strength
    param_gr["disorder_seed"] = seed

    N_h = length(h_sweep)
    N_ranks = length(temps)

    function collect_group(observable::String, data_dim, rank::Int64)
        if length(data_dim) > 1
            data = Matrix{Float64}[]
        else
            data = Float64[]
        end

        for n in 1:N_h
            fname = file_prefix*"$(n)_$(rank).h5"
            if fname in raw_files
                fid=h5open(joinpath(results_dir,fname),"r")
                push!(data, read(fid[observable]))
                close(fid)
            else
                println("file $(n) not found!")
            end 
        end
        
        if length(data_dim) > 1
            dims = length(data_dim)
            #stack along a new first axis
            return permutedims(stack(data), vcat(dims, Vector(1:dims-1)))
        else
            return data
        end
    end

    obs_dict = Dict("magnetization"=>N_h, "magnetization_err"=>N_h, "energy"=>N_h, "energy_err"=>N_h, 
                    "specific_heat"=>N_h, "specific_heat_err"=>N_h,
                    "susceptibility"=>N_h, "susceptibility_err"=>N_h, #"binder"=>N_h, "binder_err"=>N_h,  
                    "avg_spin"=>(N_h,3,4), "avg_spin_err"=>(N_h,3,4), "dSdT"=>(N_h,3,4), "dSdT_err"=>(N_h,3,4),
                    "spins"=>(N_h, size(system.spins,1), size(system.spins,2)))
    
    for rank in 0:(N_ranks-1)
        #all as a function of magnetic field h
        gr = create_group(file, "rank_$(rank)")
        for (obs, obs_dim) in obs_dict
            gr[obs] = collect_group(obs, obs_dim, rank)
        end
    end

    close(file)
end

#saves configurations at various temperatures generated from a single simulated annealing run
function write_collection_sim_anneal(path, configurations_save::Vector{Matrix{Float64}}, params::MCParams, system::SpinSystem, temp_save::Vector{Float64}, h_direction::Vector{Float64}, h_sweep::Vector{Float64}, seed::Int64)
    file = h5open(path, "w")

    param_gr = create_group(file, "parameters")
    param_gr["N_therm"] = params.N_therm
    param_gr["overrelax_rate"] = params.overrelax_rate
    param_gr["N_meas"] = params.N_meas
    param_gr["probe_rate"] = params.probe_rate
    param_gr["replica_exchange_rate"] = params.replica_exchange_rate
    param_gr["N"] = system.N
    param_gr["S"] = system.S
    param_gr["Js"] = system.Js
    param_gr["K"] = system.K
    param_gr["delta_12"] = system.delta_12
    param_gr["h_direction"] = h_direction
    param_gr["h_sweep"] = h_sweep
    param_gr["disorder_strength"] = system.disorder_strength
    param_gr["disorder_seed"] = seed

    file["Ts"] = temp_save

    for i in eachindex(temp_save)
        file["spins_$(i)"] = configurations_save[i]
    end
    
    close(file)
end

function read_configuration_hdf5(path::String, index::Int64)
    file = h5open(path, "r")
    Ts = read(file["Ts"])
    spins = read(file["spins_$(index)"])
    close(file)

    return spins, Ts
end