using HDF5, StaticArrays

#writes an array to an hdf5 file, with key keyname
function write_single(path::String, arr::AbstractArray, keyname="spins")
    h5open(path, "w") do file
        file[keyname] = arr
    end
end

#writes everything except measurements to a file
function write_all(path::String, mc::Simulation)
    h5open(path, "w") do file
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
    end
end

#writes measurements to a file
function write_observables(path::String, mc::Simulation, spin_config::Matrix{Float64}=zeros(0,0))
    obs = mc.observables
    
    heat, dheat = specific_heat(mc)
    susc, dsusc = susceptibility(mc)
    binder, dbinder = binder_cumulant(mc)
    susc_T, dsusc_T = dSdT(mc)

    h5open(path, "w") do file
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
        file["dSdT"] = susc_T
        file["dSdT_err"] = dsusc_T

        if spin_config !== zeros(0,0)
            file["spins"] = spin_config
        end
    end
end

function write_parameters(path::String, system::SpinSystem, params::MCParams, Ts::Vector{Float64}, h_direction::Vector{Float64}, h_sweep::Vector{Float64}, seed::Int64)
    h5open(path, "w") do file
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
        param_gr["Ts"] = Ts
        param_gr["delta_12"] = system.delta_12
        param_gr["h_direction"] = h_direction
        param_gr["h_sweep"] = h_sweep
        param_gr["disorder_strength"] = system.disorder_strength
        param_gr["disorder_seed"] = seed
    end
end

#collects h sweep data from all ranks and all h points into one file, as well as simulation parameters
function collect_hsweep(results_dir::String, file_prefix::String, save_dir::String, parameters_path::String)
    raw_files = readdir(results_dir, join=false, sort=false)
    
    h5open(joinpath(save_dir, file_prefix*"sweep.h5"), "w") do file
        param_gr = create_group(file, "parameters")

        #read parameters from parameters file
        N_h, N_ranks, N = h5open(parameters_path, "r") do fid
            for key in keys(fid["parameters"])
                param_gr[key] = read(fid["parameters"][key])
            end

            h_len = length(fid["parameters"]["h_sweep"])
            ranks_len = length(fid["parameters"]["Ts"])
            uc_len = read(fid["parameters"]["N"])
            
            return h_len, ranks_len, uc_len
        end

        function collect_group(observable::String, data_dim, rank::Int64)
            if length(data_dim) > 1
                data = Matrix{Float64}[]
            else
                data = Float64[]
            end

            warning_points = Int64[]

            for n in 1:N_h
                fname = file_prefix*"$(n)_$(rank).h5"
                if fname in raw_files
                    h5open(joinpath(results_dir,fname),"r") do fid
                        push!(data, read(fid[observable]))
                    end
                else
                    println("File $(n) not found! Using last available data point.")
                    push!(data, data[end])
                    push!(warning_points, n)
                end 
            end
            
            if length(data_dim) > 1
                dims = length(data_dim)
                #stack along a new first axis
                return permutedims(stack(data), vcat(dims, Vector(1:dims-1))), warning_points
            else
                return data, warning_points
            end
        end

        obs_dict = Dict("magnetization"=>N_h, "magnetization_err"=>N_h, "energy"=>N_h, "energy_err"=>N_h, 
                        "specific_heat"=>N_h, "specific_heat_err"=>N_h,
                        "susceptibility"=>N_h, "susceptibility_err"=>N_h, "binder"=>N_h, "binder_err"=>N_h,  
                        "avg_spin"=>(N_h,3,4), "avg_spin_err"=>(N_h,3,4), "dSdT"=>(N_h,3,4), "dSdT_err"=>(N_h,3,4),
                        "spins"=>(N_h, 3, 4 * N^3))
        
        warning = Set{Int64}()
        for rank in 0:(N_ranks-1)
            #all as a function of magnetic field h
            gr = create_group(file, "rank_$(rank)")
            for (obs, obs_dim) in obs_dict
                dat, warning_points = collect_group(obs, obs_dim, rank)
                
                gr[obs] = dat
                if !isempty(warning_points)
                    push!(warning, warning_points...)
                end
            end
        end

        if !isempty(warning)
            println("Warning: Missing data for h points: ", sort(collect(warning)))
        end
        file["missing_h_points"] = sort(collect(warning))

        println("Saved $N_h h points at $N_ranks temperature points to ", joinpath(save_dir, file_prefix*"sweep.h5"))
    end
end

#saves configurations at various temperatures generated from a single simulated annealing run
function write_collection_sim_anneal(path::String, configurations_save::Vector{Matrix{Float64}}, params::MCParams, system::SpinSystem, temp_save::Vector{Float64}, h_direction::Vector{Float64}, h_sweep::Vector{Float64}, seed::Int64)
    h5open(path, "w") do file
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
        param_gr["h_direction"] = h_direction
        param_gr["h_sweep"] = h_sweep
        param_gr["disorder_strength"] = system.disorder_strength
        param_gr["disorder_seed"] = seed

        file["Ts"] = temp_save

        for i in eachindex(temp_save)
            file["spins_$(i)"] = configurations_save[i]
        end
    end
end

function read_configuration_hdf5(path::String, index::Int64)
    h5open(path, "r") do file
        Ts = read(file["Ts"])
        spins = read(file["spins_$(index)"])
        return spins, Ts
    end
end

function collect_theta_sweep(results_dir::String, file_prefix::String, save_dir::String, theta_min::Float64, theta_max::Float64, N_theta::Int64)
    raw_files = readdir(results_dir, join=false, sort=false)
    theta_values = range(theta_min, theta_max, length=N_theta)
    
    h5open(joinpath(save_dir, file_prefix*"_$(theta_min)to$(theta_max).h5"), "w") do file
        file["theta_values"] = Vector(theta_values)
        
        last_fid = nothing
        missing_thetas = Float64[]
        
        for theta in theta_values
            fname = file_prefix*"_theta=$(theta)_hsweep.h5"
            
            if fname in raw_files
                h5open(joinpath(results_dir, fname), "r") do fid
                    gr = create_group(file, "$(theta)")
                    # Copy each object (group or dataset) preserving structure
                    for key in keys(fid)
                        HDF5.copy_object(fid[key], gr, key)
                    end
                end
                
                # Update last available file for fallback
                last_fid = joinpath(results_dir, fname)
            else
                if last_fid !== nothing
                    println("File for theta=$(theta) not found! Using last available data point.")
                    push!(missing_thetas, theta)
                    
                    # Copy data from last available file
                    h5open(last_fid, "r") do fid
                        gr = create_group(file, "$(theta)")
                        for key in keys(fid)
                            HDF5.copy_object(fid[key], gr, key)
                        end
                    end
                else
                    println("ERROR: File for theta=$(theta) not found and no previous data available!")
                    push!(missing_thetas, theta)
                end
            end 
        end
        
        if !isempty(missing_thetas)
            println("Warning: Missing/copied data for theta values: ", missing_thetas)
        end

        file["missing_theta_points"] = missing_thetas
        
        println("Saved $N_theta theta points to ", joinpath(save_dir, file_prefix*"_$(theta_min)to$(theta_max).h5"))
    end
end