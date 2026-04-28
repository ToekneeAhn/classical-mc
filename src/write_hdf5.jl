using HDF5, StaticArrays, YAML

function ensure_parent_dir(path::String)
    dir = dirname(path)
    if !isdir(dir)
        mkpath(dir)
    end
end

#writes an array to an hdf5 file, with key keyname
function write_single(path::String, arr::AbstractArray, keyname="spins")
    ensure_parent_dir(path)
    h5open(path, "w") do file
        file[keyname] = arr
    end
end

#writes everything except measurements to a file
function write_all(path::String, mc::Simulation)
    ensure_parent_dir(path)
    h5open(path, "w") do file
        for key in fieldnames(SpinSystem)
            value = getfield(mc.spin_system, key)
            if value isa SArray #only have to do this for H_bond because it's a StaticArray
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
function write_observables(path::String, mc::Simulation, spin_config::Union{Nothing,AbstractMatrix}=nothing)
    ensure_parent_dir(path)
    
    h5open(path, "w") do file
        #compute observables
        for (measurement_name, measurement_value) in mc.observables.output
            file[measurement_name] = measurement_value
        end

        if !isnothing(spin_config)
            file["spins"] = spin_config
        end
    end
end

function write_parameters(path::String, system::SpinSystem, params::MCParams, config::RunConfig)
    ensure_parent_dir(path)
    h5open(path, "w") do file
        param_gr = create_group(file, "parameters")

        function parameter_value(key::String)
            key_symbol = Symbol(key)
            if hasproperty(params, key_symbol)
                return getproperty(params, key_symbol)
            elseif hasproperty(system, key_symbol)
                return getproperty(system, key_symbol)
            elseif hasproperty(config, key_symbol)
                return getproperty(config, key_symbol)
            else
                error("Unknown parameter field: $(key). Update PARAMETER_FIELDS or write_parameters mapping.")
                return nothing
            end
        end

        for key in PARAMETER_FIELDS
            param_gr[key] = parameter_value(key)
        end
    end
end

#collects h sweep data from all ranks and all h points into one file, as well as simulation parameters
function collect_hsweep(results_dir::String, file_prefix::String, save_dir::String, parameters_path::String)
    raw_files = Set(readdir(results_dir, join=false, sort=false))
    
    mkpath(save_dir)

    function collect_group(observable::String, rank::Int64, N_h::Int64)
        sample_data = nothing
        for n in 1:N_h
            sample_file = file_prefix*"$(n)_$(rank).h5"
            if sample_file in raw_files
                h5open(joinpath(results_dir, sample_file), "r") do fid
                    if haskey(fid, observable)
                        sample_data = read(fid[observable])
                    end
                end
                if sample_data !== nothing
                    break
                end
            end
        end

        if sample_data === nothing
            println("Observable $(observable) not found for rank $(rank). Skipping.")
            return nothing, Int64[]
        end

        data = Vector{typeof(sample_data)}(undef, N_h)
        warning_points = Int64[]
        last_value = sample_data

        for n in 1:N_h
            fname = file_prefix*"$(n)_$(rank).h5"
            if fname in raw_files
                h5open(joinpath(results_dir, fname), "r") do fid
                    if haskey(fid, observable)
                        last_value = read(fid[observable])
                        data[n] = last_value
                    else
                        println("Observable $(observable) missing in $(fname). Using last available data point.")
                        data[n] = last_value
                        push!(warning_points, n)
                    end
                end
            else
                println("File $(fname) not found! Using last available data point.")
                data[n] = last_value
                push!(warning_points, n)
            end
        end

        #stack along a new first axis
        if sample_data isa AbstractMatrix
            return permutedims(stack(data), (3,1,2)), warning_points
        elseif sample_data isa AbstractVector
            return permutedims(stack(data), (2,1)), warning_points
        else
            return data, warning_points
        end
    end

    h5open(joinpath(save_dir, file_prefix*"sweep.h5"), "w") do file
        param_gr = create_group(file, "parameters")

        #read parameters from parameters file
        N_h, N_ranks = h5open(parameters_path, "r") do fid
            for key in keys(fid["parameters"])
                param_gr[key] = read(fid["parameters"][key])
            end

            h_len = length(fid["parameters"]["h_sweep"])
            ranks_len = length(fid["parameters"]["Ts"])
            
            return h_len, ranks_len
        end

        warning = Set{Int64}()
        for rank in 0:(N_ranks-1)
            #all as a function of magnetic field h
            gr = create_group(file, "rank_$(rank)")
            for obs in OBSERVABLE_FIELDS
                data, warning_points = collect_group(obs, rank, N_h)
                
                if data === nothing
                    continue
                end

                gr[obs] = data

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
function write_collection_sim_anneal(path::String, configurations_save::Vector{Matrix{Float64}}, temp_save::Vector{Float64})
    ensure_parent_dir(path)
    h5open(path, "w") do file
        file["temp_save"] = temp_save

        for i in eachindex(temp_save)
            file["spins_$(i)"] = configurations_save[i]
        end
    end
end

function read_configuration_hdf5(path::String, index::Int64)
    h5open(path, "r") do file
        temp_save = read(file["temp_save"])
        spins = read(file["spins_$(index)"])
        return spins, temp_save
    end
end

function collect_theta_sweep(results_dir::String, file_prefix::String, save_dir::String, theta_min::Float64, theta_max::Float64, N_theta::Int64; params_file::Union{Nothing,String}=nothing)
    raw_files = readdir(results_dir, join=false, sort=false)
    theta_values = collect(range(theta_min, theta_max, length=N_theta))
    
    mkpath(save_dir)

    h5open(joinpath(save_dir, file_prefix*"_$(theta_min)to$(theta_max).h5"), "w") do file
        file["theta_values"] = Vector(theta_values)
        
        last_fid = nothing
        parameters_source_fid = nothing
        missing_theta_indices = Int64[]
        
        for theta_index in 0:(N_theta-1)
            theta = theta_values[theta_index+1]
            fname = file_prefix*"_theta$(theta_index)_hsweep.h5"

            if fname in raw_files
                h5open(joinpath(results_dir, fname), "r") do fid
                    gr = create_group(file, "$(theta_index)")
                    # Copy each object (group or dataset) preserving structure
                    for key in keys(fid)
                        if key != "parameters" # Skip parameters group to avoid redundancy
                            HDF5.copy_object(fid[key], gr, key)
                        end
                    end
                end
                
                # Update last available file for fallback
                last_fid = joinpath(results_dir, fname)
                if parameters_source_fid === nothing
                    parameters_source_fid = last_fid
                end
            else
                if last_fid !== nothing
                    println("File for theta_index=$(theta_index) (theta=$(theta)) not found! Using last available data point.")
                    push!(missing_theta_indices, theta_index)
                    
                    # Copy data from last available file
                    h5open(last_fid, "r") do fid
                        gr = create_group(file, "$(theta_index)")
                        for key in keys(fid)
                            if key != "parameters"
                                HDF5.copy_object(fid[key], gr, key)
                            end
                        end
                    end
                else
                    println("ERROR: File for theta_index=$(theta_index) (theta=$(theta)) not found and no previous data available!")
                    push!(missing_theta_indices, theta_index)
                end
            end 
        end

        if parameters_source_fid === nothing
            error("No theta sweep files found in $(results_dir) for prefix $(file_prefix)")
        end

        if !isempty(missing_theta_indices)
            println("Warning: Missing/copied data for theta indices: ", missing_theta_indices)
        end

        file["missing_theta_points"] = missing_theta_indices
        h5open(parameters_source_fid, "r") do fid
            param_gr = create_group(file, "parameters")
            for key in keys(fid["parameters"])
                param_gr[key] = read(fid["parameters"][key])
            end

            if params_file !== nothing
                params = YAML.load_file(params_file)
                if haskey(params, "plane_n1") && haskey(params, "plane_n2")
                    pv = create_group(param_gr, "plane_vectors")
                    pv["plane_vector_1"] = Float64.(params["plane_n1"])
                    pv["plane_vector_2"] = Float64.(params["plane_n2"])
                end
            end

            file["h_values"] = read(fid["parameters"]["h_sweep"])
        end
        
        println("Saved $N_theta theta points to ", joinpath(save_dir, file_prefix*"_$(theta_min)to$(theta_max).h5"))
    end
end