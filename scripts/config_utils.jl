using LinearAlgebra, YAML

function _build_field_basis(params)
    n_1 = Float64.(params["plane_n1"])
    @assert norm(n_1) > 0 "plane_n1 must be non-zero"
    n_1 ./= norm(n_1)

    n_2_raw = Float64.(params["plane_n2"])
    n_2 = n_2_raw - dot(n_2_raw, n_1) * n_1
    @assert norm(n_2) > 0 "plane_n2 must not be parallel to plane_n1"
    n_2 ./= norm(n_2)
    return n_1, n_2
end

function normalize_disorder_seed(disorder_seed)
    return disorder_seed isa AbstractVector ? [Int(disorder_seed[1])] : [Int(disorder_seed)]
end

function load_config(params_file::AbstractString, job_type::Symbol; theta_index::Union{Nothing, Int}=nothing)
    params = YAML.load_file(params_file)

    theta_args = nothing
    if haskey(params, "theta_args")
        theta_args = params["theta_args"]
    elseif haskey(params, "theta_min") && haskey(params, "theta_max")
        theta_args = [params["theta_min"], params["theta_max"]]
    end

    h_theta = nothing
    if theta_index === nothing
        h_theta = haskey(params, "h_theta") ? params["h_theta"] :
            (theta_args === nothing ? error("Missing h_theta or theta_args in params file") : theta_args[1])
    else
        theta_args === nothing && error("Theta sweep requires theta_args or theta_min/theta_max in params file")
        theta_sweep = range(theta_args[1], theta_args[2], length=params["N_theta"])
        idx = theta_index + 1
        @assert 1 <= idx <= length(theta_sweep) "theta_index is out of bounds for theta sweep"
        h_theta = theta_sweep[idx]
    end

    n_1, n_2 = _build_field_basis(params)
    h_direction = n_1 * cosd(h_theta) + n_2 * sind(h_theta)

    common_cfg = (
        params=params,
        N=params["N_uc"],
        S=params["S"],
        Js=params["Js"],
        include_cubic=params["include_cubic"],
        K=params["K"][1] + im * params["K"][2],
        h_theta=h_theta,
        h_sweep_args=params["h_sweep_args"],
        N_h=params["N_h"],
        delta_12=params["delta_12"],
        disorder_strength=params["disorder_strength"],
        disorder_seed=params["disorder_seed"],
        h_direction=h_direction,
        theta_args=theta_args,
        breaking_field=haskey(params, "breaking_field") ? [Float64.(v) for v in params["breaking_field"]] : [zeros(3), zeros(3), zeros(3), zeros(3)],
    )

    if job_type == :sim_anneal
        params_job = params["sim_anneal"]
        job_cfg = (
            N_therm=params_job["mc_params"]["N_therm"],
            overrelax_rate=params_job["mc_params"]["overrelax_rate"],
            N_det=params_job["mc_params"]["N_det"],
            T_args=params_job["T_args"],
            save_configs=params_job["save_configs"],
            results_dir=params_job["results_dir"],
            save_dir=params_job["save_dir"],
            file_prefix=params_job["file_prefix"],
            T_save_args=haskey(params_job, "T_save_args") ? params_job["T_save_args"] : nothing,
            N_save=haskey(params_job, "N_save") ? params_job["N_save"] : nothing,
            save_configs_prefix=haskey(params_job, "save_configs_prefix") ? params_job["save_configs_prefix"] : nothing,
        )
        return merge(common_cfg, job_cfg)
    elseif job_type == :parallel_temper
        params_job = params["parallel_temper"]
        job_cfg = (
            N_therm=params_job["mc_params"]["N_therm"],
            overrelax_rate=params_job["mc_params"]["overrelax_rate"],
            N_meas=params_job["mc_params"]["N_meas"],
            probe_rate=params_job["mc_params"]["probe_rate"],
            replica_exchange_rate=params_job["mc_params"]["replica_exchange_rate"],
            optimize_temperature_rate=params_job["mc_params"]["optimize_temperature_rate"],
            T_args=params_job["T_args"],
            load_configs=params_job["load_configs"],
            load_configs_prefix=params_job["load_configs_prefix"],
            results_dir=params_job["results_dir"],
            save_dir=params_job["save_dir"],
            file_prefix=params_job["file_prefix"],
            save_configs=params_job["save_configs"],
        )
        return merge(common_cfg, job_cfg)
    end

    error("Unsupported job_type: $(job_type). Use :sim_anneal or :parallel_temper")
end