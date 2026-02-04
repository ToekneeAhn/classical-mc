using YAML, Dates

# Helper functions
function generate_sim_anneal_script(julia_script, params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    N_h = params["N_h"]
    results_dir = params["sim_anneal"]["results_dir"]
    file_prefix = params["sim_anneal"]["file_prefix"]
    
    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=$N_h
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=$(params["sim_anneal"]["job"]["mem_per_cpu"])
    #SBATCH --time=$(params["sim_anneal"]["job"]["time"])
    #SBATCH --job-name=$(params["sim_anneal"]["job"]["job_name"])
    #SBATCH --output=/scratch/antony/slurm_out/%j.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module purge
    module load StdEnv/2023 julia/1.11.3

    srun julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime

    cd $results_dir
    rm $file_prefix*
    """
end

function generate_parallel_temper_script_single_node(julia_script, params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    N_h = params["N_h"]
    N_Ts = params["parallel_temper"]["N_Ts"]
    results_dir = params["parallel_temper"]["results_dir"]
    file_prefix = params["parallel_temper"]["file_prefix"]
    
    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=$N_Ts
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=$(params["parallel_temper"]["job"]["mem_per_cpu"])
    #SBATCH --time=$(params["parallel_temper"]["job"]["time"])
    #SBATCH --job-name=$(params["parallel_temper"]["job"]["job_name"])
    #SBATCH --output=/scratch/antony/slurm_out/%j.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module purge
    module load StdEnv/2023 julia/1.11.3

    for ((i=1; i<=$N_h; i++));
    do
        srun julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime --h_index \$i
    done
    """
end

function generate_parallel_temper_script(julia_script, params_file_runtime, account, h_points_per_node)
    params = YAML.load_file(params_file_runtime)
    N_h = params["N_h"]
    N_Ts = params["parallel_temper"]["N_Ts"]
    results_dir = params["parallel_temper"]["results_dir"]
    file_prefix = params["parallel_temper"]["file_prefix"]

    N_nodes = ceil(Int, N_h / h_points_per_node)
    
    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --array=0-$(N_nodes-1)
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=$N_Ts
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=$(params["parallel_temper"]["job"]["mem_per_cpu"])
    #SBATCH --time=$(params["parallel_temper"]["job"]["time"])
    #SBATCH --job-name=$(params["parallel_temper"]["job"]["job_name"])
    #SBATCH --output=/scratch/antony/slurm_out/%j.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module purge
    module load StdEnv/2023 julia/1.11.3

    N_H=$N_h
    H_POINTS_PER_NODE=$h_points_per_node

    START_H_IDX=\$((SLURM_ARRAY_TASK_ID * H_POINTS_PER_NODE))

    # Launch multiple jobs in parallel on this node
    for ((i=0; i<H_POINTS_PER_NODE; i++)); do
        H_IDX=\$((START_H_IDX + i + 1))
        
        if [ \$H_IDX -le \$N_H ]; then
            srun --output=/scratch/antony/slurm_out/%A_h\${H_IDX}.out julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime --h_index \$H_IDX 
        fi
    done

    # Wait for all background jobs to complete
    wait
    """
end

function generate_parallel_temper_collection_script(params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    results_dir = params["parallel_temper"]["results_dir"]
    save_dir = params["parallel_temper"]["save_dir"]
    file_prefix = params["parallel_temper"]["file_prefix"]
    parameters_path = joinpath(results_dir, file_prefix*"_parameters.h5")

    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=4000M
    #SBATCH --time=00:15:00
    #SBATCH --job-name=collect_pt
    #SBATCH --output=/scratch/antony/slurm_out/%j_collect.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module load StdEnv/2023 julia/1.11.3

    # run collection script
    cd /home/antony/classical-mc
    julia --project=/home/antony/classical-mc -e "include(\\\"metropolis_pyrochlore.jl\\\"); include(\\\"write_hdf5.jl\\\"); collect_hsweep(\\\"$results_dir\\\", \\\"$file_prefix\\\", \\\"$save_dir\\\", \\\"$parameters_path\\\")"

    # clean up
    cd $results_dir
    rm $file_prefix*
    """
end

function generate_theta_collection_script(params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    theta_min = params["theta_min"]
    theta_max = params["theta_max"]
    N_theta = params["N_theta"]

    results_dir = params["sim_anneal"]["results_dir"]
    save_dir = params["sim_anneal"]["save_dir"]
    collect_dir = save_dir*"/collect"
    file_prefix = params["sim_anneal"]["file_prefix"]

    if !isdir(collect_dir)
        mkpath(collect_dir)
    end
    
    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=4000M
    #SBATCH --time=00:15:00
    #SBATCH --job-name=collect_theta
    #SBATCH --output=/scratch/antony/slurm_out/%j_collect.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module load StdEnv/2023 julia/1.11.3

    # run collection script
    cd /home/antony/classical-mc
    julia --project=/home/antony/classical-mc -e "include(\\\"metropolis_pyrochlore.jl\\\"); include(\\\"write_hdf5.jl\\\"); collect_theta_sweep(\\\"$save_dir\\\", \\\"$file_prefix\\\", \\\"$collect_dir\\\", $theta_min, $theta_max, $N_theta)"

    # clean up
    cd $results_dir
    rm $file_prefix*

    cd $save_dir
    rm $file_prefix*
    """
end

function generate_theta_sweep_script(julia_script, params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    N_h = params["N_h"]
    theta_min = params["theta_min"]
    theta_max = params["theta_max"]
    N_theta = params["N_theta"]
    
    cpus_per_node = 192  # Check with: sinfo --Node --long
    cpus_per_theta = N_h  # e.g., 64
    thetas_per_node = div(cpus_per_node, cpus_per_theta)  # e.g., 192 ÷ 64 = 3
    N_array_jobs = ceil(Int, N_theta / thetas_per_node)

    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --array=0-$(N_array_jobs-1)
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=$(thetas_per_node * cpus_per_theta)
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=0  # Request all memory on the node
    #SBATCH --time=$(params["sim_anneal"]["job"]["time"])
    #SBATCH --job-name=$(params["sim_anneal"]["job"]["job_name"])_array
    #SBATCH --output=/scratch/antony/slurm_out/%A_%a_node.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module load StdEnv/2023 julia/1.11.3

    # Configuration
    CPUS_PER_THETA=$cpus_per_theta
    THETAS_PER_NODE=$thetas_per_node
    N_THETA=$N_theta

    # Calculate which theta indices this node should process
    START_IDX=\$((SLURM_ARRAY_TASK_ID * THETAS_PER_NODE))

    # Launch multiple jobs in parallel on this node
    for i in \$(seq 0 \$((THETAS_PER_NODE - 1))); do
        THETA_IDX=\$((START_IDX + i))
        
        if [ \$THETA_IDX -lt \$N_THETA ]; then
            # Launch each theta job with its own CPUs and output file, backgrounded
            srun --ntasks=\$CPUS_PER_THETA --exclusive --mem-per-cpu=$(params["sim_anneal"]["job"]["mem_per_cpu"]) \
                --output=/scratch/antony/slurm_out/%A_theta\${THETA_IDX}.out \
                julia --project=/home/antony/classical-mc $julia_script \
                --params_file $params_file_runtime --theta_index \$THETA_IDX &
        fi
    done

    # Wait for all background jobs to complete
    wait
    """
end

# Parse command-line arguments
if length(ARGS) < 1
    println("Usage: julia submit_job.jl [job_type] [account]")
    println("Job types: sim_anneal, parallel_temper, parallel_temper_job_array, theta_sweep")
    exit(1)
end

job_type = ARGS[1]
account = length(ARGS) > 1 ? ARGS[2] : "def-ybkim" #rrg-ybkim on fir, def-ybkim by default

# Common setup
dir_name = basename(pwd())
submit_dir = "/scratch/antony/$(dir_name)"
params_dest_dir = "/scratch/antony/param_files"
date_time = now()

# Determine params file and script based on job type
if job_type == "theta_sweep"
    params_file = "params_theta_sweep.yaml"
    params_file_runtime = "$(params_dest_dir)/params_theta_sweep_$(Dates.format(date_time, "yyyymmdd_HHMMSS")).yaml"
    julia_script = "/home/antony/classical-mc/sim_anneal_runner.jl"
else
    params_file = "params.yaml"
    params_file_runtime = "$(params_dest_dir)/params_$(Dates.format(date_time, "yyyymmdd_HHMMSS")).yaml"
    julia_script = job_type == "sim_anneal" ? "/home/antony/classical-mc/sim_anneal_runner.jl" : 
                   "/home/antony/classical-mc/parallel_tempering.jl"
end

cp(params_file, params_file_runtime)
params = YAML.load_file(params_file_runtime)
N_h = params["N_h"]

# Generate job-specific SLURM script
if job_type == "sim_anneal"
    slurm_script = generate_sim_anneal_script(julia_script, params_file_runtime, account)
    slurm_filename = "$(submit_dir)/submit_simulated_annealing.sh"
    
elseif job_type == "parallel_temper"
    slurm_script = generate_parallel_temper_script_single_node(julia_script, params_file_runtime, account)
    slurm_filename = "$(submit_dir)/submit_parallel_tempering.sh"

    collection_script = generate_parallel_temper_collection_script(params_file_runtime, account)
    collection_filename = "$(submit_dir)/submit_pt_collect.sh"

elseif job_type == "parallel_temper_job_array"
    slurm_script = generate_parallel_temper_script(julia_script, params_file_runtime, account, params["parallel_temper"]["job"]["h_points_per_node"])
    slurm_filename = "$(submit_dir)/submit_parallel_tempering.sh"

    collection_script = generate_parallel_temper_collection_script(params_file_runtime, account)
    collection_filename = "$(submit_dir)/submit_pt_collect.sh"
    
elseif job_type == "theta_sweep"
    slurm_script = generate_theta_sweep_script(julia_script, params_file_runtime, account)
    slurm_filename = "$(submit_dir)/submit_theta.sh"
    
    # Also generate collection script
    collection_script = generate_theta_collection_script(params_file_runtime, account)
    collection_filename = "$(submit_dir)/submit_theta_collect.sh"
    
else
    error("Unknown job type: $job_type")
end

# Write and submit
open(slurm_filename, "w") do f
    write(f, slurm_script)
end
run(`chmod +x $(slurm_filename)`)

cd(submit_dir)

try
    # Submit main job
    output = read(`sbatch $(slurm_filename)`, String)
    println(output)
    
    # For theta_sweep, also submit dependent collection job
    if job_type == "theta_sweep" || job_type == "parallel_temper" || job_type == "parallel_temper_job_array"
        # Extract job ID from output (format: "Submitted batch job 12345")
        job_id = match(r"(\d+)", output).match
        
        # Write collection script
        open(collection_filename, "w") do f
            write(f, collection_script)
        end
        run(`chmod +x $(collection_filename)`)
        
        # Submit collection job with dependency
        collection_output = read(`sbatch --dependency=afterok:$(job_id) $(collection_filename)`, String)
        println("Collection job: ", collection_output)
    end
catch e
    println("Error submitting job: $(e)")
end