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
    #SBATCH --output=/scratch/antony/slurm_out/slurm_%j.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module purge
    module load StdEnv/2023 julia/1.11.3

    srun julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime

    cd $results_dir
    rm $file_prefix*
    """
end

function generate_parallel_temper_script(julia_script, params_file_runtime, account)
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
    #SBATCH --output=/scratch/antony/slurm_out/slurm_%j.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module purge
    module load StdEnv/2023 julia/1.11.3

    for ((i=1; i<=$N_h; i++));
    do
        srun julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime --h_index \$i
    done

    cd $results_dir
    rm $file_prefix*
    """
end

function generate_theta_sweep_script(julia_script, params_file_runtime, account)
    params = YAML.load_file(params_file_runtime)
    N_h = params["N_h"]
    theta_min = params["theta_min"]
    theta_max = params["theta_max"]
    N_theta = params["N_theta"]
    
    return """
    #!/bin/bash
    #SBATCH --account=$account
    #SBATCH --array=0-$(N_theta-1)
    #SBATCH --ntasks=$N_h
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=$(params["sim_anneal"]["job"]["mem_per_cpu"])
    #SBATCH --time=$(params["sim_anneal"]["job"]["time"])
    #SBATCH --job-name=$(params["sim_anneal"]["job"]["job_name"])_array
    #SBATCH --output=/scratch/antony/slurm_out/slurm_%A_%a.out
    #SBATCH --mail-user=t.an@mail.utoronto.ca
    #SBATCH --mail-type=ALL

    module load StdEnv/2023 julia/1.11.3
    srun julia --project=/home/antony/classical-mc $julia_script --params_file $params_file_runtime --theta_index \$SLURM_ARRAY_TASK_ID
    """
end

# Parse command-line arguments
if length(ARGS) < 1
    println("Usage: julia submit_job.jl [job_type] [account]")
    println("Job types: sim_anneal, parallel_temper, theta_sweep")
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
    slurm_script = generate_parallel_temper_script(julia_script, params_file_runtime, account)
    slurm_filename = "$(submit_dir)/submit_parallel_tempering.sh"
    
elseif job_type == "theta_sweep"
    slurm_script = generate_theta_sweep_script(julia_script, params_file_runtime, account)
    slurm_filename = "$(submit_dir)/submit_theta.sh"
    
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
    output = read(`sbatch $(slurm_filename)`, String)
    println(output)
catch e
    println("Error submitting job: $(e)")
end