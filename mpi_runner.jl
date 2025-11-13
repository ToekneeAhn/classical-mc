using MPI

include("input_file.jl")

#number of processes
if length(ARGS) == 4
    save_folder = ARGS[1] #save folder name
    n = parse(Int64, ARGS[2]) #number of replicas
    h_direction = ARGS[3] #comma separated h field direction e.g. "1,1,1"
    load_path = ARGS[4]
else
    save_folder = "pt_out"
    n = 8
    h_direction = "1,1,1"
    load_path = ""
end

@time run(`$(mpiexec()) -n $n julia parallel_tempering.jl $save_folder $h_direction $load_path`)