using MPI

include("input_file.jl")

#number of processes
if length(ARGS) == 3
    save_folder = ARGS[1] #save folder name
    n = parse(Int64, ARGS[2]) #number of replicas
    h_direction = ARGS[3] #comma separated h field direction e.g. "1,1,1"
else
    save_folder = "pt_out_test"
    n = 8
    h_direction = "1,1,1"
end

@time run(`$(mpiexec()) -n $n julia parallel_tempering.jl $save_folder $h_direction`)