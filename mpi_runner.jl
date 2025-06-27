using MPI

include("input_file.jl")

#number of processes
if length(ARGS) == 1
    n = parse(Int64, ARGS[1])
else
    n = 10
end

#save folder name
save_folder = "pt_out"

@time run(`$(mpiexec()) -n $n julia parallel_tempering.jl $save_folder`)

#h sweep
#=
for j in 1:N_h
    run(`$(mpiexec()) -n $n julia parallel_tempering.jl $save_folder $j`)
end
=#