using MPI

include("input_file.jl")

#number of processes
if length(ARGS) == 1
    n = parse(Int64, ARGS[1])
else
    n = 4
end

@time run(`$(mpiexec()) -n $n julia parallel_tempering.jl`)

#h sweep where the sweep values are defined in input_file.jl
#=
for j in 1:N_h
    run(`$(mpiexec()) -n $n julia parallel_tempering.jl $j`)
end
=#