using StaticArrays, BinningAnalysis

#sublattice-indexed pyrochlore coordinates (sipc)
struct SIPC
    r::Vector{Int64} #position of unit cell (i.e. sublattice 0)
    mu::Int64 #sublattice index 1,2,3,4
    N::Int64 #number of unit cells in each direction
end

#the physical system and lattice size
mutable struct SpinSystem
    spins::Matrix{Float64} #3 x N_sites; the spin configuration
    S::Float64 #spin length
    N::Int64 #number of unit cells in each direction
    N_sites::Int64 #total number of sites in lattice
    Js::Vector{Float64} #exchange parameters
    h::Vector{Float64} #external field
    delta_12::Vector{Float64} #quadratic zeeman field coupling
    disorder_strength::Float64 #Gamma parameter in Lorentzian distribution
    neighbours::Vector{NTuple{6,Int64}}
    H_bilinear::Vector{NTuple{6, SArray{Tuple{3,3},Float64,2,9}}}
    K::Complex{Float64} #cubic interaction parameter
    cubic_sites::Vector{NTuple{90,NTuple{3,Int64}}} #list of cubic interaction site tuples for each site
    H_cubic_sparse::Vector{NTuple{90,NTuple{2,Float64}}} #cubic interaction tensors stored sparsely as (K_313, K_323)
    #compute unique cubic triplets to avoid triple count in E_pyro
    unique_triplets::Vector{Tuple{Int64,Int64,Int64}} #list of unique cubic triplets in the lattice
    unique_H_cubic_vals::Vector{Tuple{Float64,Float64}} #corresponding (K_313, K_323) values for unique triplets
    # Pre-split cubic interaction lists by the role of the central site for local_field_pyro
    cubic_pairs_i::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as first index: store (j,k)
    cubic_pairs_j::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as second index: store (i,k)
    cubic_pairs_k::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as third index: store (i,j)
    zeeman_field::Vector{NTuple{3,Float64}}
end

#constructor without cubic interactions
function SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman_field)
    empty_triplets = Vector{NTuple{90,NTuple{3,Int64}}}()
    empty_pairs = Vector{NTuple{30,NTuple{2,Int64}}}()
    empty_H_cubic_sparse = Vector{NTuple{90,NTuple{2,Float64}}}()
    empty_unique_triplets = Vector{Tuple{3,Int64}}()
    empty_K_vals = Vector{Tuple{2,Float64}}()
    empty_K = 0 + 0im
    return SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength,
                      neighbours, H_bilinear, empty_K, empty_triplets, empty_H_cubic_sparse,
                      empty_unique_triplets, empty_K_vals,
                      empty_pairs, empty_pairs, empty_pairs, zeeman_field)
end

# observables calculated during simulation
mutable struct Observables
    energy::ErrorPropagator{Float64,32} 
    magnetization_global::Vector{ErrorPropagator{Float64,32}} # 3 components of magnetization in global frame
    magnetization_along_field::ErrorPropagator{Float64,32} # magnetization along external field direction
    local_spin::Matrix{ErrorPropagator{Float64,32}} # 3 components of average spin on 4 sublattices in local frame
    energy_spin_covariance::Matrix{ErrorPropagator{Float64,32}} # covariance between energy and local spin components, used for calculating dS/dT
    energy_quadrupolar_covariance::Vector{ErrorPropagator{Float64,32}} # covariance between energy and quadrupolar order parameters, used for calculating dQ/dT  
    output::Dict{String, Union{Float64, AbstractArray{Float64}}} # dictionary to store final results with error bars for output
    Observables() = new(ErrorPropagator(Float64, N_args=2), 
                        [ErrorPropagator(Float64, N_args=3) for i=1:3],                    
                        ErrorPropagator(Float64, N_args=3), 
                        [ErrorPropagator(Float64,N_args=2) for i=1:3,j=1:4], 
                        [ErrorPropagator(Float64,N_args=3) for i=1:3,j=1:4],
                        [ErrorPropagator(Float64,N_args=3) for i=1:length(Q_SPECS)],
                        Dict{String, Union{Float64, AbstractArray{Float64}}}())
end

#monte carlo simulation parameters
struct MCParams
    N_therm::Int64 #thermalization steps (both)
    N_det::Int64 #deterministic update steps (simulated anneal only)
    overrelax_rate::Int64 #ratio of overrelax to metropolis steps (both)
    N_meas::Int64 #measurement sweeps (parallel tempering only)
    probe_rate::Int64 #number of steps between measurements (parallel tempering only)
    replica_exchange_rate::Int64 #number of steps between replica exchange attempts (parallel tempering only)
    optimize_temperature_rate::Int64 #number of steps between temperature rank adjustments (parallel tempering only)
end

#everything packaged in one struct
mutable struct Simulation
    spin_system::SpinSystem
    T::Float64 #temperature
    parameters::MCParams
    observables::Observables
    replica_number::Int64 #keeps track of where the replicas go
    replica_label::String
end

# run parameters 
struct RunConfig
    Ts::Vector{Float64}
    h_direction::Vector{Float64}
    h_sweep::Vector{Float64}
    disorder_seed::Int64
end
