using LinearAlgebra, StaticArrays, BinningAnalysis

include("observables.jl")

#local z axis on sublattice m in in column m+1
z_local = 1/sqrt(3)*[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]'

#local dipole moments. for non-kramers, only the z component is dipolar
local_interactions = 1.0 .* [0 0 1; 0 0 1; 0 0 1; 0 0 1]'

#bond-dependent gamma factor
omega = exp(2*pi*im/3)
gamma_ij = [0 1 omega omega^2; 1 0 omega^2 omega; omega omega^2 0 1; omega^2 omega 1 0]

local_1 = [-2/sqrt(6) 1/sqrt(6) 1/sqrt(6); 0 -1/sqrt(2) 1/sqrt(2); 1/sqrt(3) 1/sqrt(3) 1/sqrt(3)]'
local_2 = [-2/sqrt(6) -1/sqrt(6) -1/sqrt(6); 0 1/sqrt(2) -1/sqrt(2); 1/sqrt(3) -1/sqrt(3) -1/sqrt(3)]'
local_3 = [2/sqrt(6) 1/sqrt(6) -1/sqrt(6); 0 -1/sqrt(2) -1/sqrt(2); -1/sqrt(3) 1/sqrt(3) -1/sqrt(3)]'
local_4 = [2/sqrt(6) -1/sqrt(6) 1/sqrt(6); 0 1/sqrt(2) 1/sqrt(2); -1/sqrt(3) -1/sqrt(3) 1/sqrt(3)]'
local_bases = [Matrix{Float64}(local_1), Matrix{Float64}(local_2), Matrix{Float64}(local_3), Matrix{Float64}(local_4)]

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
    #not sure about keeping these since they have to be computed later
    #neighbours::Matrix{Int64} #6 x N_sites; list of neighbour indices for each site
    #H_bond::Array{Float64, 4} #4 x 4 x 3 x 3; interaction matrices for each bond
    #h_site::Vector{NTuple{3,Float64}} #zeeman interaction on each sublattice
end

#monte carlo simulation parameters
struct MCParams
    N_therm::Int64 #thermalization steps (both)
    N_det::Int64 #deterministic update steps (simulated anneal only)
    overrelax_rate::Int64 #ratio of overrelax to metropolis steps (both)
    N_meas::Int64 #measurement sweeps (parallel tempering only)
    probe_rate::Int64 #number of steps between measurements (parallel tempering only)
    replica_exchange_rate::Int64 #number of steps between replica exchange attempts (parallel tempering only)
end

#everything packaged in one struct
mutable struct Simulation
    spin_system::SpinSystem
    T::Float64 #temperature
    parameters::MCParams
    observables::Observables
end

#assume periodic boundary conditions, so take unit cell positions mod N
function pos_mod(x::Vector{Int64}, m::Int64)::Vector{Int64}
    return ((x.%m).+m).%m
end

function get_sublattice(n::Int64, N::Int64)::Int64 
    return div(n-1, N^3) + 1
end

#SIPC to 1D index
function flat_index_3D(r_mu::SIPC)::Int64
    nx, ny, nz = r_mu.r
    mu = r_mu.mu
    N = r_mu.N
    return N^2*nx + N*ny + nz + (mu-1)*N^3 + 1
end

#1D index to SIPC 
function sipc_index_3D(n::Int64, N::Int64)::SIPC
    mu = get_sublattice(n, N)
    
    #label unit cell positions (nx,ny,nz) by the base N representation (nx ny nz)_N
    return SIPC(reverse(digits(n - 1 - (mu-1)*N^3, base=N, pad=3)), mu, N)
end

#6 neighbours of a pyrochlore lattice site in a tuple
function neighbours_pyro(n::Int64, N::Int64)::NTuple{6,Int64} 
    mu = get_sublattice(n, N)
    r_mu = sipc_index_3D(n, N).r
    
    neighbours_flat = pos_mod([N^3, 2*N^3, 3*N^3].+(n-1), 4*N^3).+1 #flat indices of intra-tetrahedron neighbours
    
    ee = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'
    
    for l in 1:4
        if l != mu
            r_p = pos_mod(r_mu + ee[:, mu] - ee[:,l], N)  #inter-tetrahedron neighbours
            r_neighbour = SIPC(r_p, l, N)
            append!(neighbours_flat, flat_index_3D(r_neighbour))
        end
    end 
    return Tuple(neighbours_flat)
end

function neighbours_all(N_sites)
    coord_num = 6
    
    neighbours = NTuple{coord_num, Int64}[]
    
    for n = 1:N_sites
        push!(neighbours, neighbours_pyro(n, N))
    end
    
    return neighbours
end

#S_new is a tuple for performance purposes
function set_spin!(spins::Array{Float64,2}, S_new::NTuple{3,Float64}, site::Int64)
    @inbounds spins[1, site] = S_new[1]
    @inbounds spins[2, site] = S_new[2]
    @inbounds spins[3, site] = S_new[3]
end

function get_spin(spins::Array{Float64,2}, site::Int64)::NTuple{3, Float64}
    @inbounds return (spins[1, site], spins[2, site], spins[3, site])
end

#4 x 4 x 3 x 3 array of interaction matrices (3x3) connecting sublattices
function H_matrix_all(Js::Vector{Float64})::SArray{Tuple{4,4,3,3},Float64}
    J_zz, J_pm, J_pmpm, J_zpm = Js

    H_bond = zeros(4,4,3,3)
    T = [1 im 0; 1 -im 0; 0 0 1] #rotates to (S^+, S^-, S^z) basis
    
    for sub_i in 1:4
        for sub_j in 1:4
            gamma = gamma_ij[sub_i, sub_j]
            zeta = -conj(gamma)
            
            if sub_i != sub_j
                H_bond[sub_i, sub_j, :, :] .= conj(T)' * [J_pmpm*gamma -J_pm J_zpm*zeta; -J_pm J_pmpm*conj(gamma) J_zpm*conj(zeta); J_zpm*zeta J_zpm*conj(zeta) J_zz] * T
            end
        end
    end

    #use static array for performance purposes because H_bond does not change
    return SArray{Tuple{4,4,3,3},Float64}(reinterpret(Float64, real(H_bond)))
end

function zeeman_field(h, local_bases, local_interactions)::Vector{NTuple{3,Float64}}
    zeeman_eff = NTuple{3,Float64}[]
    for mu in 1:4
        push!(zeeman_eff, Tuple((h' * local_bases[:, mu])*local_interactions[:, mu]))
    end
    #vector of tuples is faster to index into later
    return zeeman_eff
end

#h_loc at site i is dH/dS_i
function local_field_pyro(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours_n::NTuple{6,Int64}, zeeman::Vector{NTuple{3,Float64}}, n::Int64, N::Int64)::NTuple{3,Float64}
    sublattice = get_sublattice(n, N)
    
    #avoid matrix multiplication for performance purposes
    Hx=0.0
    Hy=0.0
    Hz=0.0
    
    for m in neighbours_n
        sx, sy, sz = get_spin(spins, m)
        H_ij = Matrix{Float64}
        @inbounds H_ij = H_bond[sublattice, get_sublattice(m, N), :, :]
        @inbounds Hx += H_ij[1,1] * sx + H_ij[1,2] * sy + H_ij[1,3] * sz
        @inbounds Hy += H_ij[2,1] * sx + H_ij[2,2] * sy + H_ij[2,3] * sz
        @inbounds Hz += H_ij[3,1] * sx + H_ij[3,2] * sy + H_ij[3,3] * sz
    end

    h_z = zeeman[sublattice]
    return (Hx-h_z[1], Hy-h_z[2], Hz-h_z[3])
end

#energy of spin configuration with periodic boundary conditions
function E_pyro(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)::Float64
    N_sites = 4*N^3
    E = 0.0
    
    for n in 1:N_sites
        #quadratic interaction, divide by 2 because each bond counted twice
        sublattice = get_sublattice(n, N)
        S_n = get_spin(spins, n)

        for m in neighbours[n]
            #use a view to avoid allocating an array
            @inbounds E += 0.5 * dot(S_n, H_bond[sublattice, get_sublattice(m, N),:,:] * view(spins, :, m))
        end

        #zeeman contribution
        @inbounds E += - dot(zeeman[sublattice], S_n)
    end
    
    #total energy, not energy per site
    return E
end

function energy_difference_pyro(spins::Array{Float64,2}, old_spin::NTuple{3,Float64}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours_n::NTuple{6,Int64}, zeeman::Vector{NTuple{3,Float64}}, n::Int64, N::Int64)::Float64
    h_loc = local_field_pyro(spins, H_bond, neighbours_n, zeeman, n, N)
    E_new = dot(get_spin(spins, n), h_loc)
    E_old = dot(old_spin, h_loc)

    return E_new - E_old 
end

#intializes a random spin configuration with shape 3 x 4N^3
function spins_initial_pyro(N::Int64, S::Float64)::Array{Float64,2}
    N_sites = 4*N^3
    spins = rand(3, N_sites)
    for j=1:N_sites
        spins[:,j] .*= S/norm(spins[:,j]) #normalizes each spin to length S
        #spins[j,:] = [0,0,1]
    end
    return spins
end

#picks a point on the unit sphere uniformly and returns Cartesian coordinates (Sx,Sy,Sz)
#then scales magnitude by S
function sphere_pick(S::Float64)::NTuple{3,Float64}
    #faster rng? lehmer prng
    #gaussian sphere picking for lower temperatures
    phi = 2*pi*rand()
    z = 2*rand() - 1
    return S .* (sqrt(1-z^2)*cos(phi), sqrt(1-z^2)*sin(phi), z)
end

#metropolis algorithm with deterministic updates (aligning spins to their local field)
function metropolis!(spins::Array{Float64,2}, accept_count::Array{Int64,1}, S::Float64, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64, T::Float64)
    N_sites = 4*N^3
    
    for site in 1:N_sites #1 sweep has N_sites steps
        i = rand(1:N_sites)        
        old_spin = get_spin(spins, i) #copy previous configuration 
        set_spin!(spins, sphere_pick(S), i)
        
        delta_E = energy_difference_pyro(spins, old_spin, H_bond, neighbours[i], zeeman, i, N) 
        
        #accept if energy is lower (delta E < 0) or with probability given by Boltzmann weight
        no_accept = delta_E > 0 && (rand() > exp(-delta_E/T))
        accept_count .+= 1-no_accept
        
        #otherwise revert to previous configuration
        if no_accept 
            set_spin!(spins, old_spin, i)
        end
    end 
end

function det_update!(spins::Array{Float64,2}, S::Float64, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)
    N_sites = 4*N^3
    
    for n in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, neighbours[n], zeeman, n, N)
        set_spin!(spins, -S .* h_loc ./ sqrt(h_loc[1]^2+h_loc[2]^2+h_loc[3]^2), n)
    end
end

#overrelaxation (microcanonical sweep) which reflects each spin about the local field
function overrelax_pyro!(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)
    N_sites = 4*N^3
    
    for n in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, neighbours[n], zeeman, n, N)
        S_n = get_spin(spins, n)
        S_new = 2.0 * dot(S_n, h_loc)/(h_loc[1]^2+h_loc[2]^2+h_loc[3]^2) .* h_loc .- S_n
        set_spin!(spins, S_new, n)
    end
end

#simulated annealing with overrelaxation for N_therm sweeps, then deterministic updates for N_det sweeps
function sim_anneal!(mc::Simulation, T_i::Float64, schedule::Function)    
    N_therm = mc.parameters.N_therm
    N_det = mc.parameters.N_det
    overrelax_rate = mc.parameters.overrelax_rate

    N = mc.spin_system.N
    S = mc.spin_system.S
    N_sites = mc.spin_system.N_sites
    
    #precomputes interaction matrices for each bond, neighbours for each site, and zeeman interaction for each sublattice
    H_bond = H_matrix_all(mc.spin_system.Js)
    neighbours = neighbours_all(mc.spin_system.N_sites)
    zeeman = zeeman_field(mc.spin_system.h, z_local, local_interactions)
    
    accept_count = [0]
    measurements = Any[]
    energies_therm = Float64[]
    
    #metropolis + overrelaxation
    t = 0
    T = T_i 
    T_f = mc.T #set the T parameter to the target temp

    while T > T_f
        T = T_i*schedule(t) #assume schedule(0)=1
        for sweep in 1:N_therm
            if sweep % overrelax_rate == 0
                metropolis!(mc.spin_system.spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
            else
                overrelax_pyro!(mc.spin_system.spins, H_bond, neighbours, zeeman, N)
            end
        end

        println("acceptance rate at T=", T, ": ", accept_count[1]/(N_sites*N_therm/overrelax_rate)*100, "%")
        accept_count = [0] 

        push!(energies_therm, E_pyro(mc.spin_system.spins, H_bond, neighbours, zeeman, N))
        
        t += 1
    end
    
    for sweep in 1:N_det        
        det_update!(mc.spin_system.spins, S, H_bond, neighbours, zeeman, N)    
    end        
    
    #each simulated annealing run constitues one measurement (at the end)
    E = E_pyro(mc.spin_system.spins, H_bond, neighbours, zeeman, N)
    avg_spin = spin_expec(mc.spin_system.spins, N)
    m = norm(magnetization_global(avg_spin, local_bases, mc.spin_system.h))

    push!(mc.observables.energy, E, E^2)
    push!(mc.observables.magnetization, m, m^2)
    push!(mc.observables.avg_spin, avg_spin)

    #thermalization energies 
    return energies_therm
end

function parallel_temper!(mc::Simulation, rank::Int64, temp::Vector{Float64})
    N_therm = mc.parameters.N_therm
    overrelax_rate = mc.parameters.overrelax_rate
    N_meas = mc.parameters.N_meas
    probe_rate = mc.parameters.probe_rate
    replica_exchange_rate = mc.parameters.replica_exchange_rate
    
    N = mc.spin_system.N
    S = mc.spin_system.S
    
    #precomputes interaction matrices for each bond, neighbours for each site, and zeeman interaction for each sublattice
    H_bond = H_matrix_all(mc.spin_system.Js)
    neighbours = neighbours_all(mc.spin_system.N_sites)
    zeeman = zeeman_field(mc.spin_system.h, z_local, local_interactions)
    
    T = temp[rank+1]
    N_sweeps = N_therm + N_meas
    energies = zeros(N_sweeps)
    
    accept_count = [0] #counts successful metropolis steps (not currently outputted)
    accept_count_swap = [0] #counts number of successful swaps

    for sweep in 1:N_sweeps
        #do deterministic updates on lowest temperature rank (after thermalization)?
        #=
        if rank == 0 && sweep > N_therm
            det_update!(spins, S, H_bond, neighbours, zeeman, N)
        end
        =#

        #do overrelaxation and metropolis with relative frequency overrelax_rate
        if sweep % overrelax_rate == 0
            metropolis!(mc.spin_system.spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
        else
            overrelax_pyro!(mc.spin_system.spins, H_bond, neighbours, zeeman, N)
        end
        
        E = E_pyro(mc.spin_system.spins, H_bond, neighbours, zeeman, N)
        energies[sweep] = E

        if sweep > N_therm && sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            avg_spin = spin_expec(mc.spin_system.spins, N)
            m = norm(magnetization_global(avg_spin, local_bases, mc.spin_system.h))
            #do we have to use norm(m)? 

            push!(mc.observables.energy, E, E^2)
            push!(mc.observables.magnetization, m, m^2)
            push!(mc.observables.avg_spin, avg_spin)
        end 

        if sweep % replica_exchange_rate == 0
            #println(string(rank)*": it's time to swap!")

            #alternate between swap_type 0 and swap_type 1
            swap_type = div(sweep, replica_exchange_rate)%2
            #swap_type = 0
            accept_swap = replica_exchange!(rank, mc.spin_system.spins, E, temp, swap_type)
            
            accept_count_swap += accept_swap
        end 
    end

    #=
    fname = replace(pwd(), "\\"=>"/")*"/pt_out/E_final_"*string(rank)*".txt"
    output_data = string(rank)*" "*string(T)*" "*string(energies[end])*"\n"
    write(fname, output_data)
    =#
    
    return energies, accept_count_swap
end

#do this each iteration of the loop (i.e. when it's time to try swapping)
function replica_exchange!(rank::Int64, spins::Array{Float64,2}, E_rank::Float64, temp::Vector{Float64}, swap_type::Int64)
    #swap_type=0 pairs (01)(23)..., swap_type=1 pairs 0(12)(34)...
    
    partner_rank = iseven(rank) ? rank+(-1)^swap_type : rank-(-1)^swap_type
    accept_arr = [false]

    #get partner energy and compute delta_E, delta_beta
    if rank%2 == swap_type && partner_rank < comm_size 
        E_partner = MPI.recv(comm, source=partner_rank)
        delta_E = E_partner - E_rank
        delta_beta = 1/temp[partner_rank+1] - 1/temp[rank+1]
    elseif rank%2 == 1-swap_type && partner_rank >= swap_type
        MPI.send(E_rank, comm, dest=partner_rank)
    end

    #decide if swapping
    if rank%2 == swap_type && partner_rank < comm_size 
        accept_prob = exp(delta_beta*delta_E)
        if rand() < accept_prob
            accept_arr[1] = true
        end        
        MPI.send(accept_arr, comm, dest=partner_rank)
    elseif rank%2 == 1-swap_type && partner_rank >= swap_type
        accept_arr = MPI.recv(comm, source=partner_rank)
    end

    #it's time for the swap!
    if accept_arr[1]    
        #prevents edge case of (R-1,R) swap if swap_type=1
        if rank%2 == swap_type && partner_rank < comm_size 
            MPI.send(spins, comm, dest=partner_rank)
            #receive buffer?
            spins = MPI.recv(comm, source=partner_rank)
        end
        #prevents edge case of (-1,0) swap if swap_type=1
        if rank%2 == 1-swap_type && partner_rank >= swap_type 
            spins = MPI.recv(comm, source=partner_rank)
            MPI.send(spins, comm, dest=partner_rank)
        end
    end

    return accept_arr
end
