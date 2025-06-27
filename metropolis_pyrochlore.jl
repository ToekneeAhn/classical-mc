using LinearAlgebra, StaticArrays, BinningAnalysis

include("observables.jl")

#local z axis on sublattice m in in column m+1
z_local = 1/sqrt(3)*[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]'
#local dipole moments
local_interactions = 1.0 .* [0 0 1; 0 0 1; 0 0 1; 0 0 1]'
#bond-dependent gamma factor
omega = exp(2*pi*im/3)
gamma_ij = [0 1 omega omega^2; 1 0 omega^2 omega; omega omega^2 0 1; omega^2 omega 1 0]

#sublattice-indexed coordinates (sic)
struct sic 
    r::Vector{Int64} #position of unit cell (i.e. sublattice 0)
    mu::Int64 #sublattice index 1,2,3,4
    N::Int64 #number of unit cells in each direction
end

#defines addition of sublattice-indexed coordinates (kinda) with multiple dispatch
#assume periodic boundary conditions
import Base: +, *
pos_mod(x::Vector{Int64}, m::Int64) = ((x.%m).+m).%m
+(r_mu::sic, v::Vector{Int64}) = sic(pos_mod(r_mu.r + v, r_mu.N), r_mu.mu, r_mu.N)

function get_sublattice(n::Int64, N::Int64)::Int64
    return div(n-1, N^3) + 1
end

#sic to 1D index
function flat_index_3D(r_mu::sic)::Int64
    nx, ny, nz = r_mu.r
    mu = r_mu.mu
    N = r_mu.N
    return N^2*nx + N*ny + nz + (mu-1)*N^3 + 1
end

#1D index to sic 
function sic_index_3D(n::Int64, N::Int64)::sic
    mu = get_sublattice(n, N)
    
    #label unit cell positions (nx,ny,nz) by the base N representation (nx ny nz)_N
    return sic(reverse(digits(n - 1 - (mu-1)*N^3, base=N, pad=3)), mu, N)
end

#6 neighbours of a pyrochlore lattice site in a tuple
function neighbours_pyro(n::Int64, N::Int64)::NTuple{6,Int64} 
    mu = get_sublattice(n, N)
    r_mu = sic_index_3D(n, N).r
    
    neighbours_flat = pos_mod([N^3, 2*N^3, 3*N^3].+(n-1), 4*N^3).+1 #flat indices of intra-tetrahedron neighbours
    
    ee = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'
    
    for l in 1:4
        if l != mu
            r_p = pos_mod(r_mu + ee[:, mu] - ee[:,l], N)  #inter-tetrahedron neighbours
            r_neighbour = sic(r_p, l, N)
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
function H_matrix_all(Js::NTuple{4,Float64})::SArray{Tuple{4,4,3,3},Float64}
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
    
    return E/N_sites
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
function sim_anneal!(spins::Array{Float64,2}, S::Float64, N::Int64, Js::NTuple{4,Float64}, h::Vector{Float64}, N_therm::Int64, N_det::Int64, probe_rate::Int64, overrelax_rate::Int64, T_i::Float64, T_f::Float64, schedule::Function)    
    N_sites = 4*N^3
    
    #precomputes interaction matrices for each bond, neighbours for each site, and zeeman interaction for each sublattice
    H_bond = H_matrix_all(Js)
    neighbours = neighbours_all(N_sites)
    zeeman = zeeman_field(h, z_local, local_interactions)
    
    N_probe = div(N_det, probe_rate)
    accept_count = [0]
    measurements = Any[]
    energies_therm = Float64[]
    energies = zeros(N_probe)
    avg_spin = zeros(N_probe, 3, 4)
    
    #metropolis + overrelaxation
    t = 0
    T = T_i 

    while T > T_f
        T = T_i*schedule(t) #assume schedule(0)=1
        for sweep in 1:N_therm
            if sweep % overrelax_rate == 0
                metropolis!(spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
            else
                overrelax_pyro!(spins, H_bond, neighbours, zeeman, N)
            end
        end

        println("acceptance rate at T=", T, ": ", accept_count[1]/(N_sites*N_therm/overrelax_rate)*100, "%")
        accept_count = [0] 

        push!(energies_therm, E_pyro(spins, H_bond, neighbours, zeeman, N))
        
        t += 1
    end
    
    for sweep in 1:N_det        
        det_update!(spins, S, H_bond, neighbours, zeeman, N)

        if sweep % probe_rate == 0
        #take measurements after thermalization every probe_rate sweeps
        #for magnetostriction, we just want the ensemble averaged spin for each sublattice
        #so N_det x 3 x 4 array of results for each rank
            meas_ind = div(sweep, probe_rate)
            avg_spin[meas_ind, :, :] = spin_expec(spins, N)
            energies[meas_ind] = E_pyro(spins, H_bond, neighbours, zeeman, N)
        end
    end        

    #takes the sample average 
    push!(measurements, 1/N_probe*sum(energies))
    push!(measurements, 1/N_probe*dropdims(sum(avg_spin, dims=1), dims=1))

    return [energies_therm; energies], measurements #measurements
end

#parallel tempering
function parallel_temper!(spins::Array{Float64,2}, S::Float64, N::Int64, Js::NTuple{4,Float64}, h::Vector{Float64}, rank::Int64, replica_exchange_rate::Int64, N_therm::Int64, N_det::Int64, probe_rate::Int64, overrelax_rate::Int64, temp::Vector{Float64})
    N_sites = 4*N^3
    
    #precomputes interaction matrices for each bond and neighbours for each site
    H_bond = H_matrix_all(Js)
    neighbours = neighbours_all(N_sites)
    zeeman = zeeman_field(h, z_local, local_interactions)
    
    T = temp[rank+1]
    N_sweeps = N_therm + N_det
    energies = zeros(N_sweeps)

    N_meas = div(N_det,probe_rate)
    measurements = zeros(N_meas, 3, 4)
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
            metropolis!(spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
        else
            overrelax_pyro!(spins, H_bond, neighbours, zeeman, N)
        end
        
        if sweep > N_therm && sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            #for magnetostriction, we just want the ensemble averaged spin for each sublattice
            #so N_det x 4 x 3 array of results for each rank
            meas_ind = div(sweep - N_therm, probe_rate)
            measurements[meas_ind, :,: ] = spin_expec(spins, N)
        end
        
        energies[sweep] = E_pyro(spins, H_bond, neighbours, zeeman, N)

        if sweep % replica_exchange_rate == 0
            #println(string(rank)*": it's time to swap!")

            #alternate between swap_type 0 and swap_type 1
            swap_type = div(sweep, replica_exchange_rate)%2
            accept_swap = replica_exchange!(rank, spins, energies[sweep], temp, swap_type)
            
            accept_count_swap += accept_swap
        end 
    end

    
    fname = replace(pwd(), "\\"=>"/")*"/pt_out/E_final_"*string(rank)*".txt"
    output_data = string(rank)*" "*string(T)*" "*string(energies[end])*"\n"
    write(fname, output_data)
    

    #computes standard error with log binning of depth L
    #want at least 30 bins, done automatically by BinningAnalysis
    err = zeros(3,4)
    for comp = 1:3, subl = 1:4
        dataset = measurements[:, comp, subl]
        err[comp, subl] = std_error(dataset)
    end
    
    #takes the sample average 
    measurements = sum(measurements, dims=1)[1,:,:]/N_meas
    
    return energies, measurements, err, accept_count_swap
end

#todo: is this implemented correctly?
#do this each iteration of the loop (i.e. when it's time to try swapping)
function replica_exchange!(rank::Int64, spins::Array{Float64,2}, E_rank::Float64, temp::Vector{Float64}, swap_type::Int64)
    #swap_type 0 pairs (01)(23)..., swap type 1 pairs 0(12)(34)...
    
    recv_data = copy(spins) #is this necessary?
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
        accept_prob = min(1.0, exp(delta_beta*delta_E))
        if rand(1)[1] < accept_prob
            accept_arr[1] = true
        end
        #println("rank ", r, " got delta_E = ", delta_E, ", delta_beta = ", delta_beta)
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
            #recv_data = MPI.recv(comm, source=partner_rank)
            spins = MPI.recv(comm, source=partner_rank)
            #print("rank ", r, " got ", recv_data)
        end
        #prevents edge case of (-1,0) swap if swap_type=1
        if rank%2 == 1-swap_type && partner_rank >= swap_type 
            #recv_data = MPI.recv(comm, source=partner_rank)
            spins = MPI.recv(comm, source=partner_rank)
            MPI.send(spins, comm, dest=partner_rank)
            #print("rank ", r, " got ", recv_data)
        end
    end

    return accept_arr
end

