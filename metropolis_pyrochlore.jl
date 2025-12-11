using LinearAlgebra, StaticArrays, BinningAnalysis, Random, MPI
using Interpolations, ForwardDiff, Integrals, Printf 

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
    delta_12::Vector{Float64} #quadratic zeeman field coupling
    disorder_strength::Float64 #Gamma parameter in Lorentzian distribution
    neighbours::Vector{NTuple{6,Int64}}
    H_bilinear::Vector{NTuple{6, SArray{Tuple{3,3},Float64,2,9}}}
    K::Complex{Float64} #cubic interaction parameter
    cubic_sites::Vector{NTuple{90,NTuple{3,Int64}}} #list of cubic interaction site tuples for each site
    H_cubic::Vector{NTuple{90,SArray{Tuple{3,3,3},Float64,3,27}}} #cubic interaction tensors for each cubic triplet
    # Pre-split cubic interaction lists by the role of the central site (for branch-free local field)
    cubic_pairs_i::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as first index: store (j,k)
    cubic_pairs_j::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as second index: store (i,k)
    cubic_pairs_k::Vector{NTuple{30,NTuple{2,Int64}}} # for site n as third index: store (i,j)
    zeeman_field::Vector{NTuple{3,Float64}}
end

#constructor without cubic interactions
function SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman_field)
    empty_triplets = Vector{NTuple{90,NTuple{3,Int64}}}()   
    empty_pairs = Vector{NTuple{30,NTuple{2,Int64}}}()   
    empty_H_cubic = Vector{NTuple{90,SArray{Tuple{3,3,3},Float64,3,27}}}()
    empty_K = 0 + 0im
    return SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength,
                      neighbours, H_bilinear, empty_K, empty_triplets, empty_H_cubic,
                      empty_pairs, empty_pairs, empty_pairs, zeeman_field)
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

function cubic_sites_pyro(i::Int64, N::Int64)::NTuple{90,NTuple{3,Int64}}
    neigh_i = neighbours_pyro(i, N)                # 6 neighbours of i
    role_first  = NTuple{3,Int64}[]                # (i, j, k)
    role_second = NTuple{3,Int64}[]                # (j, i, j2)
    role_third  = NTuple{3,Int64}[]                # (k, j, i)

    sizehint!(role_first, 30)
    sizehint!(role_second, 30)
    sizehint!(role_third, 30)

    # Collect endpoint and middle triplets separately
    @inbounds for j in neigh_i
        neigh_j = neighbours_pyro(j, N)
        # i as first and third positions (endpoint triplets)
        @inbounds for k in neigh_j
            if k != i
                push!(role_first,  (i, j, k))   # i first
                push!(role_third,  (k, j, i))   # i third
            end
        end
        # i as second position (middle triplets)
        @inbounds for j2 in neigh_i
            if j2 != j
                push!(role_second, (j, i, j2))  # i second
            end
        end
    end

    @assert length(role_first)  == 30
    @assert length(role_second) == 30
    @assert length(role_third)  == 30

    all_triplets = vcat(role_first, role_second, role_third)
    return Tuple(all_triplets)::NTuple{90,NTuple{3,Int64}}
end 

function cubic_sites_all(N::Int64, N_sites::Int64)
    cubic_sites = Vector{NTuple{90,NTuple{3,Int64}}}(undef, N_sites)

    for n in 1:N_sites
        cubic_sites[n] = cubic_sites_pyro(n, N)
    end

    return cubic_sites
end

"""
    cubic_pairs_split_all(cubic_sites, N_sites)

Split each site's 90 cubic triplets into three role-specific lists of 30 pairs:
 - cubic_pairs_i[n]: pairs (j,k) when triplet is (n,j,k)
 - cubic_pairs_j[n]: pairs (i,k) when triplet is (i,n,k)
 - cubic_pairs_k[n]: pairs (i,j) when triplet is (i,j,n)
"""
function cubic_pairs_split_all(cubic_sites::Vector{NTuple{90,NTuple{3,Int64}}}, N_sites::Int64)
    pairs_i = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)
    pairs_j = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)
    pairs_k = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)
    
    for n in 1:N_sites
        cs = cubic_sites[n]
        pairs_i[n] = Tuple([triplet[2:3] for triplet in cs[1:30]])
        pairs_j[n] = Tuple([ (triplet[1],triplet[3]) for triplet in cs[31:60]])
        pairs_k[n] = Tuple([triplet[1:2] for triplet in cs[61:90]])
    end
    
    return pairs_i, pairs_j, pairs_k
end

function cubic_tensors_all(K::Complex{Float64}, N::Int64, N_sites::Int64)::Vector{NTuple{90,SArray{Tuple{3,3,3}, Float64, 3, 27}}}
    cubic_tensors = Vector{NTuple{90,SArray{Tuple{3,3,3}, Float64, 3, 27}}}(undef, N_sites)
    for n in 1:N_sites
        cubic_sites_n = cubic_sites_pyro(n, N)
        cubic_tensors_n = SArray{Tuple{3,3,3}, Float64, 3, 27}[]

        for triplet in cubic_sites_n
            K_cubic = zeros(3,3,3)
            
            i, j, k = triplet
            sub_i = get_sublattice(i, N)
            sub_j = get_sublattice(j, N)
            sub_k = get_sublattice(k, N)

            phase = gamma_ij[sub_i, sub_j] * gamma_ij[sub_j, sub_k] #i hope this is correct lol
            
            K_cubic[3,1,3] = 2*real(K * phase)
            K_cubic[3,2,3] = -2*imag(K * phase)
            
            #there should be 3 indepdendent constants (K1, K2, K3) for the three independent types of "trimers" 
            #assume they are the same for now
            push!(cubic_tensors_n, SArray{Tuple{3,3,3}, Float64, 3, 27}(K_cubic))
        end

        cubic_tensors[n] = Tuple(cubic_tensors_n)
    end

    return cubic_tensors
end

function neighbours_all(N::Int64, N_sites::Int64)
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

#the 3x3 bilinear interaction matrices for all bonds
function H_bilinear_all(Js::Vector{Float64}, N::Int64, N_sites::Int64)
    J_zz, J_pm, J_pmpm, J_zpm = Js

    H_bilinear = Vector{NTuple{length(neighbours_pyro(1, N)), SArray{Tuple{3,3},Float64,2,9}}}() #list of tuples of interaction matrices for each site
    T = [1 im 0; 1 -im 0; 0 0 1] #rotates to (S^+, S^-, S^z) basis
    
    for n in 1:N_sites
        neighbours_n = neighbours_pyro(n, N)
        H_bilinear_n = []

        for m in eachindex(neighbours_n)
            push!(H_bilinear_n, 
                begin
                    sub_i = get_sublattice(n, N)
                    sub_j = get_sublattice(neighbours_n[m], N)
                    gamma = gamma_ij[sub_i, sub_j]
                    zeta = -conj(gamma)
                    
                    if sub_i != sub_j
                        SArray{Tuple{3,3},Float64,2,9}(conj(T)' * [J_pmpm*gamma -J_pm J_zpm*zeta; -J_pm J_pmpm*conj(gamma) J_zpm*conj(zeta); J_zpm*zeta J_zpm*conj(zeta) J_zz] * T)
                    else
                        SArray{Tuple{3,3},Float64,2,9}(zeros(3,3))
                    end
            end)
        end

        push!(H_bilinear, Tuple(H_bilinear_n))
    end

    return H_bilinear
end

function zeeman_field_random(h, z_local, local_interactions, delta_12, G, N_sites, seed=123, breaking_field=[zeros(3), zeros(3), zeros(3), zeros(3)])::Vector{NTuple{3,Float64}}
    Random.seed!(seed)
    
    zeeman_eff = NTuple{3,Float64}[]
    for mu in 1:4
        h_z = (h' * z_local[:, mu]) .* local_interactions[:, mu]
        h_mu = local_bases[mu]' * h
        h_xy_quadratic = delta_12[1] .* (h_mu[1] * h_mu[3], h_mu[2] * h_mu[3], 0.0) .+ delta_12[2] .* (h_mu[2]^2 - h_mu[1]^2, 2.0 *h_mu[1] * h_mu[2], 0.0)
        h_xy_breaking = breaking_field[mu]

        for n in 1:N_sites/4
            random_strength = G*tan(pi*(rand()-0.5)) #draws from a lorentzian distribution with pdf p(h) = G/pi * 1/(G^2+h^2)
            random_phase = 2*pi*rand()

            h_xy_random = random_strength .* (cos(random_phase), sin(random_phase), 0.0)
            
            push!(zeeman_eff, Tuple(h_z .+ h_xy_quadratic .+ h_xy_random .+ h_xy_breaking))
        end
    end
    #vector of tuples is faster to index into later
    return zeeman_eff
end

@inline function local_field_pyro(sys::SpinSystem, n::Int64)::NTuple{3,Float64}
    @inbounds begin
        neighs = sys.neighbours[n]          # NTuple{6,Int}
        Hs     = sys.H_bilinear[n]          # NTuple{6, SMatrix{3,3}}
        S      = sys.spins                  # 3×N_sites matrix

        Hx = 0.0; Hy = 0.0; Hz = 0.0

        # Fixed degree: 6 neighbours. Use literal indices for SMatrix getindex.
        for k in eachindex(neighs)
            m  = neighs[k]
            sx = S[1, m]; sy = S[2, m]; sz = S[3, m]
            H  = Hs[k]

            h11 = H[1,1]; h12 = H[1,2]; h13 = H[1,3]
            h21 = H[2,1]; h22 = H[2,2]; h23 = H[2,3]
            h31 = H[3,1]; h32 = H[3,2]; h33 = H[3,3]

            Hx += h11*sx + h12*sy + h13*sz
            Hy += h21*sx + h22*sy + h23*sz
            Hz += h31*sx + h32*sy + h33*sz
        end

        # Cubic part (kept as-is, but bounds checks off)
        if !isempty(sys.H_cubic)            
            H_cubic = sys.H_cubic[n]

            pairs_i = sys.cubic_pairs_i[n]
            pairs_j = sys.cubic_pairs_j[n]
            pairs_k = sys.cubic_pairs_k[n]

            for p in eachindex(pairs_i)
                K = H_cubic[p]
                j, k = pairs_i[p]
                Sj1 = S[1, j]; Sj2 = S[2, j]; Sj3 = S[3, j]
                Sk1 = S[1, k]; Sk2 = S[2, k]; Sk3 = S[3, k]
                # c=1
                Hx += (K[1,1,1]*Sj1 + K[1,2,1]*Sj2 + K[1,3,1]*Sj3) * Sk1
                Hy += (K[2,1,1]*Sj1 + K[2,2,1]*Sj2 + K[2,3,1]*Sj3) * Sk1
                Hz += (K[3,1,1]*Sj1 + K[3,2,1]*Sj2 + K[3,3,1]*Sj3) * Sk1
                # c=2
                Hx += (K[1,1,2]*Sj1 + K[1,2,2]*Sj2 + K[1,3,2]*Sj3) * Sk2
                Hy += (K[2,1,2]*Sj1 + K[2,2,2]*Sj2 + K[2,3,2]*Sj3) * Sk2
                Hz += (K[3,1,2]*Sj1 + K[3,2,2]*Sj2 + K[3,3,2]*Sj3) * Sk2
                # c=3
                Hx += (K[1,1,3]*Sj1 + K[1,2,3]*Sj2 + K[1,3,3]*Sj3) * Sk3
                Hy += (K[2,1,3]*Sj1 + K[2,2,3]*Sj2 + K[2,3,3]*Sj3) * Sk3
                Hz += (K[3,1,3]*Sj1 + K[3,2,3]*Sj2 + K[3,3,3]*Sj3) * Sk3
            end
            
            for p in eachindex(pairs_j)
                K = H_cubic[p+30]
                i, k = pairs_j[p]
                Si1 = S[1, i]; Si2 = S[2, i]; Si3 = S[3, i]
                Sk1 = S[1, k]; Sk2 = S[2, k]; Sk3 = S[3, k]
                # c=1
                Hx += (K[1,1,1]*Si1 + K[2,1,1]*Si2 + K[3,1,1]*Si3) * Sk1
                Hy += (K[1,2,1]*Si1 + K[2,2,1]*Si2 + K[3,2,1]*Si3) * Sk1
                Hz += (K[1,3,1]*Si1 + K[2,3,1]*Si2 + K[3,3,1]*Si3) * Sk1
                # c=2
                Hx += (K[1,1,2]*Si1 + K[2,1,2]*Si2 + K[3,1,2]*Si3) * Sk2
                Hy += (K[1,2,2]*Si1 + K[2,2,2]*Si2 + K[3,2,2]*Si3) * Sk2
                Hz += (K[1,3,2]*Si1 + K[2,3,2]*Si2 + K[3,3,2]*Si3) * Sk2
                # c=3
                Hx += (K[1,1,3]*Si1 + K[2,1,3]*Si2 + K[3,1,3]*Si3) * Sk3
                Hy += (K[1,2,3]*Si1 + K[2,2,3]*Si2 + K[3,2,3]*Si3) * Sk3
                Hz += (K[1,3,3]*Si1 + K[2,3,3]*Si2 + K[3,3,3]*Si3) * Sk3
            end
            
            for p in eachindex(pairs_k)
                K = H_cubic[p+60]
                i, j = pairs_k[p]
                Si1 = S[1, i]; Si2 = S[2, i]; Si3 = S[3, i]
                Sj1 = S[1, j]; Sj2 = S[2, j]; Sj3 = S[3, j]
                # c = 1 (Hx)
                Hx += (K[1,1,1]*Si1 + K[2,1,1]*Si2 + K[3,1,1]*Si3) * Sj1
                Hx += (K[1,2,1]*Si1 + K[2,2,1]*Si2 + K[3,2,1]*Si3) * Sj2
                Hx += (K[1,3,1]*Si1 + K[2,3,1]*Si2 + K[3,3,1]*Si3) * Sj3
                # c = 2 (Hy)
                Hy += (K[1,1,2]*Si1 + K[2,1,2]*Si2 + K[3,1,2]*Si3) * Sj1
                Hy += (K[1,2,2]*Si1 + K[2,2,2]*Si2 + K[3,2,2]*Si3) * Sj2
                Hy += (K[1,3,2]*Si1 + K[2,3,2]*Si2 + K[3,3,2]*Si3) * Sj3
                # c = 3 (Hz)
                Hz += (K[1,1,3]*Si1 + K[2,1,3]*Si2 + K[3,1,3]*Si3) * Sj1
                Hz += (K[1,2,3]*Si1 + K[2,2,3]*Si2 + K[3,2,3]*Si3) * Sj2
                Hz += (K[1,3,3]*Si1 + K[2,3,3]*Si2 + K[3,3,3]*Si3) * Sj3
            end
            
            #conceptually, the above code is equivalent to this comment block
            #optimizations come from precompting the pairs (no if statements) and unrolling loops
            
            #=
            H_cubic = sys.H_cubic[n]
            
            for (triplet_idx, triplet) in enumerate(sys.cubic_sites[n])
                i, j, k = triplet
                Si1 = S[1, i]; Si2 = S[2, i]; Si3 = S[3, i]
                Sj1 = S[1, j]; Sj2 = S[2, j]; Sj3 = S[3, j]
                Sk1 = S[1, k]; Sk2 = S[2, k]; Sk3 = S[3, k]
                
                Si = (Si1, Si2, Si3)
                Sj = (Sj1, Sj2, Sj3)
                Sk = (Sk1, Sk2, Sk3)

                K = H_cubic[triplet_idx]
                if i == n
                    # n is i
                    for a in 1:3, b in 1:3
                        Hx += K[1,a,b]*Sj[a]*Sk[b]
                        Hy += K[2,a,b]*Sj[a]*Sk[b]
                        Hz += K[3,a,b]*Sj[a]*Sk[b]
                    end
                elseif j == n
                    # n is j
                    for a in 1:3, b in 1:3
                        Hx += K[a,1,b]*Si[a]*Sk[b]
                        Hy += K[a,2,b]*Si[a]*Sk[b]
                        Hz += K[a,3,b]*Si[a]*Sk[b]
                    end
                    
                elseif k == n
                    # n is k
                    for a in 1:3, b in 1:3
                        Hx += K[a,b,1]*Si[a]*Sj[b]
                        Hy += K[a,b,2]*Si[a]*Sj[b]
                        Hz += K[a,b,3]*Si[a]*Sj[b]
                    end
                end
            end
            =#
        end

        h = sys.zeeman_field[n]
        return (Hx - h[1], Hy - h[2], Hz - h[3])
    end
end

function E_pyro(sys::SpinSystem)::Float64
    E_bilinear = 0.0
    E_cubic = 0.0
    E_zeeman = 0.0
    
    for n in 1:sys.N_sites
        #quadratic interaction, divide by 2 because each bond counted twice
        S_n = get_spin(sys.spins, n)

        for m in eachindex(sys.neighbours[n])
            S_m = get_spin(sys.spins, sys.neighbours[n][m])

            E_bilinear += S_n[1] * sys.H_bilinear[n][m][1,1] * S_m[1] + S_n[1] * sys.H_bilinear[n][m][1,2] * S_m[2] + S_n[1] * sys.H_bilinear[n][m][1,3] * S_m[3]
            E_bilinear += S_n[2] * sys.H_bilinear[n][m][2,1] * S_m[1] + S_n[2] * sys.H_bilinear[n][m][2,2] * S_m[2] + S_n[2] * sys.H_bilinear[n][m][2,3] * S_m[3]
            E_bilinear += S_n[3] * sys.H_bilinear[n][m][3,1] * S_m[1] + S_n[3] * sys.H_bilinear[n][m][3,2] * S_m[2] + S_n[3] * sys.H_bilinear[n][m][3,3] * S_m[3] 
        end

        if length(sys.cubic_sites) == 0
            continue
        end
        #cubic contribution
        for trimer in eachindex(sys.cubic_sites[n])
            S_i = get_spin(sys.spins, sys.cubic_sites[n][trimer][1])
            S_j = get_spin(sys.spins, sys.cubic_sites[n][trimer][2])
            S_k = get_spin(sys.spins, sys.cubic_sites[n][trimer][3])
            
            K = sys.H_cubic[n][trimer]
            for a in 1:3, b in 1:3, c in 1:3
                E_cubic += K[a,b,c] * S_i[a] * S_j[b] * S_k[c]
            end
        end
          
        #zeeman contribution
        E_zeeman += - dot(sys.zeeman_field[n], S_n)
    end
    
    #total energy, not energy per site
    return E_bilinear/2.0 + E_cubic/3.0 + E_zeeman
end

function energy_difference_pyro(sys::SpinSystem, old_spin::NTuple{3,Float64}, n::Int64)::Float64
    h_loc = local_field_pyro(sys, n)
    E_new = dot(get_spin(sys.spins, n), h_loc)
    E_old = dot(old_spin, h_loc)

    return E_new - E_old 
end

#intializes a random spin configuration with shape 3 x 4N^3
function spins_initial_pyro(N::Int64, S::Float64)::Array{Float64,2}
    N_sites = 4*N^3
    spins = rand(3, N_sites)
    for j=1:N_sites
        spins[:,j] .*= S/norm(spins[:,j]) #normalizes each spin to length S
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
function metropolis!(sys::SpinSystem, accept_count::Array{Int64,1},T::Float64)
    N_sites = sys.N_sites
    
    for site in 1:N_sites #1 sweep has N_sites steps
        i = rand(1:N_sites)        
        old_spin = get_spin(sys.spins, i) #copy previous configuration 
        set_spin!(sys.spins, sphere_pick(sys.S), i)
        
        delta_E = energy_difference_pyro(sys, old_spin, i) 
        
        #accept if energy is lower (delta E < 0) or with probability given by Boltzmann weight
        no_accept = delta_E > 0 && (rand() > exp(-delta_E/T))
        accept_count[1] += 1 - no_accept
        
        #otherwise revert to previous configuration
        if no_accept 
            set_spin!(sys.spins, old_spin, i)
        end
    end 
end

function det_update!(sys::SpinSystem)
    for n in 1:sys.N_sites
        h_loc = local_field_pyro(sys, n)
        set_spin!(sys.spins, -sys.S .* h_loc ./ sqrt(h_loc[1]^2+h_loc[2]^2+h_loc[3]^2), n)
    end
end

#overrelaxation (microcanonical sweep) which reflects each spin about the local field
function overrelax_pyro!(sys::SpinSystem)
    for n in 1:sys.N_sites
        h_loc = local_field_pyro(sys, n)
        S_n = get_spin(sys.spins, n)
        S_new = 2.0 * dot(S_n, h_loc)/(h_loc[1]^2+h_loc[2]^2+h_loc[3]^2) .* h_loc .- S_n
        set_spin!(sys.spins, S_new, n)
    end
end

#simulated annealing with overrelaxation for N_therm sweeps, then deterministic updates for N_det sweeps
function sim_anneal!(mc::Simulation, schedule::Function, output_temp::Vector{Float64}=Float64[], print_progress::Bool=true)
    N_therm = mc.parameters.N_therm
    N_det = mc.parameters.N_det
    overrelax_rate = mc.parameters.overrelax_rate

    N = mc.spin_system.N
    N_sites = mc.spin_system.N_sites
    
    accept_count = [0]
    N_output_temp = length(output_temp)
    output_configurations = Array{Matrix{Float64}}(undef, N_output_temp)
    
    #metropolis + overrelaxation
    T = schedule(0)::Float64
    T_f = mc.T #set the T parameter to the target temp
    
    t0 = 0
    T_schedule = Float64[]
    while T > T_f
        T = schedule(t0)::Float64
        push!(T_schedule, T)
        t0 += 1
    end

    sort!(append!(T_schedule, output_temp), rev=true)    
    energies_therm = similar(T_schedule)
    
    output_count = 1
    output_temp_rev = sort(output_temp, rev=true)
    
    save_ind = similar(output_temp_rev, Int64)
    for tt in eachindex(output_temp_rev)
	    save_ind[tt] = argmin(abs.(T_schedule .- output_temp_rev[tt]))
    end
    println("Metropolis acceptance rate:")
    for t in eachindex(T_schedule)
        T = T_schedule[t]
        for sweep in 1:N_therm
            if sweep % overrelax_rate == 0
                metropolis!(mc.spin_system, accept_count, T)
            else
                overrelax_pyro!(mc.spin_system)
            end
        end

        if print_progress
            @printf("T=%.6f: %.3f%%\n", T, Float64(accept_count[1]/(N_sites*N_therm/overrelax_rate)*100))
        end
        accept_count = [0] 

        energies_therm[t] = E_pyro(mc.spin_system)

        #save spin configuration
        if t in save_ind           
            output_configurations[N_output_temp - output_count + 1] = copy(mc.spin_system.spins)
            output_count += 1
        end
    end
    
    for _ in 1:N_det        
        det_update!(mc.spin_system)
    end        
    
    #each simulated annealing run constitutes one measurement (at the end)
    E = E_pyro(mc.spin_system)
    avg_spin = spin_expec(mc.spin_system.spins, N)
    m = norm(magnetization_global(avg_spin, local_bases, mc.spin_system.h))

    push!(mc.observables.energy, E, E^2)
    push!(mc.observables.magnetization, m, m^2, m^4)
    push!(mc.observables.avg_spin, avg_spin, spin_expec(mc.spin_system.spins.^2, N))

    #thermalization energies and output configurations at requested temperatures
    return energies_therm, output_configurations
end

function parallel_temper!(mc::Simulation, rank::Int64, temp::Vector{Float64}, comm::MPI.Comm, comm_size::Int64)
    N_therm = mc.parameters.N_therm
    overrelax_rate = mc.parameters.overrelax_rate
    N_meas = mc.parameters.N_meas
    probe_rate = mc.parameters.probe_rate
    replica_exchange_rate = mc.parameters.replica_exchange_rate
    optimize_temperature_rate = mc.parameters.optimize_temperature_rate

    N = mc.spin_system.N

    N_ranks = length(temp)
    T = mc.T
    N_sweeps = N_therm + N_meas
    energies = zeros(N_sweeps)
    
    accept_count_metropolis = [0] #counts successful metropolis steps (not currently outputted)
    accept_count_swap = 0 #counts number of successful swaps

    n_up = 0 #number of replicas going "up" through the temperature T
    n_down = 0 #number of replicas going "down" through the temperature T

    if rank == 0
        mc.replica_label = "up"
    elseif rank == N_ranks-1
        mc.replica_label = "down"
    end
    
    new_spins = copy(mc.spin_system.spins) #buffer for replica exchange

    for sweep in 1:N_sweeps
        n_up += (mc.replica_label == "up")
        n_down += (mc.replica_label == "down")
        
        #do overrelaxation and metropolis with relative frequency overrelax_rate
        if sweep % overrelax_rate == 0
            metropolis!(mc.spin_system, accept_count_metropolis, T)
        else
            overrelax_pyro!(mc.spin_system)
        end
        
        E = E_pyro(mc.spin_system)
        energies[sweep] = E

        if sweep > N_therm && sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            avg_spin = spin_expec(mc.spin_system.spins, N)
            m = norm(magnetization_global(avg_spin, local_bases, mc.spin_system.h))
            #do we have to use norm(m)? 

            push!(mc.observables.energy, E, E^2)
            push!(mc.observables.magnetization, m, m^2, m^4)
            push!(mc.observables.avg_spin, avg_spin, spin_expec(mc.spin_system.spins.^2, N))

            for i in 1:3
                for mu in 1:4
                    S_i_mu = avg_spin[i,mu]
                    push!(mc.observables.energy_spin_covariance[i,mu], E*S_i_mu, E, S_i_mu)
                end
            end
        end 

        if sweep % replica_exchange_rate == 0
            #println(string(rank)*": it's time to swap!")
            #alternate between swap_type 0 and swap_type 1
            swap_type = div(sweep, replica_exchange_rate)%2
            
            new_spins, partner_replica_number, partner_label, accepted = replica_exchange!(mc.spin_system.spins, rank, E, mc.replica_number, mc.replica_label, temp, swap_type, comm, comm_size)
            
            if accepted
                mc.spin_system.spins .= copy(new_spins)
                mc.replica_number = partner_replica_number
                mc.replica_label = partner_label

                #change the "direction" of the replica if it reaches the highest or lowest rank
                if rank == 0
                    mc.replica_label = "up"
                elseif rank == N_ranks - 1
                    mc.replica_label = "down"
                end
                accept_count_swap += 1
            end

            #don't update temperatures while taking measurements
            #=
            if sweep < N_therm && sweep % optimize_temperature_rate == 0
                denom = n_up + n_down
                flow = denom == 0 ? 0.0 : n_up / denom
                temp .= feedback_optimize_temperature(temp, flow, rank, comm)
                T = temp[rank+1]
                mc.T = temp[rank+1]
            end
            =#
            if sweep == N_therm
                n_up = 0
                n_down = 0
            end
        end 
    end

    denom = n_up + n_down
    flow = denom == 0 ? 0.0 : n_up / denom
    fname = replace(pwd(), "\\" => "/") * "/pt_out/E_final_" * string(rank) * ".txt"
    output_data = "rank $(rank) at T=$(T)\nE=" * string(energies[end]) * "\n" * mc.replica_label
    write(fname, output_data)

    return energies, accept_count_metropolis, accept_count_swap, flow
end

function swap_adjacent!(arr::AbstractArray, rank::Int64, partner_rank::Int64, swap_type::Int64, comm::MPI.Comm, comm_size::Int64)
    # No-op if partner is out of range
    if partner_rank < 0 || partner_rank >= comm_size
        return arr
    end

    buffer = similar(arr)
    # Complementary ordering to avoid deadlock
    if rank % 2 == swap_type
        MPI.send(arr, comm, dest=partner_rank)
        buffer .= MPI.recv(comm, source=partner_rank)
    else
        buffer .= MPI.recv(comm, source=partner_rank)
        MPI.send(arr, comm, dest=partner_rank)
    end

    # copy into original array (preserve shape)
    arr .= buffer
    return arr
end

#do this each iteration of the loop (i.e. when it's time to try swapping)
function replica_exchange!(spins::Array{Float64,2}, rank::Int64, E_rank::Float64, replica_number::Int64, replica_label::String, temp::Vector{Float64}, swap_type::Int64, comm::MPI.Comm, comm_size::Int64)
    # determine partner (pairs: swap_type=0 -> (0,1)(2,3)..., swap_type=1 -> (1,2)(3,4)...)
    partner_rank = (rank % 2 == swap_type) ? rank + 1 : rank - 1

    # If partner out of range, no swap
    if partner_rank < 0 || partner_rank >= comm_size
        return spins, replica_number, replica_label, false
    end

    # exchange energies with complementary ordering (to avoid deadlock)
    if rank % 2 == swap_type
        MPI.send(E_rank, comm, dest=partner_rank)
        E_partner = MPI.recv(comm, source=partner_rank)
        delta_E = E_partner - E_rank
        delta_beta = 1.0/temp[partner_rank+1] - 1.0/temp[rank+1]
    else
        E_partner = MPI.recv(comm, source=partner_rank)
        MPI.send(E_rank, comm, dest=partner_rank)
        # other side does not need delta_E/delta_beta
        delta_E = nothing
        delta_beta = nothing
    end

    # decide acceptance (only the side that computed delta evaluates)
    accept = false
    if rank % 2 == swap_type
        accept_prob = exp(delta_beta * delta_E)
        
        accept = rand() < accept_prob
        MPI.send(accept, comm, dest=partner_rank)
    else
        accept = MPI.recv(comm, source=partner_rank)
    end

    # perform swap if accepted (both sides must do this)
    if accept
        spins = swap_adjacent!(spins, rank, partner_rank, swap_type, comm, comm_size)

        # swap replica_number and label through the same helper
        partner_replica_number = swap_adjacent!([replica_number], rank, partner_rank, swap_type, comm, comm_size)[1]
        partner_replica_label = swap_adjacent!([replica_label], rank, partner_rank, swap_type, comm, comm_size)[1]

        return spins, partner_replica_number, partner_replica_label, true
    end

    return spins, replica_number, replica_label, false
end

function feedback_optimize_temperature(temp::Vector{Float64}, flow::Float64, rank::Int64, comm::MPI.Comm)
    gather_flow = MPI.Gather(flow, comm, root=0)
    
    if rank==0
        function filter_flow(xs, ys, ys_opt, tol=0.25)
            x_interp = [xs[1]]
            y_interp = [ys[1]] #should automatically be 1

            y_prev = ys[1]
            for j in eachindex(ys)
                val = ys[j]
                if val < y_prev && abs(val - ys_opt[j]) < tol && val > 1e-10
                    push!(x_interp, xs[j])
                    push!(y_interp, val)
                end
                y_prev = val
            end

            push!(x_interp, xs[end])
            push!(y_interp, ys[end])

            return x_interp, y_interp
        end

        function bisection(f, a, b, tol=1e-6, max_iter=30)
            # Ensure f(a) and f(b) have opposite signs
            if sign(f(a)) == sign(f(b))
                error("Function must have opposite signs at interval endpoints.")
            end

            for i in 1:max_iter
                c = (a + b) / 2 # Calculate the midpoint
                
                # Check for convergence
                if abs(f(c)) < tol || (b - a) / 2 < tol
                    return c
                end

                # Update the interval
                if sign(f(c)) == sign(f(a))
                    a = c
                else
                    b = c
                end
            end
            println("Bisection method did not converge within $max_iter iterations.")
            return (a + b) / 2 # Return the last midpoint as an approximation
        end

        N_ranks = length(temp)
        flow_opt = 1 .- Vector(range(0, N_ranks-1, N_ranks)) ./ (N_ranks-1)
        T_min = temp[1]
        T_max = temp[end]
        x_filter, y_filter = filter_flow(temp, gather_flow, flow_opt)

        interp_monotone = interpolate(x_filter, y_filter, SteffenMonotonicInterpolation())
        #println(x_filter, y_filter)

        #interpolated flow vector and its derivative
        g(x) = interp_monotone(x) 
        dg(x) = -1.0 * ForwardDiff.derivative(g,x)

        C(x) = solve(IntegralProblem((x, p) -> sqrt(dg(x)), (T_min, x)), QuadGKJL()).u
        new_temp = copy(temp) 
        
        try
            C0 = C(T_max - 1e-6) #normalization constant for eta
            #solves int_{T_min}^x C(x)/C0 = r/(N_ranks-1) for rank r
            new_temp[2:end-1] = [bisection(x -> C(x)/C0 - (1-f_opt), T_min, T_max - 1e-6) for f_opt in flow_opt[2:end-1]]
        catch DomainError
            println("Failed to adjust temperatures.")
        end
    else
        new_temp = nothing
    end
    
    new_temp = MPI.bcast(new_temp, 0, comm)

    return new_temp
end