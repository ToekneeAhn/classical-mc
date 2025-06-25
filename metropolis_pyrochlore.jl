using LinearAlgebra, StaticArrays, BinningAnalysis

include("observables.jl")

#local z axis on sublattice m in in column m+1
z_local = 1/sqrt(3)*[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]'
#local dipole moments
local_interactions = 1.0 .* [0 0 1; 0 0 1; 0 0 1; 0 0 1]'
#bond-dependent gamma factor
omega = exp(2*pi*im/3)
gamma_ij = [0 1 omega omega^2; 1 0 omega^2 omega; omega omega^2 0 1; omega^2 omega 1 0]

#sublattice-indexed coordinates
struct sic 
    r::Vector{Int64} #position of unit cell (i.e. sublattice 0)
    mu::Int64 #sublattice index 0,1,2,3
    N::Int64 #number of unit cells in each direction
end

#defines addition of sublattice-indexed coordinates (kinda) with multiple dispatch
#assume periodic boundary conditions
import Base: +, *
pos_mod(x::Vector{Int64}, m) = ((x.%m).+m).%m
+(r_mu::sic, v::Vector{Int64}) = sic(pos_mod(r_mu.r + v, r_mu.N), r_mu.mu, r_mu.N)

function flat_index_3D(r_mu::sic) #only for the 3d case
    nx, ny, nz = r_mu.r
    mu = r_mu.mu
    N = r_mu.N
    return nx + N^2*ny + N*nz + mu*N^3 + 1
end

function sic_index_3D(n::Int64, N::Int64)
    n -= 1
    mu = div(n, N^3)
    nx = n%N
    nz = div(n - nx, N) % N
    ny = div(n - nx - div(n - nx, N), N) % N
    
    return sic([nx, ny, nz], mu, N)
end

#indices of neighbours given sic r_mu
function neighbours_pyro(r_mu::sic)
    ind = flat_index_3D(r_mu)-1
    mu = r_mu.mu
    N = r_mu.N
    
    neighbours_flat = Array{Int64}(undef, 7)
    neighbours_flat[1:3] = pos_mod([N^3, 2*N^3, 3*N^3].+ind, 4*N^3).+1 #flat indices of intra-tetrahedron neighbours
    
    ee = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'
    
    for l in 0:3
        r_p = (r_mu + ee[:, mu+1] + (-1)*ee[:,l+1]).r #inter-tetrahedron neighbours
        r_neighbour = sic(r_p, l, N)
        neighbours_flat[l+1+3] = flat_index_3D(r_neighbour)
    end 
    return deleteat!(neighbours_flat, mu+4)
end

#indices of neighbours given flat index (multiple dispatch)
function neighbours_pyro(n::Int64, N::Int64) 
    ind = n-1
    mu = div(n-1, N^3)
    r_mu = sic_index_3D(n, N)
    
    neighbours_flat = Array{Int64}(undef, 7)
    neighbours_flat[1:3] = pos_mod([N^3, 2*N^3, 3*N^3].+ind, 4*N^3).+1 #flat indices of intra-tetrahedron neighbours
    
    ee = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'
    
    for l in 0:3
        r_p = (r_mu + ee[:, mu+1] + (-1)*ee[:,l+1]).r #inter-tetrahedron neighbours
        r_neighbour = sic(r_p, l, N)
        neighbours_flat[l+1+3] = flat_index_3D(r_neighbour)
    end 
    return Tuple(deleteat!(neighbours_flat, mu+4))
end

function neighbours_all(N_sites)
    coord_num = 6
    
    #neighbours = zeros(Int64, coord_num, N_sites)
    neighbours = NTuple{coord_num, Int64}[]
    
    for n = 1:N_sites
        #neighbours[:, n] = neighbours_pyro(n, N)
        push!(neighbours, neighbours_pyro(n, N))
    end
    
    #return SMatrix{coord_num, N_sites, Int64}(neighbours)
    return neighbours
end

function set_spin!(spins::Array{Float64,2}, S_new::Vector{Float64}, site::Int64)
    @inbounds spins[:, site] .= S_new
end

function set_spin!(spins::Array{Float64,2}, S_new::NTuple{3,Float64}, site::Int64)
    @inbounds spins[1, site] = S_new[1]
    @inbounds spins[2, site] = S_new[2]
    @inbounds spins[3, site] = S_new[3]
end

function get_spin(spins::Array{Float64,2}, site::Int64)::NTuple{3, Float64}
    @inbounds return (spins[1, site], spins[2, site], spins[3, site])
end

#4 x 4 x 3 x 3 array of interaction matrices (3x3) connecting sublattices
function H_matrix_all(Js::NTuple{4,Float64})
    J_zz, J_pm, J_pmpm, J_zpm = Js

    H_bond = zeros(4,4,3,3)
    T = [1 im 0; 1 -im 0; 0 0 1] #rotates to (S^+, S^-, S^z) basis
    
    for sub_i in 0:3
        for sub_j in 0:3
            gamma = gamma_ij[sub_i+1, sub_j+1]
            zeta = -conj(gamma)
            
            if sub_i != sub_j
                H_bond[sub_i+1, sub_j+1, :, :] .= conj(T)' * [J_pmpm*gamma -J_pm J_zpm*zeta; -J_pm J_pmpm*conj(gamma) J_zpm*conj(zeta); J_zpm*zeta J_zpm*conj(zeta) J_zz] * T
            end
        end
    end

    return SArray{Tuple{4,4,3,3},Float64}(reinterpret(Float64, real(H_bond)))
end

function zeeman_field(h, local_bases, local_interactions)
    #=
    zeeman_eff = zeros(3, 4)
    for mu in 1:4
        zeeman_eff[:,mu] = (h' * local_bases[:, mu])*local_interactions[:, mu]
    end
    
    return SMatrix{3,4,Float64}(zeeman_eff)
    =#
    zeeman_eff = NTuple{3,Float64}[]
    for mu in 1:4
        push!(zeeman_eff, Tuple((h' * local_bases[:, mu])*local_interactions[:, mu]))
    end
    return zeeman_eff
end

#h_loc at site i is dH/dS_i
function local_field_pyro(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours_n::NTuple{6,Int64}, zeeman::Vector{NTuple{3,Float64}}, n::Int64)::NTuple{3,Float64}
    sublattice = div(n-1, N^3)
    
    h_loc = zeros(3)

    for m in neighbours_n
        h_loc .+= H_bond[sublattice+1, div(m-1, N^3)+1,:,:] * spins[:,m]
    end

    h_z = zeeman[sublattice+1]
    return (h_loc[1]-h_z[1], h_loc[2]-h_z[2], h_loc[3]-h_z[3])
    
    #=
    Hx=0.0
    Hy=0.0
    Hz=0.0
    
    for m in neighbours_n
        sx, sy, sz = get_spin(spins, m)
        H_ij = Matrix{Float64}
        @inbounds H_ij = H_bond[sublattice+1, div(m-1, N^3)+1, :, :]
        @inbounds Hx += H_ij[1,1] * sx + H_ij[1,2] * sy + H_ij[1,3] * sz
        @inbounds Hy += H_ij[2,1] * sx + H_ij[2,2] * sy + H_ij[2,3] * sz
        @inbounds Hz += H_ij[3,1] * sx + H_ij[3,2] * sy + H_ij[3,3] * sz
    end

    h_z = zeeman[sublattice+1]
    return (Hx-h_z[1], Hy-h_z[2], Hz-h_z[3])
    =#
end

#energy of spin configuration with periodic boundary conditions
function E_pyro(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)::Float64
    N_sites = 4*N^3
    E = 0.0
    
    for n in 1:N_sites
        #quadratic interaction, divide by 2 because each bond counted twice
        sublattice = div(n-1, N^3)
        
        for m in neighbours[n]
            @inbounds E += 0.5 * (spins[:,n]' * H_bond[sublattice+1, div(m-1, N^3)+1,:,:] * spins[:,m])
        end

        #zeeman contribution
        @inbounds E += - dot(zeeman[sublattice+1], spins[:,n])
    end
    
    return E/N_sites
end

function energy_difference_pyro(spins::Array{Float64,2}, old_spin::Vector{Float64}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours_n::NTuple{6,Int64}, zeeman::Vector{NTuple{3,Float64}}, n::Int64)::Float64
    h_loc = local_field_pyro(spins, H_bond, neighbours_n, zeeman, n)
    E_new = dot(get_spin(spins, n), h_loc)
    E_old = dot(old_spin, h_loc)

    return E_new - E_old 
end

#temperature as a function of sweep number, decreases exponentially
#=
function anneal_schedule(sweep, N_sweeps, T_initial, T_final)  
    a = 1/N_sweeps*log(T_final/T_initial)
    return T_initial*exp(a*sweep)
end
=#
function anneal_schedule(t::Int64)::Float64
    return 0.9^t
end

#intializes a random spin configuration with shape 3 x 4N^3
function spins_initial_pyro(N::Int64, S::Float64)::Array{Float64,2}
    spins = rand(3, 4*N^3)
    for j=1:4*N^3
        spins[:,j] .*= S/norm(spins[:,j]) #normalizes each spin
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
    
    for site in 1:N_sites #do this once for every site
        i = rand(1:N_sites)        
        old_spin = copy(spins[:,i]) #copy previous configuration 
        set_spin!(spins, sphere_pick(S), i)
        
        delta_E = energy_difference_pyro(spins, old_spin, H_bond, neighbours[i], zeeman, i) 
        
        #accept if energy is lower (delta E < 0) or with probability given by Boltzmann weight
        no_accept = delta_E > 0 && (rand() > exp(-delta_E/T))
        accept_count .+= 1-no_accept
        
        if no_accept 
            set_spin!(spins, old_spin, i)
        end
    end 
end

function det_update!(spins::Array{Float64,2}, S::Float64, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)
    N_sites = 4*N^3
    
    for n in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, neighbours[n], zeeman, n)
        set_spin!(spins, -S .* h_loc ./ norm(h_loc), n)
    end
end

#overrelaxation (microcanonical sweep) which reflects each spin about the local field
function overrelax_pyro!(spins::Array{Float64,2}, H_bond::SArray{Tuple{4,4,3,3}, Float64}, neighbours::Vector{NTuple{6,Int64}}, zeeman::Vector{NTuple{3,Float64}}, N::Int64)
    N_sites = 4*N^3
    
    for n in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, neighbours[n], zeeman, n)
        S_i = get_spin(spins, n)
        set_spin!(spins, .- S_i .+ 2.0 * dot(S_i, h_loc)/(h_loc[1]^2+h_loc[2]^2+h_loc[3]^2) .* h_loc, n)
    end
end

#number of thermalization sweeps, deterministic sweeps, overrelax:metropolis rate
#initial temperature, target temperature, number of unit cells in each direction
#Js = (J_zz, J_pm, J_pmpm, J_zpm), external magnetic field vector, initial spin configuration
function sim_anneal(N_therm::Int64, N_det::Int64, probe_rate::Int64, overrelax_rate::Int64, T_i::Float64, T_f::Float64, Js::NTuple{4,Float64}, h::Vector{Float64}, N::Int64, S::Float64, spins=[])
    #can set initial spin configuration, generate randomly if not
    N_sites = 4*N^3
    if length(spins) != 3*N_sites
        spins = spins_initial_pyro(N, S)
    end

    #precomputes interaction matrices for each bond, neighbours for each site, and zeeman interaction for each sublattice
    H_bond = H_matrix_all(Js)
    neighbours = neighbours_all(N_sites)
    zeeman = zeeman_field(h, z_local, local_interactions)
    
    #using 0.9^t annealing schedule
    t_steps = convert(Int, ceil(log(T_f/T_i)/log(0.9)))

    N_probe = div(N_det, probe_rate)
    accept_count = [0]
    measurements = Any[]
    energies_therm = zeros(t_steps)
    energies = zeros(N_probe)
    avg_spin = zeros(N_probe, 3, 4)
    
    #metropolis + overrelaxation
    for t in 1:t_steps
        T = 0.9^(t-1)
        
        for sweep in 1:N_therm
            if sweep % overrelax_rate == 0
                metropolis!(spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
            else
                overrelax_pyro!(spins, H_bond, neighbours, zeeman, N)
            end
        end

        println("acceptance rate at T=", T, ": ", accept_count[1]/(4*N^3*N_therm/overrelax_rate)*100, "%")
        accept_count = [0] 

        energies_therm[t] = E_pyro(spins, H_bond, neighbours, zeeman, N)
    end
    
    for sweep in 1:N_det        
        det_update!(spins, S, H_bond, neighbours, zeeman, N)

        if sweep % probe_rate == 0
        #take measurements after thermalization every probe_rate sweeps
        #for magnetostriction, we just want the ensemble averaged spin for each sublattice
        #so N_det x 3 x 4 array of results for each rank
            meas_ind = div(sweep, probe_rate)
            avg_spin[meas_ind, :, :] = spin_expec(spins)
            energies[meas_ind] = E_pyro(spins, H_bond, neighbours, zeeman, N)
        end
    end        

    #takes the sample average 
    push!(measurements, 1/N_probe*sum(energies))
    push!(measurements, 1/N_probe*dropdims(sum(avg_spin, dims=1), dims=1))

    return spins, energies_therm, measurements #measurements
end

#parallel tempering
function parallel_temper(rank::Int64, replica_exchange_rate::Int64, N_therm::Int64, N_det::Int64, probe_rate::Int64, overrelax_rate::Int64, temp::Vector{Float64}, Js::NTuple{4,Float64}, h::Vector{Float64}, N::Int64, S::Float64, spins=[])
    #can set initial spin configuration, generate randomly if not
    N_sites = 4*N^3
    if length(spins) != 3*N_sites
        spins = spins_initial_pyro(N, S)
    end

    #precomputes interaction matrices for each bond and neighbours for each site
    H_bond = H_matrix_all(Js)
    neighbours = neighbours_all(N_sites)
    zeeman = zeeman_field(h, z_local, local_interactions)
    
    T = temp[rank+1]
    N_sweeps = N_therm + N_det
    energies = zeros(N_sweeps)
    measurements = zeros(div(N_det,probe_rate), 3, 4)
    accept_count = [0] #counts successful metropolis steps (not currently outputted)
    accept_count_swap = [0] #counts number of successful swaps

    for sweep in 1:N_sweeps
        if sweep > N_therm && sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            #for magnetostriction, we just want the ensemble averaged spin for each sublattice
            #so N_det x 4 x 3 array of results for each rank
            meas_ind = div(sweep - N_therm, probe_rate)
            measurements[meas_ind, :,: ] = spin_expec(spins)
        end
        
        #do deterministic updates on lowest temperature rank (after thermalization)?
        
        #do overrelaxation and metropolis with relative frequency overrelax_rate
        if sweep % overrelax_rate == 0
            metropolis!(spins, accept_count, S, H_bond, neighbours, zeeman, N, T)
        else
            overrelax_pyro!(spins, H_bond, neighbours, zeeman, N)
        end
        
        energies[sweep] = E_pyro(spins, H_bond, neighbours, zeeman, N)

        if sweep % replica_exchange_rate == 0
            #println(string(rank)*": it's time to swap!")

            #alternate between swap_type 0 and swap_type 1
            swap_type = div(sweep, replica_exchange_rate)%2
            spins, accept_swap = replica_exchange!(rank, spins, energies[sweep], temp, swap_type)
            accept_count_swap += accept_swap
        end 
    end

    #=
    fname = replace(pwd(), "\\"=>"/")*"/pt_out/E_final_"*string(rank)*".txt"
    output_data = string(rank)*" "*string(T)*" "*string(energies[end])*"\n"
    write(fname, output_data)
    =#

    #computes standard error with log binning of depth L
    #want at least 30 bins, done automatically by BinningAnalysis
    err = zeros(3,4)
    for comp = 1:3, subl = 1:4
        dataset = measurements[:, comp, subl]
        err[comp, subl] = std_error(dataset)
    end
    
    #takes the sample average 
    measurements = 1/N^3*sum(measurements, dims=1)[1,:,:]/(N_det/probe_rate)
    
    return spins, energies, measurements, err, accept_count_swap
end

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
            recv_data = MPI.recv(comm, source=partner_rank)
            #print("rank ", r, " got ", recv_data)
        end
        #prevents edge case of (-1,0) swap if swap_type=1
        if rank%2 == 1-swap_type && partner_rank >= swap_type 
            recv_data = MPI.recv(comm, source=partner_rank)
            MPI.send(spins, comm, dest=partner_rank)
            #print("rank ", r, " got ", recv_data)
        end
    end

    return recv_data, accept_arr
end

