using LinearAlgebra, BinningAnalysis

include("observables.jl")

#local z axis on sublattice m in in column m+1
z_local = 1/sqrt(3)*[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]' 
#bond-dependent gamma factor
omega = exp(2*pi*im/3)
gamma_ij = [0 1 omega omega^2; 1 0 omega^2 omega; omega omega^2 0 1; omega^2 omega 1 0]

#sublattice-indexed coordinates
struct sic 
    r::Vector{Int64} #position of unit cell (i.e. sublattice 0)
    mu::Int #sublattice index 0,1,2,3
    N::Int #number of unit cells in each direction
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

function sic_index_3D(n, N)
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

#indices of  neighbours given flat index (multiple dispatch)
#this is the bottleneck?
function neighbours_pyro(n::Int, N::Int) 
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
    return deleteat!(neighbours_flat, mu+4)
end

#4 x 4 x 3 x 3 array of interaction matrices (3x3) connecting sublattices
function H_matrix_all(Js)
    J_zz, J_pm, J_pmpm, J_zpm = Js

    H_bond = zeros(4,4,3,3)
    T = [1 im 0; 1 -im 0; 0 0 1] #rotates to (S^+, S^-, S^z) basis
    
    for sub_i in 0:3
        for sub_j in 0:3
            gamma = gamma_ij[sub_i+1, sub_j+1]
            zeta = -conj(gamma)
            
            if sub_i != sub_j
                H_bond[sub_i+1, sub_j+1, :, :] = conj(T)' * [J_pmpm*gamma -J_pm J_zpm*zeta; -J_pm J_pmpm*conj(gamma) J_zpm*conj(zeta); J_zpm*zeta J_zpm*conj(zeta) J_zz] * T
            end
        end
    end

    return reinterpret(Float64, real(H_bond))
end

#h_loc at site i is dH/dS_i
function local_field_pyro(spins, H_bond, h, n, N)
    sublattice = div(n-1, N^3)
    neighbours = neighbours_pyro(n, N)
    
    h_loc = zeros(3)
    for m in neighbours
        h_loc += H_bond[sublattice+1, div(m-1, N^3)+1,:,:] * spins[:,m]
    end

    return h_loc - (h' * z_local[:, sublattice+1])*[0,0,1]
end

#energy of spin configuration with periodic boundary conditions
function E_pyro(spins, H_bond, h, N)
    N_sites = 4*N^3
    E = 0
    
    for n in 1:N_sites
        #quadratic interaction, divide by 2 because each bond counted twice
        sublattice = div(n-1, N^3)
        neighbours = neighbours_pyro(n, N)
        
        for m in neighbours
            E += 0.5 * spins[:,n]' * H_bond[sublattice+1, div(m-1, N^3)+1,:,:] * spins[:,m]
        end

        #zeeman contribution
        E += -(h' * z_local[:, sublattice+1])*(spins[:,n]' * [0,0,1]) 
    end
    
    return E/N_sites
end

function energy_difference_pyro(spins, old_spin, H_bond, h, n, N)
    h_loc = local_field_pyro(spins, H_bond, h, n, N)
    E_new = spins[:,n]' * h_loc
    E_old = old_spin' * h_loc

    return E_new - E_old 
end

#temperature as a function of sweep number, decreases exponentially
function anneal_schedule(sweep, N_sweeps, T_initial, T_final)  
    a = 1/N_sweeps*log(T_final/T_initial)
    return T_initial*exp(a*sweep)
end

#intializes a random spin configuration with shape 3 x 4N^3
function spins_initial_pyro(N, S)
    spins = rand(3, 4*N^3)
    for j=1:4*N^3
        spins[:,j] *= S/norm(spins[:,j]) #normalizes each spin
        #spins[j,:] = [0,0,1]
    end
    return spins
end

#picks a point on the unit sphere uniformly and returns Cartesian coordinates (Sx,Sy,Sz)
#then scales magnitude by S
function sphere_pick(S) 
    phi = 2*pi*rand()
    z = 2*rand() - 1
    theta = acos(z)
    return S*[sin(theta)*cos(phi), sin(theta)*sin(phi), z]
end

#metropolis algorithm with deterministic updates (aligning spins to their local field)
function metropolis!(spins, S, H_bond, h, N, T)
    N_sites = 4*N^3
    
    for site in 1:N_sites #do this once for every site
        i = rand(1:N_sites)        
        old_spin = copy(spins[:,i]) #copy previous configuration 
        spins[:,i] = sphere_pick(S)

        delta_E = energy_difference_pyro(spins, old_spin, H_bond, h, i, N) 
    
        if rand() > exp(-delta_E/T) #accept if energy is lower (delta E < 0) or with probability given by Boltzmann weight
            spins[:,i] = old_spin # if not accepted, revert to old spin
        end
    end 
end

function det_update!(spins, S, H_bond, h, N)
    N_sites = 4*N^3
    
    #can modify spins on the spot or use a copy for h_local 
    #don't know which one is right

    #spins_copy = copy(spins)
    for i in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, h, i, N)
        spins[:,i] = -S*h_loc/norm(h_loc)
    end
end

#overrelaxation (microcanonical sweep) which reflects each spin about the local field
function overrelax_pyro!(spins, H_bond, h, N)
    N_sites = 4*N^3
    
    for site in 1:N_sites
        h_loc = local_field_pyro(spins, H_bond, h, site, N)
        S_i = spins[:, site]
        spins[:, site] = -S_i + 2* (S_i' * h_loc)/norm(h_loc)^2 * h_loc
    end
end

#number of thermalization sweeps, deterministic sweeps, overrelax:metropolis rate
#initial temperature, target temperature, number of unit cells in each direction
#Js = (J_zz, J_pm, J_pmpm, J_zpm), external magnetic field vector, initial spin configuration
function sim_anneal(N_therm, N_det, probe_rate, overrelax_rate, T_i, T_f, Js, h, N, S, spins=[])
    #can set initial spin configuration, generate randomly if not
    if length(spins) != 4*N^3
        spins = spins_initial_pyro(N, S)
    end

    H_bond = H_matrix_all(Js)

    N_sweeps = N_therm + N_det
    measurements = Any[]
    energies = zeros(N_sweeps)
    avg_spin = zeros(div(N_det,probe_rate), 3, 4)
    
    T = T_i #temperature

    for sweep in 1:N_sweeps
        T = max(anneal_schedule(sweep, N_therm, T_i, T_f), T_f) #stop changing T after N_therm sweeps
        if sweep > N_therm #switch to deterministic sweeps after N_therm sweeps
            det_update!(spins, S, H_bond, h, N)

            if sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            #for magnetostriction, we just want the ensemble averaged spin for each sublattice
            #so N_det x 3 x 4 array of results for each rank
                meas_ind = div(sweep - N_therm, probe_rate)
                avg_spin[meas_ind, :, :] = spin_expec(spins)
            end
        else #otherwise, do overrelaxation and metropolis with relative frequency overrelax_rate
            if sweep % overrelax_rate == 0
                metropolis!(spins, S, H_bond, h, N, T)
            else
                overrelax_pyro!(spins, H_bond, h, N)
            end
        end
            
        energies[sweep] = E_pyro(spins, H_bond, h, N)
    end        

    #takes the sample average 
    N_meas = div(N_det, probe_rate)
    push!(measurements, 1/N_meas*sum(energies[N_therm+probe_rate:probe_rate:end]))
    push!(measurements, 1/N_meas*dropdims(sum(avg_spin, dims=1), dims=1))

    return spins, energies, measurements #measurements
    #return measurements
end

#parallel tempering
function parallel_temper(rank, replica_exchange_rate, N_therm, N_det, probe_rate, overrelax_rate, temp, Js, h, N, S, spins=[])
    #can set initial spin configuration, generate randomly if not
    if length(spins) != 4*N^3
        spins = spins_initial_pyro(N, S)
    end

    H_bond = H_matrix_all(Js)
    
    T = temp[rank+1]
    N_sweeps = N_therm + N_det
    energies = zeros(N_sweeps)
    measurements = zeros(div(N_det,probe_rate), 3, 4)
    #accept_count = 0 #counts number of successful swaps
    accepts = zeros(N_sweeps)

    det = false #whether to do deterministic updates
    
    for sweep in 1:N_sweeps
        if sweep > N_therm && sweep % probe_rate == 0
            #take measurements after thermalization every probe_rate sweeps
            #for magnetostriction, we just want the ensemble averaged spin for each sublattice
            #so N_det x 4 x 3 array of results for each rank
            meas_ind = div(sweep - N_therm, probe_rate)
            measurements[meas_ind, :, 1] = sum(spins[:,1:N^3], dims=2)[:,1]
            measurements[meas_ind, :, 2] = sum(spins[:,(N^3+1):(2*N^3)], dims=2)[:,1]
            measurements[meas_ind, :, 3] = sum(spins[:,(2*N^3+1):(3*N^3)], dims=2)[:,1]
            measurements[meas_ind, :, 4] = sum(spins[:,(3*N^3+1):(4*N^3)], dims=2)[:,1]
        end
        
        #only do deterministic updates on lowest temperature rank (after thermalization)
        if rank == 0 && sweep > N_therm
            det = true 
            det_update!(spins, S, H_bond, h, N)
        else #do overrelaxation and metropolis with relative frequency overrelax_rate
            if sweep % overrelax_rate == 0
                metropolis!(spins, S, H_bond, h, N, T)
            else
                overrelax_pyro!(spins, H_bond, h, N)
            end
        end
        energies[sweep] = E_pyro(spins, H_bond, h, N)

        if sweep % replica_exchange_rate == 0
            #println(string(rank)*": it's time to swap!")

            #alternate between swap_type 0 and swap_type 1
            swap_type = div(sweep, replica_exchange_rate)%2
            #swap_type = 0
            spins, accept = replica_exchange!(rank, spins, energies[sweep], temp, swap_type)
            accepts[sweep] = accept
        end 
    end

    fname = replace(pwd(), "\\"=>"/")*"/pt_out/E_final_"*string(rank)*".txt"
    output_data = string(rank)*" "*string(T)*" "*string(energies[end])*"\n"
    write(fname, output_data)

    #fname = "swaps_"*string(rank)*".txt"
    #write(fname, string(accepts))

    #computes standard error with log binning of depth L
    #want at least 30 bins, done automatically by BinningAnalysis
    err = zeros(3,4)
    for comp = 1:3, subl = 1:4
        dataset = measurements[:, comp, subl]
        err[comp, subl] = std_error(dataset)
    end
    
    #takes the sample average 
    measurements = 1/N^3*sum(measurements, dims=1)[1,:,:]/(N_det/probe_rate)
    
    return spins, energies, measurements, err, accepts
end

#do this each iteration of the loop (i.e. when it's time to try swapping)
function replica_exchange!(rank, spins, E_rank, temp, swap_type)
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

    return recv_data, convert(Int, accept_arr[1])
end