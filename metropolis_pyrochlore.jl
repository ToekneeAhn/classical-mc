using LinearAlgebra, StaticArrays, BinningAnalysis, Random, Interpolations, ForwardDiff, Integrals

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
    disorder_strength::Float64 #Gamma parameter in Lorentzian distribution
    H_bond::SArray{Tuple{4,4,3,3},Float64}
    neighbours::Vector{NTuple{6,Int64}}
    zeeman_field::Vector{NTuple{3,Float64}}
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

function zeeman_field_random(h, z_local, local_interactions, delta_12, G, N_sites, seed=123)::Vector{NTuple{3,Float64}}
    Random.seed!(seed)
    
    zeeman_eff = NTuple{3,Float64}[]
    for mu in 1:4
        h_z = (h' * z_local[:, mu]) .* local_interactions[:, mu]
        h_mu = local_bases[mu]' * h
        h_xy_quadratic = delta_12[1] .* (h_mu[1] * h_mu[3], h_mu[2] * h_mu[3], 0.0) .+ delta_12[2] .* (h_mu[2]^2 - h_mu[1]^2, 2.0 *h_mu[1] * h_mu[2], 0.0)

        for n in 1:N_sites/4
            random_strength = G*tan(pi*(rand()-0.5)) #draws from a lorentzian distribution with pdf p(h) = G/pi * 1/(G^2+h^2)
            random_phase = 2*pi*rand()

            h_xy_random = random_strength .* (cos(random_phase), sin(random_phase), 0.0)
            
            push!(zeeman_eff, Tuple(h_z .+ h_xy_quadratic .+ h_xy_random))
        end
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

    h_z = zeeman[n]
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
        @inbounds E += - dot(zeeman[n], S_n)
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
    
    H_bond = mc.spin_system.H_bond
    neighbours = mc.spin_system.neighbours
    zeeman = mc.spin_system.zeeman_field
    
    accept_count = [0]
    
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
    push!(mc.observables.magnetization, m, m^2, m^4)
    push!(mc.observables.avg_spin, avg_spin, avg_spin.^2)

    #thermalization energies 
    #return energies_therm
end

function parallel_temper!(mc::Simulation, rank::Int64, temp::Vector{Float64})
    N_therm = mc.parameters.N_therm
    overrelax_rate = mc.parameters.overrelax_rate
    N_meas = mc.parameters.N_meas
    probe_rate = mc.parameters.probe_rate
    replica_exchange_rate = mc.parameters.replica_exchange_rate
    optimize_temperature_rate = mc.parameters.optimize_temperature_rate
    
    N = mc.spin_system.N
    S = mc.spin_system.S
    
    H_bond = mc.spin_system.H_bond
    neighbours = mc.spin_system.neighbours
    zeeman = mc.spin_system.zeeman_field
    
    N_ranks = length(temp)
    T = mc.T
    N_sweeps = N_therm + N_meas
    energies = zeros(N_sweeps)
    
    accept_count_metropolis = [0] #counts successful metropolis updates
    accept_count_swap = [0] #counts number of successful swaps
    
    if rank == 0
        mc.replica_label = "up"
    elseif rank == N_ranks-1
        mc.replica_label = "down"
    end

    n_up = 0 #number of replicas going "up" through the temperature T
    n_down = 0 #number of replicas going "down" through the temperature T
    new_spins = similar(mc.spin_system.spins) #buffer for replica exchange
    
    for sweep in 1:N_sweeps
        n_up += (mc.replica_label == "up")
        n_down += (mc.replica_label == "down")

        #do overrelaxation and metropolis with relative frequency overrelax_rate
        if sweep % overrelax_rate == 0
            metropolis!(mc.spin_system.spins, accept_count_metropolis, S, H_bond, neighbours, zeeman, N, T)
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
            swap_type = div(sweep, replica_exchange_rate)%2            
            new_spins, partner_replica_number, partner_label, accept_swap = replica_exchange!(mc.spin_system.spins, rank, E, mc.replica_number, mc.replica_label, temp, swap_type)
            
            if accept_swap[1]
                mc.spin_system.spins .= copy(new_spins)
                
                mc.replica_number = partner_replica_number
                mc.replica_label = partner_label

                #change the "direction" of the replica if it reaches the highest or lowest rank
                if rank == 0
                    mc.replica_label = "up"
                elseif rank == N_ranks - 1
                    mc.replica_label = "down"
                end
            end
            accept_count_swap += accept_swap
        end 

        #=
        #don't adjust temperatures while taking measurements
        if sweep < N_therm && sweep % optimize_temperature_rate == 0
            flow = n_up/(n_up + n_down)
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
    
    flow = n_up/(n_up + n_down)
        
    return energies, accept_count_metropolis, accept_count_swap, flow
end

function swap_adjacent!(arr::AbstractArray, rank::Int64, partner_rank::Int64, swap_type::Int64, comm::MPI.Comm)
    buffer = similar(arr)
    #prevents edge case of (R-1,R) swap if swap_type=1
    if rank%2 == swap_type && partner_rank < comm_size 
        MPI.send(arr, comm, dest=partner_rank)
        buffer = MPI.recv(comm, source=partner_rank)
    end
    #prevents edge case of (-1,0) swap if swap_type=1
    if rank%2 == 1-swap_type && partner_rank >= swap_type 
        buffer = MPI.recv(comm, source=partner_rank)
        MPI.send(arr, comm, dest=partner_rank)
    end
    arr .= copy(buffer)

    return arr
end
#do this each iteration of the loop (i.e. when it's time to try swapping)
function replica_exchange!(spins::Array{Float64,2}, rank::Int64, E_rank::Float64, replica_number::Int64, replica_label::String, temp::Vector{Float64}, swap_type::Int64)
    #swap_type=0 pairs (01)(23)..., swap_type=1 pairs 0(12)(34)...
    
    partner_rank = iseven(rank) ? rank+(-1)^swap_type : rank-(-1)^swap_type
    accept_arr = [false]
    
    #buffers
    partner_replica_number = [-1]
    partner_replica_label = ["none"]

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
        spins = swap_adjacent!(spins, rank, partner_rank, swap_type, comm)
        partner_replica_number = swap_adjacent!([replica_number], rank, partner_rank, swap_type, comm)
        partner_replica_label = swap_adjacent!([replica_label], rank, partner_rank, swap_type, comm)
    end

    return spins, partner_replica_number[1], partner_replica_label[1], accept_arr
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