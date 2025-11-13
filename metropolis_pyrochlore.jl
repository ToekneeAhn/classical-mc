using LinearAlgebra, StaticArrays, BinningAnalysis, Random, MPI, Interpolations, ForwardDiff, Integrals, Printf

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
    #cubic_sites::Vector{NTuple{N3,NTuple{3,Int64}}} #list of cubic interaction site tuples for each site
    #H_cubic::SArray{Tuple{3,3,3},Float64} #assume a cubic interaction with no position-dependence
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

function neighbours_all(N, N_sites)
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

function local_field_pyro(sys::SpinSystem, n::Int64)::NTuple{3,Float64}
    neighbours = sys.neighbours[n]
    H_bilinear_n = sys.H_bilinear[n]
    
    #avoid matrix multiplication for performance purposes
    Hx=0.0
    Hy=0.0
    Hz=0.0
    
    @inbounds begin
        for m in eachindex(neighbours)
            sx, sy, sz = get_spin(sys.spins, neighbours[m])
            
            H_ij = H_bilinear_n[m]
            Hx += H_ij[1,1] * sx + H_ij[1,2] * sy + H_ij[1,3] * sz
            Hy += H_ij[2,1] * sx + H_ij[2,2] * sy + H_ij[2,3] * sz
            Hz += H_ij[3,1] * sx + H_ij[3,2] * sy + H_ij[3,3] * sz
        end
    end

    h_z = sys.zeeman_field[n]
    return (Hx-h_z[1], Hy-h_z[2], Hz-h_z[3])
end

function E_pyro(sys::SpinSystem)::Float64
    E = 0.0
    
    for n in 1:sys.N_sites
        #quadratic interaction, divide by 2 because each bond counted twice
        S_n = get_spin(sys.spins, n)

        for m in eachindex(sys.neighbours[n])
            #use a view to avoid allocating an array
            @inbounds E += 0.5 * dot(S_n, sys.H_bilinear[n][m] * view(sys.spins, :, sys.neighbours[n][m]))
        end

        #zeeman contribution
        @inbounds E += - dot(sys.zeeman_field[n], S_n)
    end
    
    #total energy, not energy per site
    return E
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
            @printf("T=%.6f: %.3f%%\n", T, accept_count[1]/(N_sites*N_therm/overrelax_rate)*100)
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