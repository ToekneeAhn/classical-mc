using LinearAlgebra, StaticArrays, BinningAnalysis, Random, MPI
using Interpolations, ForwardDiff, Integrals, Printf 

#S_new is a tuple for performance purposes
function set_spin!(spins::Array{Float64,2}, S_new::NTuple{3,Float64}, site::Int64)
    @inbounds spins[1, site] = S_new[1]
    @inbounds spins[2, site] = S_new[2]
    @inbounds spins[3, site] = S_new[3]
end

function get_spin(spins::Array{Float64,2}, site::Int64)::NTuple{3, Float64}
    @inbounds return (spins[1, site], spins[2, site], spins[3, site])
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

        #cubic part
        if !isempty(sys.H_cubic_sparse)
            Hc = sys.H_cubic_sparse[n]
            pairs_i = sys.cubic_pairs_i[n]
            pairs_j = sys.cubic_pairs_j[n]
            pairs_k = sys.cubic_pairs_k[n]

            # n is first index (i): dE/dSi = K[a,b,c]*Sj[b]*Sk[c] → only a=3,b∈{1,2},c=3 nonzero
            @inbounds for p in 1:30
                K313, K323 = Hc[p]
                j, k = pairs_i[p]
                Sj1 = S[1, j]; Sj2 = S[2, j]
                Sk3 = S[3, k]
                Hz += (K313*Sj1 + K323*Sj2) * Sk3
            end

            # n is second index (j): dE/dSj = K[a,b,c]*Si[a]*Sk[c] → only a=3,b∈{1,2},c=3 nonzero
            @inbounds for p in 1:30
                K313, K323 = Hc[p+30]
                i, k = pairs_j[p]
                prod = S[3, i] * S[3, k]
                Hx += K313 * prod
                Hy += K323 * prod
            end

            # n is third index (k): dE/dSk = K[a,b,c]*Si[a]*Sj[b] → only a=3,b∈{1,2},c=3 nonzero
            @inbounds for p in 1:30
                K313, K323 = Hc[p+60]
                i, j = pairs_k[p]
                Sj1 = S[1, j]; Sj2 = S[2, j]
                Si3 = S[3, i]
                Hz += Si3 * (K313*Sj1 + K323*Sj2)
            end
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

        #zeeman contribution
        E_zeeman += - dot(sys.zeeman_field[n], S_n)
    end
    if length(sys.cubic_sites) > 0
        @inbounds for t in eachindex(sys.unique_triplets)
            i, j, k = sys.unique_triplets[t]
            K313, K323 = sys.unique_H_cubic_vals[t]
            Si3 = sys.spins[3, i]; Sj1 = sys.spins[1, j]; Sj2 = sys.spins[2, j]; Sk3 = sys.spins[3, k]
            E_cubic += (K313*Sj1 + K323*Sj2) * Si3 * Sk3
        end        
    end
    
    #total energy, not energy per site
    return E_bilinear/2.0 + E_cubic + E_zeeman
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

#picks a point on the unit sphere uniformly and returns Cartesian coordinates (Sx,Sy,Sz), then scales magnitude by S
function sphere_pick(S::Float64)::NTuple{3,Float64}
    phi = 2*pi*rand()
    z = 2*rand() - 1
    return S .* (sqrt(1-z^2)*cos(phi), sqrt(1-z^2)*sin(phi), z)
end

#metropolis algorithm 
#TODO: implement adaptive metropolis which adjusts theta_max depending on the acceptance rate
function metropolis!(sys::SpinSystem, accept_count::Array{Int64,1}, T::Float64)
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

#deterministic updates (aligning spins to their local field)
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
    measure!(mc, E_pyro(mc.spin_system))
    compute_observables!(mc)
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
            measure!(mc, E)
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

    # calculate final observables 
    compute_observables!(mc)
    
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