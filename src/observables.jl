using BinningAnalysis

#average spin on sublattice in local frame
function spin_expec(spins::Array{Float64,2}, N::Int64)::Array{Float64,2}
    s_avg = zeros(3,4)
    
    for mu in 1:4
        s_avg[:,mu] = sum(spins[:,(N^3*(mu-1)+1):(N^3*mu)], dims=2)[:,1]
    end

    return s_avg / N^3
end

function measure!(mc::Simulation, energy::Float64)
    spins = mc.spin_system.spins
    h = mc.spin_system.h

    # local spin expectation values
    local_spin_expec = spin_expec(spins, mc.spin_system.N)
    
    # global magnetization
    local_spin_z = (local_spin_expec .* LOCAL_INTERACTIONS)[3,:] # only z component contributes to magnetization for non-kramers doublets
    m_global = [dot(Z_LOCAL[i,:], local_spin_z) for i in 1:3] # transform to global frame
    if norm(h) > 1e-6
        m_along_field = (m_global' * h) * h/(norm(h)^2) # project onto field direction
        m_along_field = norm(m_along_field) 
    else
        m_along_field = 0.0
    end

    for i in 1:3
         push!(mc.observables.magnetization_global[i], m_global[i], m_global[i]^2, m_global[i]^4)
         for mu in 1:4
            S_i_mu = local_spin_expec[i,mu]
            push!(mc.observables.local_spin[i,mu], S_i_mu, S_i_mu^2)
            push!(mc.observables.energy_spin_covariance[i,mu], energy*S_i_mu, energy, S_i_mu) # covariance between energy and local spin component
         end
    end
    push!(mc.observables.magnetization_along_field, m_along_field, m_along_field^2, m_along_field^4)
    push!(mc.observables.energy, energy, energy^2)
end

#the same as std_error() but takes absolute value of variance 
#due to floating point error, the variance can become negative if it's too close to zero (?)
function std_error_safe(ep::ErrorPropagator, gradient::Function, lvl = BinningAnalysis._reliable_level(ep))
    return sqrt(abs(varN(ep, gradient, lvl)))
end

#specific heat per site
function specific_heat(mc::Simulation)
    E_E_sq = mc.observables.energy

    temp = mc.T
    N_sites = mc.spin_system.N_sites 

    C(e) = 1/temp^2 * (e[2]-e[1]*e[1]) / N_sites
    grad_C(e) = [-2.0 * 1/temp^2 * e[1] / N_sites, 1/temp^2 / N_sites] 

    heat = mean(E_E_sq, C)
    dheat = std_error_safe(E_E_sq, grad_C)

    return heat, dheat
end

#magnetic susceptibility per site, we multiply by N_sites because magnetization_global is per site takes care of it
function susceptibility(mc::Simulation)
    m_m_sq = mc.observables.magnetization_along_field

    temp = mc.T
    N_sites = mc.spin_system.N_sites

    chi(m) = 1/temp * N_sites * (m[2]-m[1]*m[1]) 
    grad_chi(m) = N_sites .* [-2.0 * 1/temp * m[1], 1/temp, 0.0] 

    susc = mean(m_m_sq, chi)
    dsusc = std_error_safe(m_m_sq, grad_chi)

    return susc, dsusc
end

function binder_cumulant(mc::Simulation)
    ms = mc.observables.magnetization_along_field

    U(m) = 1.0 - m[3]/(3*m[2]^2)
    grad_U(m) = [0.0, 2/3*m[3]/m[2]^3, - 1/(3*m[2]^2)] 

    U_L = mean(ms, U)
    dU_L = std_error_safe(ms, grad_U)

    return U_L, dU_L
end

function dSdT(mc::Simulation)
    #nb: 3x4 matrix of ErrorPropagator, not ErrorPropagator of 3x4 matrices
    HS = mc.observables.energy_spin_covariance 
    
    dsdt_comp = zeros(3,4)
    d_dsdt_comp = similar(dsdt_comp)
    
    temp = mc.T
    
    cov(v) = 1/temp^2 * (v[1] - v[2]*v[3])
    grad_cov(v) = 1/temp^2 .* [1.0, -v[3], -v[2]]
    
    for i in 1:3
        for mu in 1:4
            dsdt_comp[i,mu] = mean(HS[i,mu], cov)
            d_dsdt_comp[i,mu] = std_error_safe(HS[i,mu], grad_cov)
        end
    end
    
    return dsdt_comp, d_dsdt_comp
end

function local_spin_expectation(mc::Simulation)
    local_spin_expec = zeros(3,4)
    d_local_spin_expec = similar(local_spin_expec)

    for i in 1:3
        for mu in 1:4
            local_spin_expec[i,mu] = mean(mc.observables.local_spin[i,mu], 1) # 1 refers to the index of the dataset
            d_local_spin_expec[i,mu] = std_error(mc.observables.local_spin[i,mu], 1)
        end
    end
    return local_spin_expec, d_local_spin_expec
end

function magnetization_global(mc::Simulation)
    m_global = zeros(3)
    d_m_global = similar(m_global)

    for i in 1:3
        m_global[i] = mean(mc.observables.magnetization_global[i], 1) 
        d_m_global[i] = std_error(mc.observables.magnetization_global[i], 1)
    end
    return m_global, d_m_global
end

function compute_observables!(mc::Simulation)
    measurements = Dict("specific_heat" => specific_heat, 
                        "susceptibility" => susceptibility, 
                        "binder_cumulant" => binder_cumulant, 
                        "local_spin" => local_spin_expectation,
                        "dSdT" => dSdT,
                        "magnetization_global" => magnetization_global)

    # default measurements: energy, magnetization along field
    # could make specific functions for these for cleaner code
    mc.observables.output["energy"] = mean(mc.observables.energy, 1)
    mc.observables.output["energy_err"] = std_error(mc.observables.energy, 1)
    mc.observables.output["magnetization"] = mean(mc.observables.magnetization_along_field, 1)
    mc.observables.output["magnetization_err"] = std_error(mc.observables.magnetization_along_field, 1)

    for measurement_name in keys(measurements)
        measurement_func = measurements[measurement_name]
        result, error = measurement_func(mc)
        mc.observables.output[measurement_name] = result
        mc.observables.output["$(measurement_name)_err"] = error
    end
end