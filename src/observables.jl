using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::ErrorPropagator{Float64,32}
    avg_spin::ErrorPropagator{Matrix{Float64},32} #get the mean and std_error by adding extra argument 1 (for 1st dataset)
    energy_spin_covariance::Matrix{ErrorPropagator{Float64,32}}
    Observables() = new(ErrorPropagator(Float64, N_args=2), ErrorPropagator(Float64, N_args=3), 
    ErrorPropagator(zeros(Float64, 3,4), zeros(Float64, 3,4)), [ErrorPropagator(Float64,N_args=3) for i=1:3,j=1:4])
end

#average spin on sublattice in local frame
function spin_expec(spins::Array{Float64,2}, N::Int64)::Array{Float64,2}
    s_avg = zeros(3,4)
    
    for mu in 1:4
        s_avg[:,mu] = sum(spins[:,(N^3*(mu-1)+1):(N^3*mu)], dims=2)[:,1]
    end

    return s_avg / N^3
end

#magnetization (net moment) per site in global frame, along external field direction
function magnetization_global(local_spin_expec::Array{Float64,2}, local_frames::Vector{Matrix{Float64}}, h::Vector{Float64})::Vector{Float64}
    m_avg = zeros(3) 

    for mu in 1:4
        m_avg .+= local_frames[mu] * ([0,0,1] .* local_spin_expec[:,mu])
    end

    if norm(h) > 0.0 #for nonzero field, calculate magnetization along the field
        m_avg = (m_avg' * h) * h/(norm(h)^2)
    end
    
    return m_avg
end

#the same as std_error() but takes absolute value of variance 
#due to floating point error, the variance can become negative if it's too close to zero (?)
function std_error_safe(ep::ErrorPropagator, gradient::Function, lvl = BinningAnalysis._reliable_level(ep))
    return sqrt(abs(varN(ep, gradient, lvl)))
end

#specific heat per site
function specific_heat(mc)
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
function susceptibility(mc)
    m_m_sq = mc.observables.magnetization

    temp = mc.T
    N_sites = mc.spin_system.N_sites

    chi(m) = 1/temp * N_sites * (m[2]-m[1]*m[1]) 
    grad_chi(m) = N_sites .* [-2.0 * 1/temp * m[1], 1/temp, 0.0] 

    susc = mean(m_m_sq, chi)
    dsusc = std_error_safe(m_m_sq, grad_chi)

    return susc, dsusc
end

function binder_cumulant(mc)
    ms = mc.observables.magnetization

    U(m) = 1.0 - m[3]/(3*m[2]^2)
    grad_U(m) = [0.0, 2/3*m[3]/m[2]^3, - 1/(3*m[2]^2)] 

    U_L = mean(ms, U)
    dU_L = std_error_safe(ms, grad_U)

    return U_L, dU_L
end

function dSdT(mc)
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