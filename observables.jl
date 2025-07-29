using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::ErrorPropagator{Float64,32}
    avg_spin::ErrorPropagator{Matrix{Float64},32} #get the mean and std_error by adding extra argument 1 (for 1st dataset)
    Observables() = new(ErrorPropagator(Float64), ErrorPropagator(Float64), ErrorPropagator(zeros(Float64, 3,4)))
end

#average spin on sublattice in local frame
function spin_expec(spins::Array{Float64,2}, N::Int64)::Array{Float64,2}
    s_avg = zeros(3,4)
    
    for mu in 1:4
        s_avg[:,mu] = sum(spins[:,(N^3*(mu-1)+1):(N^3*mu)], dims=2)[:,1]
    end

    return s_avg / N^3
end

#magnetization (net moment) in global frame, along external field direction
function magnetization_global(local_spin_expec::Array{Float64,2}, local_frames::Vector{Matrix{Float64}}, h::Vector{Float64})::Vector{Float64}
    m_avg = zeros(3) 

    for mu in 1:4
        #keep all components
        #m_avg .+= local_frames[mu] * local_spin_expec[:,mu]
        
        #only keep local z component
        m_avg .+= local_frames[mu] * ([0,0,1] .* local_spin_expec[:,mu])
    end

    if norm(h) > 0.0 #for nonzero field, calculate magnetization along the field
        m_avg = (m_avg' * h) * h/(norm(h)^2)
    end
    
    return m_avg
end

function specific_heat(mc)
    E_E_sq = mc.observables.energy

    temp = mc.T
    N_sites = mc.spin_system.N_sites 

    #compute specific heat 
    C(e) = 1/temp^2 * (e[2]-e[1]*e[1]) / N_sites
    grad_C(e) = [-2.0 * 1/temp^2 * e[1] / N_sites, 1/temp^2 / N_sites] 

    heat = mean(E_E_sq, C)
    dheat = std_error(E_E_sq, grad_C)

    return heat, dheat
end

function susceptibility(mc)
    m_m_sq = mc.observables.magnetization

    temp = mc.T
    N_sites = mc.spin_system.N_sites 

    #compute specific heat 
    C(m) = 1/temp * (m[2]-m[1]*m[1]) / N_sites
    grad_C(m) = [-2.0 * 1/temp * m[1] / N_sites, 1/temp / N_sites] 

    susc = mean(m_m_sq, C)
    dsusc = std_error(m_m_sq, grad_C)

    return susc, dsusc
end