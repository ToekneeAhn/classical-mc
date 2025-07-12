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
        m_avg .+= local_frames[mu] * ([0,0,1] .* local_spin_expec[:,mu])
    end

    if norm(h) > 0.0 #for nonzero field, calculate magnetization along the field
        m_avg = (m_avg' * h) * h/(norm(h)^2)
    end
    
    return m_avg
end
