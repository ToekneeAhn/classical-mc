#average spin on sublattice
function spin_expec(spins)
    N_uc = div(size(spins)[2], 4)

    s_avg = zeros(3,4)
    for mu in 1:4
        s_avg[:,mu] = sum(spins[:,(N_uc*(mu-1)+1):(N_uc*mu)], dims=2)[:,1]
    end

    return s_avg/N_uc
end