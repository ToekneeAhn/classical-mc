using LinearAlgebra, StaticArrays, Random

function local_to_global(spin_local::Vector{Float64}, mu::Int64)::Vector{Float64}
    return LOCAL_BASES[mu] * spin_local
end

function unique_cubic_triplets(K::Complex{Float64}, N::Int64, N_sites::Int64)
    seen = Set{NTuple{3,Int64}}()
    triplets = NTuple{3,Int64}[]
    K_vals = NTuple{2,Float64}[]
    for n in 1:N_sites
        cs = cubic_sites_pyro(n, N)
        for idx in 1:30  # only role_first to avoid duplicates
            triplet = cs[idx]
            if triplet ∉ seen
                push!(seen, triplet)
                push!(triplets, triplet)
                i, j, k = triplet
                sub_i = get_sublattice(i, N)
                sub_j = get_sublattice(j, N)
                sub_k = get_sublattice(k, N)
                phase = GAMMA_IJ[sub_i, sub_j] * GAMMA_IJ[sub_j, sub_k]

                push!(K_vals, (2 * real(K * phase), -2 * imag(K * phase)))
            end
        end
    end
    return triplets, K_vals
end

function cubic_tensors_sparse_all(K::Complex{Float64}, N::Int64, N_sites::Int64)::Vector{NTuple{90,NTuple{2,Float64}}}
    cubic_tensors = Vector{NTuple{90,NTuple{2,Float64}}}(undef, N_sites)
    for n in 1:N_sites
        cubic_sites_n = cubic_sites_pyro(n, N)
        tensors_n = NTuple{2,Float64}[]
        sizehint!(tensors_n, 90)

        for triplet in cubic_sites_n
            i, j, k = triplet
            sub_i = get_sublattice(i, N)
            sub_j = get_sublattice(j, N)
            sub_k = get_sublattice(k, N)

            phase = GAMMA_IJ[sub_i, sub_j] * GAMMA_IJ[sub_j, sub_k]
            K_313 = 2 * real(K * phase)
            K_323 = -2 * imag(K * phase)
            push!(tensors_n, (K_313, K_323))
        end
        cubic_tensors[n] = Tuple(tensors_n)
    end
    return cubic_tensors
end

function cubic_tensors_all(K::Complex{Float64}, N::Int64, N_sites::Int64)::Vector{NTuple{90,SArray{Tuple{3,3,3}, Float64, 3, 27}}}
    cubic_tensors = Vector{NTuple{90,SArray{Tuple{3,3,3}, Float64, 3, 27}}}(undef, N_sites)
    for n in 1:N_sites
        cubic_sites_n = cubic_sites_pyro(n, N)
        cubic_tensors_n = SArray{Tuple{3,3,3}, Float64, 3, 27}[]

        for triplet in cubic_sites_n
            K_cubic = zeros(3, 3, 3)

            i, j, k = triplet
            sub_i = get_sublattice(i, N)
            sub_j = get_sublattice(j, N)
            sub_k = get_sublattice(k, N)

            phase = GAMMA_IJ[sub_i, sub_j] * GAMMA_IJ[sub_j, sub_k] #i hope this is correct lol
            K_cubic[3, 1, 3] = 2 * real(K * phase)
            K_cubic[3, 2, 3] = -2 * imag(K * phase)

            #there should be 3 indepdendent constants (K1, K2, K3) for the three independent types of "trimers"
            #assume they are the same for now
            push!(cubic_tensors_n, SArray{Tuple{3,3,3}, Float64, 3, 27}(K_cubic))
        end

        cubic_tensors[n] = Tuple(cubic_tensors_n)
    end

    return cubic_tensors
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
                    gamma = GAMMA_IJ[sub_i, sub_j]
                    zeta = -conj(gamma)

                    if sub_i != sub_j
                        SArray{Tuple{3,3},Float64,2,9}(conj(T)' * [J_pmpm*gamma -J_pm J_zpm*zeta; -J_pm J_pmpm*conj(gamma) J_zpm*conj(zeta); J_zpm*zeta J_zpm*conj(zeta) J_zz] * T)
                    else
                        SArray{Tuple{3,3},Float64,2,9}(zeros(3, 3))
                    end
                end)
        end

        push!(H_bilinear, Tuple(H_bilinear_n))
    end

    return H_bilinear
end

function effective_zeeman_field(h, z_local, local_interactions, delta_12, G, N_sites, seed=123, breaking_field=[zeros(3), zeros(3), zeros(3), zeros(3)])::Vector{NTuple{3,Float64}}
    Random.seed!(seed)

    zeeman_eff = NTuple{3,Float64}[]
    for mu in 1:4
        h_z = (h' * z_local[:, mu]) .* local_interactions[:, mu]
        h_mu = LOCAL_BASES[mu]' * h
        h_xy_quadratic = delta_12[1] .* (h_mu[1] * h_mu[3], h_mu[2] * h_mu[3], 0.0) .+ delta_12[2] .* (h_mu[2]^2 - h_mu[1]^2, 2.0 * h_mu[1] * h_mu[2], 0.0)
        h_xy_breaking = breaking_field[mu]

        for _ in 1:N_sites÷4
            random_strength = G * tan(pi * (rand() - 0.5)) #draws from a lorentzian distribution with pdf p(h) = G/pi * 1/(G^2+h^2)
            random_phase = 2 * pi * rand()

            h_xy_random = random_strength .* (cos(random_phase), sin(random_phase), 0.0)

            push!(zeeman_eff, Tuple(h_z .+ h_xy_quadratic .+ h_xy_random .+ h_xy_breaking))
        end
    end
    #vector of tuples is faster to index into later
    return zeeman_eff
end
