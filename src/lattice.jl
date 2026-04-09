#assume periodic boundary conditions, so take unit cell positions mod N
function pos_mod(x::Vector{Int64}, m::Int64)::Vector{Int64}
    return ((x.%m).+m).%m
end

function get_sublattice(n::Int64, N::Int64)::Int64
    return div(n - 1, N^3) + 1
end

#SIPC to 1D index
function flat_index_3D(r_mu::SIPC)::Int64
    nx, ny, nz = r_mu.r
    mu = r_mu.mu
    N = r_mu.N
    return N^2 * nx + N * ny + nz + (mu - 1) * N^3 + 1
end

#1D index to SIPC
function sipc_index_3D(n::Int64, N::Int64)::SIPC
    mu = get_sublattice(n, N)

    #label unit cell positions (nx,ny,nz) by the base N representation (nx ny nz)_N
    return SIPC(reverse(digits(n - 1 - (mu - 1) * N^3, base=N, pad=3)), mu, N)
end

#6 neighbours of a pyrochlore lattice site in a tuple
function neighbours_pyro(n::Int64, N::Int64)::NTuple{6,Int64}
    mu = get_sublattice(n, N)
    r_mu = sipc_index_3D(n, N).r

    neighbours_flat = pos_mod([N^3, 2 * N^3, 3 * N^3] .+ (n - 1), 4 * N^3) .+ 1 #flat indices of intra-tetrahedron neighbours

    ee = [0 0 0; 1 0 0; 0 1 0; 0 0 1]'

    for l in 1:4
        if l != mu
            r_p = pos_mod(r_mu + ee[:, mu] - ee[:, l], N)  #inter-tetrahedron neighbours
            r_neighbour = SIPC(r_p, l, N)
            append!(neighbours_flat, flat_index_3D(r_neighbour))
        end
    end
    return Tuple(neighbours_flat)
end

function cubic_sites_pyro(i::Int64, N::Int64)::NTuple{90,NTuple{3,Int64}}
    neigh_i = neighbours_pyro(i, N)                # 6 neighbours of i
    role_first = NTuple{3,Int64}[]                 # (i, j, k)
    role_second = NTuple{3,Int64}[]                # (j, i, j2)
    role_third = NTuple{3,Int64}[]                 # (k, j, i)

    sizehint!(role_first, 30)
    sizehint!(role_second, 30)
    sizehint!(role_third, 30)

    # Collect endpoint and middle triplets separately
    @inbounds for j in neigh_i
        neigh_j = neighbours_pyro(j, N)
        # i as first and third positions (endpoint triplets)
        @inbounds for k in neigh_j
            if k != i
                push!(role_first, (i, j, k))   # i first
                push!(role_third, (k, j, i))   # i third
            end
        end
        # i as second position (middle triplets)
        @inbounds for j2 in neigh_i
            if j2 != j
                push!(role_second, (j, i, j2))  # i second
            end
        end
    end

    @assert length(role_first) == 30
    @assert length(role_second) == 30
    @assert length(role_third) == 30

    all_triplets = vcat(role_first, role_second, role_third)
    return Tuple(all_triplets)::NTuple{90,NTuple{3,Int64}}
end

function cubic_sites_all(N::Int64, N_sites::Int64)
    cubic_sites = Vector{NTuple{90,NTuple{3,Int64}}}(undef, N_sites)

    for n in 1:N_sites
        cubic_sites[n] = cubic_sites_pyro(n, N)
    end

    return cubic_sites
end

"""
    cubic_pairs_split_all(cubic_sites, N_sites)

Split each site's 90 cubic triplets into three role-specific lists of 30 pairs:
 - cubic_pairs_i[n]: pairs (j,k) when triplet is (n,j,k)
 - cubic_pairs_j[n]: pairs (i,k) when triplet is (i,n,k)
 - cubic_pairs_k[n]: pairs (i,j) when triplet is (i,j,n)
"""
function cubic_pairs_split_all(cubic_sites::Vector{NTuple{90,NTuple{3,Int64}}}, N_sites::Int64)
    pairs_i = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)
    pairs_j = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)
    pairs_k = Vector{NTuple{30,NTuple{2,Int64}}}(undef, N_sites)

    for n in 1:N_sites
        cs = cubic_sites[n]
        pairs_i[n] = Tuple([triplet[2:3] for triplet in cs[1:30]])
        pairs_j[n] = Tuple([(triplet[1], triplet[3]) for triplet in cs[31:60]])
        pairs_k[n] = Tuple([triplet[1:2] for triplet in cs[61:90]])
    end

    return pairs_i, pairs_j, pairs_k
end

function neighbours_all(N::Int64, N_sites::Int64)
    coord_num = 6

    neighbours = NTuple{coord_num, Int64}[]

    for n = 1:N_sites
        push!(neighbours, neighbours_pyro(n, N))
    end

    return neighbours
end
