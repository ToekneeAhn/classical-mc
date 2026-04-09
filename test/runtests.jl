using Test
using LinearAlgebra
using PyroClassicalMC

@testset "Constants and Local Bases" begin
    # shape of constants and local bases
    @test size(Z_LOCAL) == (3, 4)
    @test size(LOCAL_INTERACTIONS) == (3, 4)
    @test length(LOCAL_BASES) == 4

    for mu in 1:4
        @test size(LOCAL_BASES[mu]) == (3, 3)
        # local bases are orthonormal
        @test isapprox(LOCAL_BASES[mu]' * LOCAL_BASES[mu], I(3), atol = 1e-12)
    end

    # local z axes sum to zero
    @test isapprox(norm(sum(Z_LOCAL[:, mu] for mu in 1:4)), 0.0, atol = 1e-12)
end
   
@testset "Lattice Indexing and Connectivity" begin
    N = 3
    N_sites = 4 * N^3

    # test sublattice-indexed pyrochlore coordinates (SIPC) and flat indexing
    @test get_sublattice(1, N) == 1
    @test get_sublattice(N^3 + 1, N) == 2
    @test get_sublattice(2 * N^3 + 1, N) == 3
    @test get_sublattice(3 * N^3 + 1, N) == 4

    @test pos_mod([-1, 0, 1], N) == [2, 0, 1]

    for n in (1, 2, N^3, N^3 + 1, 2 * N^3 + 3, 4 * N^3)
        sipc = sipc_index_3D(n, N)
        @test flat_index_3D(sipc) == n
    end

    # test neighbours and cubic sites for a single site
    n0 = div(N_sites, 2)
    neighs = neighbours_pyro(n0, N)
    @test length(neighs) == 6
    @test length(unique(neighs)) == 6
    @test all(1 .<= collect(neighs) .<= N_sites)

    cubic_n0 = cubic_sites_pyro(n0, N)
    @test length(cubic_n0) == 90

    cubic_all = cubic_sites_all(N, N_sites)
    @test length(cubic_all) == N_sites

    pairs_i, pairs_j, pairs_k = cubic_pairs_split_all(cubic_all, N_sites)
    @test length(pairs_i) == N_sites
    @test length(pairs_j) == N_sites
    @test length(pairs_k) == N_sites
    @test length(pairs_i[n0]) == 30
    @test length(pairs_j[n0]) == 30
    @test length(pairs_k[n0]) == 30
end

@testset "Spin Initialization" begin
    N = 2
    S = 1.0
    spins = spins_initial_pyro(N, S)

    @test size(spins) == (3, 4 * N^3)

    for j in axes(spins, 2)
        @test norm(spins[:, j]) ≈ S atol = 1e-12
    end

    s_new = (1.0, 2.0, 3.0) ./ norm((1.0, 2.0, 3.0)) .* S
    set_spin!(spins, s_new, 1)
    @test get_spin(spins, 1) == s_new
end
