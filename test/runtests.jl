using Test
using LinearAlgebra
using HDF5
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

@testset "write_observables I/O" begin
    N = 2
    S = 1.0
    N_sites = 4 * N^3

    spins = spins_initial_pyro(N, S)
    neighbours = neighbours_all(N, N_sites)
    Js = zeros(4)
    h = [0.0, 0.0, 1.0]
    delta_12 = [0.0, 0.0]
    disorder_strength = 0.0

    H_bilinear = H_bilinear_all(Js, N, N_sites)
    zeeman = effective_zeeman_field(h, Z_LOCAL, LOCAL_INTERACTIONS, delta_12, disorder_strength, N_sites, 123)

    system = SpinSystem(spins, S, N, N_sites, Js, h, delta_12, disorder_strength, neighbours, H_bilinear, zeeman)
    params = MCParams(0, 0, 0, 0, 0, 0, 0)
    simulation = Simulation(system, 1.0, params, Observables(), 0, "none")

    simulation.observables.output["energy"] = 2.5
    simulation.observables.output["magnetization"] = 0.25
    simulation.observables.output["magnetization_global"] = [1.0, 2.0, 3.0]
    simulation.observables.output["local_spin"] = reshape(collect(1.0:12.0), 3, 4)

    mktempdir() do tmpdir
        path_no_spins = joinpath(tmpdir, "observables_no_spins.h5")
        path_with_spins = joinpath(tmpdir, "observables_with_spins.h5")

        write_observables(path_no_spins, simulation)
        write_observables(path_with_spins, simulation, spins)

        h5open(path_no_spins, "r") do fid
            @test haskey(fid, "energy")
            @test haskey(fid, "magnetization")
            @test haskey(fid, "magnetization_global")
            @test haskey(fid, "local_spin")
            @test !haskey(fid, "spins")
        end

        h5open(path_with_spins, "r") do fid
            @test haskey(fid, "spins")
            @test size(read(fid["spins"])) == size(spins)
            @test read(fid["energy"]) == 2.5
        end
    end
end

@testset "collect_hsweep basic aggregation" begin
    mktempdir() do tmpdir
        results_dir = joinpath(tmpdir, "results")
        save_dir = joinpath(tmpdir, "save")
        mkpath(results_dir)
        mkpath(save_dir)

        file_prefix = "toy_h"
        N_h = 3
        N_ranks = 1

        parameters_path = joinpath(results_dir, "toy_parameters.h5")
        h5open(parameters_path, "w") do fid
            param_gr = create_group(fid, "parameters")
            param_gr["h_sweep"] = collect(range(0.1, 0.3, length=N_h))
            param_gr["Ts"] = [0.5]
            param_gr["N"] = 1
        end

        for n in 1:N_h
            fname = joinpath(results_dir, "$(file_prefix)$(n)_0.h5")
            h5open(fname, "w") do fid
                fid["magnetization"] = 1.0 * n
                fid["magnetization_global"] = [1.0 * n, 2.0 * n, 3.0 * n]
                fid["energy"] = -1.0 * n
                fid["specific_heat"] = 0.1 * n
                fid["susceptibility"] = 0.2 * n
                fid["binder_cumulant"] = 0.3 * n
                fid["local_spin"] = fill(1.0 * n, 3, 4)
                fid["dSdT"] = fill(-1.0 * n, 3, 4)
            end
        end

        collect_hsweep(results_dir, file_prefix, save_dir, parameters_path)

        outpath = joinpath(save_dir, "$(file_prefix)sweep.h5")
        @test isfile(outpath)

        h5open(outpath, "r") do fid
            @test haskey(fid, "parameters")
            @test haskey(fid, "rank_0")
            @test read(fid["missing_h_points"]) == Int64[]

            rank0 = fid["rank_0"]
            @test size(read(rank0["magnetization"])) == (N_h,)
            @test size(read(rank0["magnetization_global"])) == (N_h, 3)
            @test size(read(rank0["local_spin"])) == (N_h, 3, 4)
            @test size(read(rank0["dSdT"])) == (N_h, 3, 4)

            @test read(rank0["magnetization"]) == [1.0, 2.0, 3.0]
            @test read(rank0["energy"]) == [-1.0, -2.0, -3.0]
            @test read(fid["parameters"]["Ts"]) == [0.5]
        end
    end
end
