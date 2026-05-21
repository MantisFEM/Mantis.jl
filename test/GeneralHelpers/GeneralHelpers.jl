module GeneralHelpersTests

using Mantis
using Mantis: GeneralHelpers as GH
using Test

@testset "export_path" begin
    new_directory = [pwd(), "new", "directory"]
    new_file = "new_file.vtu"
    another_file = "another_file.vtu"
    output_file = GH.export_path(new_directory, new_file)
    # Create a new directory
    @test isdir(joinpath(new_directory...))
    @test joinpath(new_directory..., new_file) == output_file
    # Re-use a directory
    output_file = GH.export_path(new_directory, another_file)
    @test joinpath(new_directory..., another_file) == output_file
    # Remove created directory
    rm(joinpath(new_directory[1:(end - 1)]...); recursive=true)
end

@testset "num_der_indices" begin
    n = 1
    for i in 0:5
        @test GH.num_der_indices(n, i) == 1
    end

    d = 1
    for i in 1:5
        @test GH.num_der_indices(i, d) == i
        @test GH.num_der_indices(i, 0) == 1
    end

    n = 2
    @test GH.num_der_indices(n, 2) == 3
    @test GH.num_der_indices(n, 3) == 4
    @test GH.num_der_indices(n, 4) == 5

    n = 3
    @test GH.num_der_indices(n, 2) == 6
    @test GH.num_der_indices(n, 3) == 10
    @test GH.num_der_indices(n, 4) == 15
end

@testset "cache_dict" begin
    a = GH.cache_dict(Int, String)
    @test isa(a, Dict{Int, String})
    a[1] = "one"
    @test GH.cache_dict(Int, String) == Dict(1 => "one")
    b = GH.cache_dict(Int, String)
    @test a === b
    c = GH.cache_dict(Int, String, Val(2))
    @test !(a === c)
end

@testset "get_derivative_idx" begin
    # 1D
    @test_throws ArgumentError GH._get_derivative_idx((-1,))
    for d in 0:4
        @test GH._get_derivative_idx((d,)) == 1
    end
    # 2D
    @test_throws ArgumentError GH._get_derivative_idx((0, -1))
    @test GH._get_derivative_idx((0, 0)) == 1
    @test GH._get_derivative_idx((1, 0)) == 1
    @test GH._get_derivative_idx((0, 1)) == 2
    @test GH._get_derivative_idx((0, 2)) == 1
    @test GH._get_derivative_idx((1, 1)) == 2
    @test GH._get_derivative_idx((2, 0)) == 3
    # 3D
    @test_throws ArgumentError GH._get_derivative_idx((0, 0, -1))
    @test GH._get_derivative_idx((0, 0, 0)) == 1
    @test GH._get_derivative_idx((1, 0, 0)) == 1
    @test GH._get_derivative_idx((0, 1, 0)) == 2
    @test GH._get_derivative_idx((0, 0, 1)) == 3
    @test GH._get_derivative_idx((0, 0, 2)) == 1
    @test GH._get_derivative_idx((0, 1, 1)) == 2
    @test GH._get_derivative_idx((0, 2, 0)) == 3
    @test GH._get_derivative_idx((1, 0, 1)) == 4
    @test GH._get_derivative_idx((1, 1, 0)) == 5
    @test GH._get_derivative_idx((2, 0, 0)) == 6

    der_idx_val = Val(7)
    key_dict = GH.cache_dict(NTuple{3, Int}, Int, der_idx_val)
    @test isa(key_dict, Dict{NTuple{3, Int}, Int})
    @test GH.get_derivative_idx((0, 0, 0), der_idx_val) == 1
    @test GH.get_derivative_idx((1, 0, 0), der_idx_val) == 1
    @test GH.get_derivative_idx((0, 1, 0), der_idx_val) == 2
    @test GH.get_derivative_idx((0, 0, 1), der_idx_val) == 3
    @test GH.get_derivative_idx((0, 0, 2), der_idx_val) == 1
    @test GH.get_derivative_idx((0, 1, 1), der_idx_val) == 2
    @test GH.get_derivative_idx((0, 2, 0), der_idx_val) == 3
    @test GH.get_derivative_idx((1, 0, 1), der_idx_val) == 4
    @test GH.get_derivative_idx((1, 1, 0), der_idx_val) == 5
    @test GH.get_derivative_idx((2, 0, 0), der_idx_val) == 6
    @test key_dict == Dict(
        (0, 0, 0) => 1,
        (1, 0, 0) => 1,
        (0, 1, 0) => 2,
        (0, 0, 1) => 3,
        (0, 0, 2) => 1,
        (0, 1, 1) => 2,
        (0, 2, 0) => 3,
        (1, 0, 1) => 4,
        (1, 1, 0) => 5,
        (2, 0, 0) => 6,
    )
end

@testset "integer_sums" begin
    #1D
    val_one = Val(1)
    @test GH._integer_sums(-1, val_one) == Tuple{Int}[]
    for i in 0:4
        @test GH._integer_sums(i, val_one) == [(i,)]
    end

    #2D
    @test GH._integer_sums(-1, Val(2)) == Tuple{Int, Int}[]
    @test GH._integer_sums(0, Val(2)) == [(0, 0)]
    @test GH._integer_sums(1, Val(2)) == [(0, 1), (1, 0)]
    @test GH._integer_sums(2, Val(2)) == [(0, 2), (1, 1), (2, 0)]

    #3D
    @test GH._integer_sums(-1, Val(3)) == Tuple{Int, Int, Int}[]
    @test GH._integer_sums(0, Val(3)) == [(0, 0, 0)]
    @test GH._integer_sums(1, Val(3)) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    @test GH._integer_sums(2, Val(3)) ==
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    int_sums_val = Val(8)
    key_dict = GH.cache_dict(Int, Vector{NTuple{3, Int}}, int_sums_val)
    @test isa(key_dict, Dict{Int, Vector{NTuple{3, Int}}})
    @test GH.integer_sums(0, Val(3), int_sums_val) == [(0, 0, 0)]
    @test GH.integer_sums(1, Val(3), int_sums_val) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    @test GH.integer_sums(2, Val(3), int_sums_val) ==
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    @test key_dict == Dict(
        0 => [(0, 0, 0)],
        1 => [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
        2 => [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)],
    )

    @test GH.integer_sums(0, 2, Val(3)) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 2),
        (0, 1, 1),
        (0, 2, 0),
        (1, 0, 1),
        (1, 1, 0),
        (2, 0, 0),
    ]
end

end
