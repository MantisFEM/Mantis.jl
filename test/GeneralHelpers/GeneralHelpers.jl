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

@testset "get_from_cache" begin
    dict = GH.cache_dict(String, Int, Val(3))
    @test isa(dict, Dict{String, Int})
    @test GH.get_from_cache(String, Int, "test", s -> length(s), Val(3)) == 4
    @test GH.get_from_cache(String, Int, "testing", s -> length(s), Val(3)) == 7
    @test dict == Dict("test" => 4, "testing" => 7)
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

@testset "depth" begin
    # Scalar types have depth 0
    @test GH.depth(Float64) == 0
    @test GH.depth(Int) == 0
    @test GH.depth(String) == 0
    @test GH.depth(Tuple{Int, Int}) == 0
    @test GH.depth(Any) == 0

    # Single-level arrays have depth 1
    @test GH.depth(Vector{Float64}) == 1
    @test GH.depth(Matrix{Int}) == 1
    @test GH.depth(Array{String, 3}) == 1

    # Two-level nested arrays have depth 2
    @test GH.depth(Vector{Vector{Float64}}) == 2
    @test GH.depth(Vector{Matrix{Float64}}) == 2
    @test GH.depth(Vector{Array{String, 1}}) == 2

    # Three-level nested arrays have depth 3
    @test GH.depth(Vector{Vector{Matrix{Float64}}}) == 3
    @test GH.depth(Vector{Vector{Vector{Int}}}) == 3
    @test GH.depth(Vector{Matrix{Vector{String}}}) == 3
end

@testset "matches" begin
    # Scalar type matches zero sizes
    @test GH.matches(Float64) == true
    @test GH.matches(Int, 3) == false

    # Single-level
    @test GH.matches(Vector{Int}, 5) == true
    # too many levels
    @test GH.matches(Vector{String}, 5, 3) == false
    # inner ndims mismatch
    @test GH.matches(Vector{Float64}, (2, 3)) == false

    @test GH.matches(Matrix{Int}, (2, 3)) == true
    # inner ndims mismatch
    @test GH.matches(Matrix{Float64}, (2,)) == false

    # Two-level
    @test GH.matches(Vector{Matrix{Float64}}, 2, (2, 4)) == true
    # inner ndims mismatch
    @test GH.matches(Vector{Matrix{String}}, 3, 2) == false
    # missing inner level
    @test GH.matches(Vector{Matrix{Int}}, 3) == false
    # too many levels
    @test GH.matches(Vector{Vector{Int32}}, 5, 3, 1, 4) == false

    # Three-level
    @test GH.matches(Vector{Vector{Matrix{Float64}}}, 3, 1, (2, 2)) == true
    @test GH.matches(Vector{Vector{Matrix{Float64}}}, 3, 1) == false
    @test GH.matches(Vector{Vector{Matrix{Float64}}}, 3, 1, 10, 20) == false
end

@testset "haszero" begin
    @test GH.haszero(0) == true
    @test GH.haszero(1) == false
    @test GH.haszero(5) == false

    @test GH.haszero((0,)) == true
    @test GH.haszero((1,)) == false
    @test GH.haszero((1, 2, 3)) == false
    @test GH.haszero((1, 0, 3)) == true
    @test GH.haszero(()) == false
    @test GH.haszero((1, 2, (1, 0), 3)) == true
    @test GH.haszero((1, 2, (1, (1, (1, (1, 0)))), 3)) == true
end

@testset "allocate" begin
    @testset "Two-level" begin
        A = Vector{Vector{Float64}}(undef, 3, 4)
        @test isa(A, Vector{Vector{Float64}})
        @test length(A) == 3
        @test all(x -> isa(x, Vector{Float64}) && length(x) == 4, A)

        B = Vector{Matrix{String}}(undef, 3, (2, 2))
        @test isa(B, Vector{Matrix{String}})
        @test length(B) == 3
        @test all(x -> isa(x, Matrix{String}) && size(x) == (2, 2), B)
    end

    @testset "Three-level" begin
        A = Vector{Matrix{Vector{Float64}}}(undef, 3, (2, 2), 1)
        @test isa(A, Vector{Matrix{Vector{Float64}}})
        @test length(A) == 3
        @test all(x -> isa(x, Matrix{Vector{Float64}}) && size(x) == (2, 2), A)
        @test all(M -> all(V -> isa(V, Vector{Float64}) && length(V) == 1, M), A)
    end

    @testset "Zero length" begin
        # Zero-length leaf arrays
        A = Vector{Vector{Float64}}(undef, 2, 0)
        @test isa(A, Vector{Vector{Float64}})
        @test length(A) == 2
        @test all(x -> length(x) == 0, A)

        A = Vector{Matrix{String}}(undef, 2, (0, 3))
        @test isa(A, Vector{Matrix{String}})
        @test all(x -> size(x) == (0, 3), A)
    end

    @testset "errors" begin
        # Depth mismatch: too many sizes
        @test_throws DimensionMismatch Vector{Float64}(undef, (3, 4))

        # ndims mismatch: passing a scalar size to a Matrix level
        @test_throws DimensionMismatch Vector{Matrix{Float64}}(undef, 3, 2)

        # Zero at a non-leaf level is not allowed
        @test_throws ArgumentError Vector{Vector{Float64}}(undef, 0, 4)

        # Zero at multiple non-leaf levels
        @test_throws ArgumentError Vector{Matrix{Matrix{Float64}}}(undef, 0, (1, 0), (3, 3))
    end
end

end
