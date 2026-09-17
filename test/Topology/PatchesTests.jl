module PatchesTestModule

using Test
using Mantis
import MeshCore

include("../TestHelpers.jl")

############################################################################################
#                                   Test per Patch Type                                    #
############################################################################################
@testset "Line" begin
    expected_ir = Dict(
        (1, 1) => Vector{Vector{Int}}(),
        (1, 2) => [[1], [1]],
        (2, 1) => [[1, 2]],
        (2, 2) => Vector{Vector{Int}}(),
    )
    expected_size_per_dim = Dict(1 => 2, 2 => 1)
    expected_id_to_position = Dict((0, 1) => (-1,), (0, 2) => (1,), (1, 1) => (0,))
    expected_position_to_id = Dict((-1,) => 1, (1,) => 2, (0,) => 1)
    expected_dof_division_to_id = Dict((0, 1) => 1, (0, 2) => 3, (1, 1) => 2)

    @test Topology.get_manifold_dim(Topology.LINE) == 1
    @test Topology.get_incidence_relations_dim(Topology.LINE) == 2
    @test Topology.get_num_patch_vertices(Topology.LINE) == 2
    @test Topology.get_meshcore_patch(Topology.LINE) == MeshCore.L2
    @test @expected expected_ir k -> Topology.LINE[k[1], k[2]]
    @test size(Topology.LINE) == (2, 1)
    @test @expected expected_size_per_dim k -> size(Topology.LINE, k)
    @test @expected expected_id_to_position k ->
        Topology.id_to_position(Topology.LINE, k[1], k[2])
    @test @expected expected_position_to_id k -> Topology.position_to_id(Topology.LINE, k)
    @test @expected expected_dof_division_to_id k ->
        Topology.id_to_dof_division(Topology.LINE, k[1], k[2])
    @test isnothing(Topology.get_skeleton_patch(Topology.LINE))
end

@testset "Quad" begin
    expected_ir = Dict(
        (1, 1) => Vector{Vector{Int}}(),
        (1, 2) => [[1, 3], [2, 3], [2, 4], [1, 4]],
        (1, 3) => [[1], [1], [1], [1]],
        (2, 1) => [[1, 4], [2, 3], [1, 2], [4, 3]],
        (2, 2) => Vector{Vector{Int}}(),
        (2, 3) => [[1], [1], [1], [1]],
        (3, 1) => [[1, 2, 3, 4]],
        (3, 2) => [[1, 2, 3, 4]],
        (3, 3) => Vector{Vector{Int}}(),
    )
    expected_size_per_dim = Dict(1 => 4, 2 => 4, 3 => 1)
    expected_id_to_position = Dict(
        (0, 1) => (-1, -1),
        (0, 2) => (1, -1),
        (0, 3) => (1, 1),
        (0, 4) => (-1, 1),
        (1, 1) => (-1, 0),
        (1, 3) => (0, -1),
        (1, 4) => (0, 1),
        (2, 1) => (0, 0),
    )
    expected_position_to_id = Dict(
        (-1, 1) => 1,
        (1, -1) => 2,
        (1, 1) => 3,
        (-1, 1) => 4,
        (-1, 0) => 1,
        (1, 0) => 2,
        (0, -1) => 3,
        (0, 1) => 4,
        (0, 0) => 1,
    )
    expected_dof_division_to_id = Dict(
        (0, 1) => 1,
        (0, 2) => 3,
        (0, 3) => 9,
        (0, 4) => 7,
        (1, 1) => 4,
        (1, 2) => 6,
        (1, 3) => 2,
        (1, 4) => 8,
        (2, 1) => 5,
    )

    @test Topology.get_manifold_dim(Topology.QUAD) == 2
    @test Topology.get_incidence_relations_dim(Topology.QUAD) == 3
    @test Topology.get_num_patch_vertices(Topology.QUAD) == 4
    @test Topology.get_meshcore_patch(Topology.QUAD) == MeshCore.Q4
    @test @expected expected_ir k -> Topology.QUAD[k[1], k[2]]
    @test size(Topology.QUAD) == (4, 4, 1)
    @test @expected expected_size_per_dim k -> size(Topology.QUAD, k)
    @test @expected expected_id_to_position k ->
        Topology.id_to_position(Topology.QUAD, k[1], k[2])
    @test @expected expected_position_to_id k -> Topology.position_to_id(Topology.QUAD, k)
    @test @expected expected_dof_division_to_id k ->
        Topology.id_to_dof_division(Topology.QUAD, k[1], k[2])
    @test Topology.get_skeleton_patch(Topology.QUAD) == Topology.LINE
end

@testset "Hex" begin
    expected_ir = Dict(
        (1, 1) => Vector{Vector{Int}}(),
        (1, 2) => [
            [3, 7, 11],
            [3, 8, 12],
            [4, 8, 9],
            [4, 7, 10],
            [2, 6, 11],
            [2, 5, 12],
            [1, 5, 9],
            [1, 6, 10],
        ],
        (1, 3) => [
            [1, 3, 5],
            [2, 3, 5],
            [2, 5, 4],
            [1, 4, 5],
            [1, 3, 6],
            [2, 3, 6],
            [2, 4, 6],
            [1, 4, 6],
        ],
        (1, 4) => [[1], [1], [1], [1], [1], [1], [1], [1]],
        (2, 1) => [
            [8, 7],
            [5, 6],
            [1, 2],
            [4, 3],
            [6, 7],
            [5, 8],
            [1, 4],
            [2, 3],
            [3, 7],
            [4, 8],
            [1, 5],
            [2, 6],
        ],
        (2, 2) => Vector{Vector{Int}}(),
        (2, 3) => [
            [4, 6],
            [3, 6],
            [3, 5],
            [4, 5],
            [2, 6],
            [1, 6],
            [1, 5],
            [2, 5],
            [2, 4],
            [1, 4],
            [1, 3],
            [2, 3],
        ],
        (2, 4) => [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
        (3, 1) => [
            [1, 5, 8, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [4, 8, 7, 3],
            [1, 4, 3, 2],
            [5, 6, 7, 8],
        ],
        (3, 2) => [
            [11, 10, 7, 6],
            [12, 9, 8, 5],
            [11, 12, 3, 2],
            [10, 9, 4, 1],
            [7, 8, 3, 4],
            [6, 5, 2, 1],
        ],
        (3, 3) => Vector{Vector{Int}}(),
        (3, 4) => [[1], [1], [1], [1], [1], [1]],
        (4, 1) => [[1, 2, 3, 4, 5, 6, 7, 8]],
        (4, 2) => [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        (4, 3) => [[1, 2, 3, 4, 5, 6]],
        (4, 4) => Vector{Vector{Int}}(),
    )
    expected_size_per_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
    expected_id_to_position = Dict(
        (0, 1) => (-1, -1, -1),
        (0, 2) => (1, -1, -1),
        (0, 3) => (1, 1, -1),
        (0, 4) => (-1, 1, -1),
        (0, 5) => (-1, -1, 1),
        (0, 6) => (1, -1, 1),
        (0, 7) => (1, 1, 1),
        (0, 8) => (-1, 1, 1),
        (1, 1) => (0, 1, 1),
        (1, 2) => (0, -1, 1),
        (1, 3) => (0, -1, -1),
        (1, 4) => (0, 1, -1),
        (1, 5) => (1, 0, 1),
        (1, 6) => (-1, 0, 1),
        (1, 7) => (-1, 0, -1),
        (1, 8) => (1, 0, -1),
        (1, 9) => (1, 1, 0),
        (1, 10) => (-1, 1, 0),
        (1, 11) => (-1, -1, 0),
        (1, 12) => (1, -1, 0),
        (2, 1) => (-1, 0, 0),
        (2, 2) => (1, 0, 0),
        (2, 3) => (0, -1, 0),
        (2, 4) => (0, 1, 0),
        (2, 5) => (0, 0, -1),
        (2, 6) => (0, 0, 1),
        (3, 1) => (0, 0, 0),
    )
    expected_position_to_id = Dict(
        (-1, -1, -1) => 1,
        (1, -1, -1) => 2,
        (1, 1, -1) => 3,
        (-1, 1, -1) => 4,
        (-1, -1, 1) => 5,
        (1, -1, 1) => 6,
        (1, 1, 1) => 7,
        (-1, 1, 1) => 8,
        (0, 1, 1) => 1,
        (0, -1, 1) => 2,
        (0, -1, -1) => 3,
        (0, 1, -1) => 4,
        (1, 0, 1) => 5,
        (-1, 0, 1) => 6,
        (-1, 0, -1) => 7,
        (1, 0, -1) => 8,
        (1, 1, 0) => 9,
        (-1, 1, 0) => 10,
        (-1, -1, 0) => 11,
        (1, -1, 0) => 12,
        (-1, 0, 0) => 1,
        (1, 0, 0) => 2,
        (0, -1, 0) => 3,
        (0, 1, 0) => 4,
        (0, 0, -1) => 5,
        (0, 0, 1) => 6,
        (0, 0, 0) => 1,
    )
    expected_dof_division_to_id = Dict(
        (0, 1) => 1,
        (0, 2) => 3,
        (0, 3) => 9,
        (0, 4) => 7,
        (0, 5) => 19,
        (0, 6) => 21,
        (0, 7) => 27,
        (0, 8) => 25,
        (1, 1) => 26,
        (1, 2) => 20,
        (1, 3) => 2,
        (1, 4) => 8,
        (1, 5) => 24,
        (1, 6) => 22,
        (1, 7) => 4,
        (1, 8) => 6,
        (1, 9) => 18,
        (1, 10) => 16,
        (1, 11) => 10,
        (1, 12) => 12,
        (2, 1) => 13,
        (2, 2) => 15,
        (2, 3) => 11,
        (2, 4) => 17,
        (2, 5) => 5,
        (2, 6) => 23,
        (3, 1) => 14,
    )

    @test Topology.get_manifold_dim(Topology.HEX) == 3
    @test Topology.get_incidence_relations_dim(Topology.HEX) == 4
    @test Topology.get_num_patch_vertices(Topology.HEX) == 8
    @test Topology.get_meshcore_patch(Topology.HEX) == MeshCore.H8
    @test @expected expected_ir k -> Topology.HEX[k[1], k[2]]
    @test size(Topology.HEX) == (8, 12, 6, 1)
    @test @expected expected_size_per_dim k -> size(Topology.HEX, k)
    @test @expected expected_id_to_position k ->
        Topology.id_to_position(Topology.HEX, k[1], k[2])
    @test @expected expected_position_to_id k -> Topology.position_to_id(Topology.HEX, k)
    @test @expected expected_dof_division_to_id k ->
        Topology.id_to_dof_division(Topology.HEX, k[1], k[2])
    @test Topology.get_skeleton_patch(Topology.HEX) == Topology.QUAD
end

end
