module ActiveInfoTests

using Mantis
using Mantis.Hierarchical
using Test

@testset "Constructor" verbose = true begin
    @test_throws ArgumentError ActiveInfo([[1, 2, 1], [2, 3]])
    @test_throws ArgumentError ActiveInfo([[1, 2, 3], [4, 5, 6, 5, 7]])
end

@testset "Getters" verbose = true begin
    active_info = ActiveInfo([[1, 3, 5], [2, 4]])

    @test get_level_ids(active_info) == [[1, 3, 5], [2, 4]]
    @test get_level_ids(active_info, 1) == [1, 3, 5]
    @test get_level_ids(active_info, 2) == [2, 4]

    @test Hierarchical.get_level_cum_num_ids(active_info) == [0, 3, 5]
    @test Hierarchical.get_level_cum_num_ids(active_info, 0) == 0
    @test Hierarchical.get_level_cum_num_ids(active_info, 1) == 3
    @test Hierarchical.get_level_cum_num_ids(active_info, 2) == 5

    @test Hierarchical.get_level_num_ids(active_info, 1) == 3
    @test Hierarchical.get_level_num_ids(active_info, 2) == 2

    @test get_num_levels(active_info) == 2
    @test get_num_objects(active_info) == 5

    @test get_level(active_info, 1) == 1
    @test get_level(active_info, 2) == 1
    @test get_level(active_info, 3) == 1
    @test get_level(active_info, 4) == 2
    @test get_level(active_info, 5) == 2

    @test get_level_set(active_info, 1) == Set([1, 3, 5])
    @test get_level_set(active_info, 2) == Set([2, 4])

    @test Hierarchical.get_level_lookup(active_info, 1) == Dict(1 => 1, 3 => 2, 5 => 3)
    @test Hierarchical.get_level_lookup(active_info, 2) == Dict(2 => 1, 4 => 2)
end

@testset "Conversions" verbose = true begin
    active_info = ActiveInfo([[1, 3, 5], [2, 4]])

    @test convert_to_level_id(active_info, 1) == 1
    @test convert_to_level_id(active_info, 2) == 3
    @test convert_to_level_id(active_info, 3) == 5
    @test convert_to_level_id(active_info, 4) == 2
    @test convert_to_level_id(active_info, 5) == 4

    @test_throws BoundsError get_level(active_info, -1)
    @test_throws BoundsError get_level(active_info, 42)
    @test_throws BoundsError convert_to_level_id(active_info, -1)
    @test_throws BoundsError convert_to_level_id(active_info, 42)

    @test_throws BoundsError convert_to_hier_id(active_info, -1, 3)
    @test_throws BoundsError convert_to_hier_id(active_info, 3, 3)
    @test_throws KeyError convert_to_hier_id(active_info, 1, 2)

    @test convert_to_level_and_level_id(active_info, 1) == (1, 1)
    @test convert_to_level_and_level_id(active_info, 2) == (1, 3)
    @test convert_to_level_and_level_id(active_info, 3) == (1, 5)
    @test convert_to_level_and_level_id(active_info, 4) == (2, 2)
    @test convert_to_level_and_level_id(active_info, 5) == (2, 4)

    @test convert_to_hier_id(active_info, 1, 1) == 1
    @test convert_to_hier_id(active_info, 1, 3) == 2
    @test convert_to_hier_id(active_info, 1, 5) == 3
    @test convert_to_hier_id(active_info, 2, 2) == 4
    @test convert_to_hier_id(active_info, 2, 4) == 5
end

@testset "Update" verbose = true begin
    active_info = ActiveInfo([[1, 3, 5], [2, 4]])

    update!(active_info, 1, [3], [6, 7])

    @test get_level_ids(active_info) == [[1, 5], [2, 4, 6, 7]]
    @test get_level_sets(active_info) == [Set([1, 5]), Set([2, 4, 6, 7])]
    @test Hierarchical.get_level_lookup(active_info, 1) == Dict(1 => 1, 5 => 2)
    @test Hierarchical.get_level_lookup(active_info, 2) ==
        Dict(2 => 1, 4 => 2, 6 => 3, 7 => 4)
    @test Hierarchical.get_level_cum_num_ids(active_info) == [0, 2, 6]

    @test convert_to_level_and_level_id(active_info, 1) == (1, 1)
    @test convert_to_level_and_level_id(active_info, 2) == (1, 5)
    @test convert_to_level_and_level_id(active_info, 3) == (2, 2)
    @test convert_to_level_and_level_id(active_info, 4) == (2, 4)
    @test convert_to_level_and_level_id(active_info, 5) == (2, 6)
    @test convert_to_level_and_level_id(active_info, 6) == (2, 7)

    @test convert_to_hier_id(active_info, 1, 1) == 1
    @test convert_to_hier_id(active_info, 1, 5) == 2
    @test convert_to_hier_id(active_info, 2, 2) == 3
    @test convert_to_hier_id(active_info, 2, 4) == 4
    @test convert_to_hier_id(active_info, 2, 6) == 5
    @test convert_to_hier_id(active_info, 2, 7) == 6

    update!(active_info, 2, [2], [8, 9])

    @test get_num_levels(active_info) == 3
    @test get_level_ids(active_info) == [[1, 5], [4, 6, 7], [8, 9]]
    @test get_level_ids(active_info, 1) == [1, 5]
    @test get_level_ids(active_info, 2) == [4, 6, 7]
    @test get_level_ids(active_info, 3) == [8, 9]
    @test Hierarchical.get_level_cum_num_ids(active_info) == [0, 2, 5, 7]
    @test Hierarchical.get_level_lookup(active_info, 3) == Dict(8 => 1, 9 => 2)
    @test get_level_set(active_info, 3) == Set([8, 9])

    @test convert_to_hier_id(active_info, 2, 7) == 5
    @test convert_to_level_and_level_id(active_info, 7) == (3, 9)
end

end
