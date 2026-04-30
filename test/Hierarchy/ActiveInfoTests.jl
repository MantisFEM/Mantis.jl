module ActiveInfoTests

using Mantis
using Test
using InteractiveUtils

############################################################################################
#                                          Setup                                           #
############################################################################################

# 1D three-level hierarchy
# L0 = |-------|-------|-------| 03 elements
# L1 = |---|---|---|---|---|---| 06 elements
# L2 = |-|-|-|-|-|-|-|-|-|-|-|-| 12 elements
# HM = |-------|-|-|---|-------|
level_ids = [[1, 3], [4], [5, 6]]
active_info = Hierarchy.ActiveInfo(level_ids)

@testset "Getters" verbose = true begin
    @test Hierarchy.get_level_ids(active_info) == level_ids
    @test Hierarchy.get_level_ids(active_info, 1) == [1, 3]
    @test Hierarchy.get_level_ids(active_info, 2) == [4]
    @test Hierarchy.get_level_ids(active_info, 3) == [5, 6]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 3, 5]
    @test Hierarchy.get_level_cum_num_ids(active_info, 1) == 2
    @test Hierarchy.get_level_cum_num_ids(active_info, 2) == 3
    @test Hierarchy.get_level_cum_num_ids(active_info, 3) == 5
    @test Hierarchy.get_level_num_ids(active_info, 0) == 0
    @test Hierarchy.get_level_num_ids(active_info, 1) == 2
    @test Hierarchy.get_level_num_ids(active_info, 2) == 1
    @test Hierarchy.get_level_num_ids(active_info, 3) == 2
    @test Hierarchy.get_num_levels(active_info) == 3
    @test Hierarchy.get_num_objects(active_info) == 5
    @test Hierarchy.get_level(active_info, 1) == 1
    @test Hierarchy.get_level(active_info, 2) == 1
    @test Hierarchy.get_level(active_info, 3) == 2
    @test Hierarchy.get_level(active_info, 4) == 3
    @test Hierarchy.get_level(active_info, 5) == 3
end

@testset "Conversions" verbose = true begin
    @test Hierarchy.convert_to_level_id(active_info, 1) == 1
    @test Hierarchy.convert_to_level_id(active_info, 2) == 3
    @test Hierarchy.convert_to_level_id(active_info, 3) == 4
    @test Hierarchy.convert_to_level_id(active_info, 4) == 5
    @test Hierarchy.convert_to_level_id(active_info, 5) == 6
    @test Hierarchy.convert_to_level_and_level_id(active_info, 1) == (1, 1)
    @test Hierarchy.convert_to_level_and_level_id(active_info, 2) == (1, 3)
    @test Hierarchy.convert_to_level_and_level_id(active_info, 3) == (2, 4)
    @test Hierarchy.convert_to_level_and_level_id(active_info, 4) == (3, 5)
    @test Hierarchy.convert_to_level_and_level_id(active_info, 5) == (3, 6)
    @test Hierarchy.convert_to_hier_id(active_info, 1, 1) == 1
    @test Hierarchy.convert_to_hier_id(active_info, 1, 3) == 2
    @test Hierarchy.convert_to_hier_id(active_info, 2, 4) == 3
    @test Hierarchy.convert_to_hier_id(active_info, 3, 5) == 4
    @test Hierarchy.convert_to_hier_id(active_info, 3, 6) == 5
    @test_throws MethodError Hierarchy.convert_to_hier_id(active_info, 1, 1337)
end

@testset "Field Changes" verbose = true begin
    # Update
    Hierarchy.update!(active_info, 2, [4], Int[])
    @test Hierarchy.get_level_ids(active_info) == [[1, 3], Int[], [5, 6]]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 2, 4]
    @test Hierarchy.get_num_levels(active_info) == 3
    Hierarchy.update!(active_info, 1, Int[], [4])
    @test Hierarchy.get_level_ids(active_info) == [[1, 3], [4], [5, 6]]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 3, 5]
    @test Hierarchy.get_num_levels(active_info) == 3
    Hierarchy.update!(active_info, 3, [6], [26, 62])
    @test Hierarchy.get_level_ids(active_info) == [[1, 3], [4], [5], [26, 62]]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 3, 4, 6]
    @test Hierarchy.get_num_levels(active_info) == 4
    # Add level
    Hierarchy.add_level!(active_info)
    @test Hierarchy.get_level_ids(active_info) == [[1, 3], [4], [5], [26, 62], Int[]]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 3, 4, 6, 6]
    @test Hierarchy.get_num_levels(active_info) == 5
    Hierarchy.add_level!(active_info)
    @test Hierarchy.get_level_ids(active_info) == [[1, 3], [4], [5], [26, 62], Int[], Int[]]
    @test Hierarchy.get_level_cum_num_ids(active_info) == [0, 2, 3, 4, 6, 6, 6]
    @test Hierarchy.get_num_levels(active_info) == 6
end

end
