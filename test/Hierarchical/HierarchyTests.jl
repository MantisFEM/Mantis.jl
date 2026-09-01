module HierarchyTests

using Mantis
using Mantis.Hierarchical
using Test
using InteractiveUtils

struct NewObject{n}
    function NewObject(n::Int)
        return new{n}()
    end
    function NewObject(::Val{n}) where {n}
        return new{n}()
    end
end

TensorProducts.get_num_objects(obj::NewObject) = obj.n

@testset "Construction and getters" verbose = true begin
    p0 = NewObject(2)
    p1 = NewObject(3)
    p2 = NewObject(4)

    pc_1(i) = i == 1 ? [1, 2] : [2, 3]
    cp_1(i) =
        if i == 1
            [1]
        elseif i == 2
            [1, 2]
        else
            [2]
        end
    rel_1 = Relations(
        RelationExplicit{Hierarchical.PC}(pc_1), RelationExplicit{Hierarchical.CP}(cp_1)
    )
    s1 = Hierarchical.Scaling(p0, p1, rel_1)
    pc_2(i) = [i, i + 1]
    cp_2(i) = div(i+1, 2)
    rel_2 = Relations(
        RelationExplicit{Hierarchical.PC}(pc_2), RelationExplicit{Hierarchical.CP}(cp_2)
    )
    s2 = Scaling(p1, p2, rel_2)

    active = ActiveInfo([[1, 2], [2, 3], Int[]])

    hierarchy = Hierarchy(active, s1, s2)

    @test get_scalings(hierarchy) == (s1, s2)
    @test get_scaling(hierarchy, 1) === s1
    @test get_scaling(hierarchy, 2) === s2
    @test get_active_info(hierarchy) === active
end

@testset "Mismatched parent/child objects" verbose = true begin
    s1 = Scaling(
        NewObject(2),
        NewObject(3),
        Relations(RelationEmpty{Hierarchical.PC}(), RelationEmpty{Hierarchical.CP}()),
    )

    # Parent does not match child of s1 (must be the same object in memory)
    s2 = Hierarchical.Scaling(
        NewObject(3),
        NewObject(4),
        Relations(RelationEmpty{Hierarchical.PC}(), RelationEmpty{Hierarchical.CP}()),
    )

    active = ActiveInfo([[1], [1]])

    @test_throws ArgumentError Hierarchy(active, s1, s2)
end

@testset "Incorrect number of levels" verbose = true begin
    s = Hierarchical.Scaling(
        NewObject(2),
        NewObject(3),
        Relations(RelationEmpty{Hierarchical.PC}(), RelationEmpty{Hierarchical.CP}()),
    )

    active = ActiveInfo([[1]])
    @test_throws ArgumentError Hierarchy(active, s)

    active = ActiveInfo([[1], [2], [3]])
    @test_throws ArgumentError Hierarchy(active, s)
end

@testset "Nesting" verbose = true begin
    p0 = NewObject(1)
    p1 = NewObject(2)
    p2 = NewObject(4)

    rel = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )
    scaling0 = Scaling(p0, p1, rel)
    scaling1 = Scaling(p1, p2, rel)

    # (1,1) and (2, 1) both active
    active = ActiveInfo([[1], [1], Int[]])
    @test_throws ArgumentError NestedHierarchy(active, scaling0, scaling1)
    # (1,1) and (2, 2) both active
    active = ActiveInfo([[1], [2], Int[]])
    @test_throws ArgumentError NestedHierarchy(active, scaling0, scaling1)
    # (1,1) and (3, 4), (3, 3) active
    active = ActiveInfo([[1], Int[], [3, 4]])
    @test_throws ArgumentError NestedHierarchy(active, scaling0, scaling1)
    # (2, 3), (2, 4) and (3, 5) active
    active = ActiveInfo([[1], [3, 4], [5]])
    @test_throws ArgumentError NestedHierarchy(active, scaling0, scaling1)

    # Nested domains
    active = ActiveInfo([[1], [3, 4], [9, 10, 11, 12]])
    hier = NestedHierarchy(active, scaling0, scaling1)
    @test map(l -> sort(collect(l)), get_nested_ids(hier)) ==
        [[1, 2, 3], [3, 4, 5, 6], [9, 10, 11, 12]]

    active = ActiveInfo([[1], Int[], [5, 6, 7, 8, 9, 10, 11, 12]])
    hier = NestedHierarchy(active, scaling0, scaling1)
    @test map(l -> sort(collect(l)), get_nested_ids(hier)) ==
        [[1, 2, 3], [3, 4, 5, 6], [5, 6, 7, 8, 9, 10, 11, 12]]
end

@testset "Iteration" verbose = true begin
    p0 = NewObject(1)
    scaling = Scaling(p0)
    active = ActiveInfo([[1], Int[]])
    hierarchy = Hierarchy(active, scaling)
    for obj in hierarchy
        @test obj === p0
    end

    @test get_sets(hierarchy) == (NewObject(1),)

    p1 = NewObject(2)
    p2 = NewObject(4)
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )

    scaling0 = Scaling(p0, p1, rel)
    scaling1 = Scaling(p1, p2, rel)
    active = ActiveInfo([[1], [3, 4], [9, 10, 11, 12]])
    hierarchy = Hierarchy(active, scaling0, scaling1)
    for (l, obj) in enumerate(hierarchy)
        if l == 1
            @test obj === p0
        elseif l == 2
            @test obj === p1
        elseif l == 3
            @test obj === p2
        end
    end

    @test get_sets(hierarchy) == (NewObject(1), NewObject(2), NewObject(4))
end

@testset "Updates" verbose = true begin
    p0 = NewObject(1)
    p1 = NewObject(2)
    p2 = NewObject(4)
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )
    scaling0 = Scaling(p0, p1, rel)
    scaling1 = Scaling(p1, p2, rel)
    active = ActiveInfo([[1], Int[]])
    hierarchy = Hierarchy(active, scaling0)
    hierarchy = add_level!(hierarchy, scaling1)
    @test get_num_levels(hierarchy) == 3
    @test get_scalings(hierarchy) == (scaling0, scaling1)
    @test get_sets(hierarchy) == (p0, p1, p2)
    @test get_level_ids(get_active_info(hierarchy)) == [[1], Int[], Int[]]
    # Reset hierarchy
    active = ActiveInfo([[1], Int[]])
    hierarchy = Hierarchy(active, scaling0)

    # With Scaling

    # Nothing is refined.
    hierarchy = update!(hierarchy, 1, Int[], Int[], scaling0)
    @test get_num_levels(hierarchy) == 2
    @test get_scalings(hierarchy) == (scaling0,)
    @test get_sets(hierarchy) == (p0, p1)
    @test get_level_ids(get_active_info(hierarchy)) == [[1], Int[]]

    # Refined, but no new level.
    hierarchy = update!(hierarchy, 1, [1], [1, 2], scaling0)
    @test get_num_levels(hierarchy) == 2
    @test get_scalings(hierarchy) == (scaling0,)
    @test get_sets(hierarchy) == (p0, p1)
    @test get_level_ids(get_active_info(hierarchy)) == [Int[], [1, 2]]

    # Refined, with new level.
    hierarchy = update!(hierarchy, 2, [2], [3, 4], scaling1)
    @test get_num_levels(hierarchy) == 3
    @test get_scalings(hierarchy) == (scaling0, scaling1)
    @test get_sets(hierarchy) == (p0, p1, p2)
    @test get_level_ids(get_active_info(hierarchy)) == [Int[], [1], [3, 4]]
end

end
