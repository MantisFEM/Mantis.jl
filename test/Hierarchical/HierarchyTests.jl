module HierarchyTests

using Mantis
using Mantis.Hierarchical
using Test

struct NewObject
    n::Int
end

function NewObject(::Val{n}) where {n}
    return NewObject(n)
end

TensorProducts.get_num_objects(obj::NewObject) = obj.n

@testset "Construction and getters" verbose = true begin
    p0 = NewObject(2)
    p1 = NewObject(3)
    p2 = NewObject(4)

    # Both scalings must share one concrete type, so their relations are built from the same
    # functions (each function has its own type), parametrized by the level.
    pc(level) = i -> if level == 1
        i == 1 ? [1, 2] : [2, 3]
    else
        [i, i + 1]
    end
    cp(level) = i -> if level == 1
        i == 1 ? [1] : (i == 2 ? [1, 2] : [2])
    else
        [div(i + 1, 2)]
    end
    relations(level) = Relations(
        RelationExplicit{Hierarchical.PC}(pc(level)),
        RelationExplicit{Hierarchical.CP}(cp(level)),
    )
    s1 = Hierarchical.Scaling(p0, p1, relations(1))
    s2 = Scaling(p1, p2, relations(2))

    active = ActiveInfo([[1, 2], [2, 3], Int[]])

    hierarchy = Hierarchy(active, [s1, s2])

    @test get_scalings(hierarchy) == [s1, s2]
    @test get_scaling(hierarchy, 1) === s1
    @test get_scaling(hierarchy, 2) === s2
    @test get_active_info(hierarchy) === active
end

@testset "Mismatched parent/child objects" verbose = true begin
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(_ -> ()),
        RelationExplicit{Hierarchical.CP}(_ -> ()),
    )
    s1 = Scaling(NewObject(2), NewObject(3), rel)
    # Parent does not match child of s1. Note that `NewObject` is immutable, so `===`
    # compares by value.
    s2 = Hierarchical.Scaling(NewObject(5), NewObject(4), rel)

    active = ActiveInfo([[1], [1], [1]])

    @test_throws ArgumentError Hierarchy(active, [s1, s2])
end

@testset "Non-concrete scalings" verbose = true begin
    p0 = NewObject(1)
    p1 = NewObject(2)
    p2 = NewObject(4)
    # Same relations, but built from different functions, so the scalings have different
    # concrete types.
    rel_0 = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )
    rel_1 = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )
    scaling0 = Scaling(p0, p1, rel_0)
    scaling1 = Scaling(p1, p2, rel_1)
    active = ActiveInfo([[1], Int[], Int[]])

    @test_throws ArgumentError Hierarchy(active, [scaling0, scaling1])
    @test_throws ArgumentError TreeHierarchy(active, [scaling0, scaling1])

    # Appending a scaling of a different type when refining is also rejected.
    hierarchy = Hierarchy(ActiveInfo([[1], Int[]]), [scaling0])
    @test_throws ArgumentError Hierarchical.refine(hierarchy, 2, [4], scaling1)
end

@testset "Incorrect number of levels" verbose = true begin
    s = Hierarchical.Scaling(
        NewObject(2),
        NewObject(3),
        Relations(
            RelationExplicit{Hierarchical.PC}(_ -> ()),
            RelationExplicit{Hierarchical.CP}(_ -> ()),
        ),
    )

    active = ActiveInfo([[1]])
    @test_throws ArgumentError Hierarchy(active, [s])

    active = ActiveInfo([[1], [2], [3]])
    @test_throws ArgumentError Hierarchy(active, [s])
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
    @test_throws ArgumentError TreeHierarchy(active, [scaling0, scaling1])
    # (1,1) and (2, 2) both active
    active = ActiveInfo([[1], [2], Int[]])
    @test_throws ArgumentError TreeHierarchy(active, [scaling0, scaling1])
    # (1,1) and (3, 4), (3, 3) active
    active = ActiveInfo([[1], Int[], [3, 4]])
    @test_throws ArgumentError TreeHierarchy(active, [scaling0, scaling1])
    # (2, 3), (2, 4) and (3, 5) active
    active = ActiveInfo([[1], [3, 4], [5]])
    @test_throws ArgumentError TreeHierarchy(active, [scaling0, scaling1])

    # Nested domains
    active = ActiveInfo([[1], [3, 4], [9, 10, 11, 12]])
    hier = TreeHierarchy(active, [scaling0, scaling1])
    @test map(l -> sort(collect(l)), get_tree_ids(hier)) ==
        [[1, 2, 3], [3, 4, 5, 6], [9, 10, 11, 12]]

    active = ActiveInfo([[1], Int[], [5, 6, 7, 8, 9, 10, 11, 12]])
    hier = TreeHierarchy(active, [scaling0, scaling1])
    @test map(l -> sort(collect(l)), get_tree_ids(hier)) ==
        [[1, 2, 3], [3, 4, 5, 6], [5, 6, 7, 8, 9, 10, 11, 12]]
end

@testset "Sets" verbose = true begin
    p0 = NewObject(1)
    scaling = Scaling(p0)
    active = ActiveInfo([[1], Int[]])
    hierarchy = Hierarchy(active, [scaling])

    @test get_sets(hierarchy) == [NewObject(1)]

    p1 = NewObject(2)
    p2 = NewObject(4)
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(i -> [(i - 1) * 2 + 1, i * 2]),
        RelationExplicit{Hierarchical.CP}(i -> [div(i + 1, 2)]),
    )

    scaling0 = Scaling(p0, p1, rel)
    scaling1 = Scaling(p1, p2, rel)
    active = ActiveInfo([[1], [3, 4], [9, 10, 11, 12]])
    hierarchy = Hierarchy(active, [scaling0, scaling1])

    @test get_sets(hierarchy) == [NewObject(1), NewObject(2), NewObject(4)]

    # Sets of different types make the scalings have different types.
    struct AnotherObject
        n::Int
    end

    a1 = AnotherObject(2)
    scaling0 = Scaling(p0, a1, rel)
    scaling1 = Scaling(a1, p2, rel)

    @test_throws ArgumentError Hierarchy(active, [scaling0, scaling1])
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
    active = ActiveInfo([Int[], [1, 2]])
    hierarchy = Hierarchy(active, [scaling0])
    refined = @inferred Hierarchical.refine(hierarchy, 2, [2], scaling1)
    # Adding a level does not change the type, nor the original hierarchy.
    @test typeof(refined) == typeof(hierarchy)
    @test get_scalings(hierarchy) == [scaling0]
    @test get_num_levels(hierarchy) == 2
    hierarchy = refined
    @test get_num_levels(hierarchy) == 3
    @test get_scalings(hierarchy) == [scaling0, scaling1]
    @test get_sets(hierarchy) == [p0, p1, p2]
    @test get_level_ids(get_active_info(hierarchy)) == [Int[], [1], [3, 4]]
    # Reset hierarchy
    active = ActiveInfo([[1], Int[]])
    hierarchy = Hierarchy(active, [scaling0])

    # With Scaling

    # Nothing is refined.
    hierarchy = @inferred update(hierarchy, 1, Int[], Int[], scaling0)
    @test get_num_levels(hierarchy) == 2
    @test get_scalings(hierarchy) == [scaling0]
    @test get_sets(hierarchy) == [p0, p1]
    @test get_level_ids(get_active_info(hierarchy)) == [[1], Int[]]

    # Refined, but no new level.
    hierarchy = @inferred update(hierarchy, 1, [1], [1, 2], scaling0)
    @test get_num_levels(hierarchy) == 2
    @test get_scalings(hierarchy) == [scaling0]
    @test get_sets(hierarchy) == [p0, p1]
    @test get_level_ids(get_active_info(hierarchy)) == [Int[], [1, 2]]

    # Refined, with new level.
    hierarchy = @inferred update(hierarchy, 2, [2], [3, 4], scaling1)
    @test get_num_levels(hierarchy) == 3
    @test get_scalings(hierarchy) == [scaling0, scaling1]
    @test get_sets(hierarchy) == [p0, p1, p2]
    @test get_level_ids(get_active_info(hierarchy)) == [Int[], [1], [3, 4]]
end

end
