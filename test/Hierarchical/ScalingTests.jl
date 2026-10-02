module ScalingTests

using Mantis
using Mantis.Hierarchical
using Test
using SparseArrays

struct NewObject
    n::Int
end

TensorProducts.get_num_objects(obj::NewObject) = obj.n

@testset "Relations" verbose = true begin
    S = sparse([1, 2, 2, 3], [1, 1, 2, 2], ones(Int, 4), 3, 2)
    relations = Relations(S)

    @test Hierarchical.get_parent_to_children(relations) isa Hierarchical.RelationExplicit
    @test Hierarchical.get_child_to_parents(relations) isa Hierarchical.RelationExplicit

    @test collect(get_children(relations, 1)) == [1, 2]
    @test collect(get_children(relations, 2)) == [2, 3]

    @test collect(get_parents(relations, 1)) == [1]
    @test collect(get_parents(relations, 2)) == [1, 2]
    @test collect(get_parents(relations, 3)) == [2]
end

@testset "Placeholder scaling" verbose = true begin
    parent = NewObject(2)
    scaling = Scaling(parent)

    @test get_parent(scaling) === parent
    @test get_child(scaling) === nothing

    @test Hierarchical.get_relations(scaling) isa Relations

    @test isempty(get_children(scaling, 1))
    @test isempty(get_parents(scaling, 1))

    @test isempty(get_children(Hierarchical.get_relations(scaling), 1))
    @test isempty(get_parents(Hierarchical.get_relations(scaling), 1))
end

@testset "Tensor-product scaling" verbose = true begin
    p1 = NewObject(2)
    c1 = NewObject(3)

    pc(i) = i == 1 ? [1, 2] : [2, 3]
    cp(i) =
        if i == 1
            [1]
        elseif i == 2
            [1, 2]
        else
            [2]
        end
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(pc), RelationExplicit{Hierarchical.CP}(cp)
    )
    s1 = Scaling(p1, c1, rel)
    parent = TensorProducts.TensorProduct((p1, p1))
    child = TensorProducts.TensorProduct((c1, c1))
    scaling = Scaling(parent, child, (rel, rel))

    @test get_parent(scaling) === parent
    @test get_child(scaling) === child
    @test Hierarchical.get_relations(scaling) isa Relations

    plin = TensorProducts.get_lin_ids(parent)
    clin = TensorProducts.get_lin_ids(child)

    p11 = plin[1, 1]

    @test sort(collect(get_children(scaling, p11))) ==
        sort([clin[1, 1], clin[1, 2], clin[2, 1], clin[2, 2]])

    p22 = plin[2, 2]

    @test sort(collect(get_children(scaling, p22))) ==
        sort([clin[2, 2], clin[2, 3], clin[3, 2], clin[3, 3]])

    c22 = clin[2, 2]

    @test sort(collect(get_parents(scaling, c22))) ==
        sort([plin[1, 1], plin[1, 2], plin[2, 1], plin[2, 2]])

    @test collect(get_parents(scaling, clin[1, 1])) == [plin[1, 1]]
    @test collect(get_parents(scaling, clin[3, 3])) == [plin[2, 2]]
end

@testset "Consistency" verbose=true begin
    p = NewObject(2)
    c = NewObject(3)

    pc(i) = i == 1 ? [1, 2] : [2, 3]
    cp(i) =
        if i == 1
            [1]
        elseif i == 2
            [1, 2]
        else
            [2]
        end
    rel = Relations(
        RelationExplicit{Hierarchical.PC}(pc), RelationExplicit{Hierarchical.CP}(cp)
    )

    scaling = Scaling(p, c, rel)

    for parent in 1:2
        for child in get_children(scaling, parent)
            @test parent in get_parents(scaling, child)
        end
    end

    for child in 1:3
        for parent in get_parents(scaling, child)
            @test child in get_children(scaling, parent)
        end
    end
end

end
