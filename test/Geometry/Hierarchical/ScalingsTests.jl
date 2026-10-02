module ScalingsTests

using Mantis
using Mantis.Hierarchical
using Test

@testset "Uniform Relations" verbose = true begin
    @testset "1D" verbose = true begin
        @test_throws ArgumentError Geometry.parent_to_children_uniform(-1)
        @test_throws ArgumentError Geometry.child_to_parents_uniform(0)
        rel1 = Geometry.parent_to_children_uniform(1)
        @test rel1(5) == 5:5
        rel2 = Geometry.parent_to_children_uniform(2)
        @test rel2(5) == 9:10
        rel3 = Geometry.parent_to_children_uniform(3)
        @test rel3(5) == 13:15

        rel_pc = Geometry.parent_to_children_uniform(3)
        rel_cp = Geometry.child_to_parents_uniform(3)
        for p in 1:5
            children = rel_pc(p)
            for c in children
                @test rel_cp(c)[1] == p
            end
        end
    end

    @testset "2D" verbose = true begin
        cart_1d_1 = Geometry.create_cartesian_box((0.0,), (1.0,), (1,))
        cart_1d_2 = Geometry.create_cartesian_box((0.0,), (1.0,), (2,))
        rel_1d = Relations(
            Geometry.parent_to_children_uniform(2), Geometry.child_to_parents_uniform(2)
        )
        tp_1 = Geometry.TensorProductGeometry((cart_1d_1, cart_1d_1))
        tp_2 = Geometry.TensorProductGeometry((cart_1d_2, cart_1d_2))

        scal_2d = Scaling(tp_1, tp_2, (rel_1d, rel_1d))
        @test collect(get_children(scal_2d, 1)) == [1, 2, 3, 4]
        for c in 1:4
            @test collect(get_parents(scal_2d, c)) == [1]
        end

        # Child points to the wrong object in memory
        scal_1d = Scaling(cart_1d_1, cart_1d_2, rel_1d)
        scal_fail = Scaling(cart_1d_1, cart_1d_1, rel_1d)
        @test_throws ArgumentError Scaling(tp_1, tp_2, (scal_1d, scal_fail))

        # 2D Cartesian
        cart_2d_1 = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (2, 2))
        cart_2d_2 = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4))
        @test_throws ArgumentError Geometry.parent_to_children_uniform(cart_2d_1, (2, 0))
        @test_throws ArgumentError Geometry.child_to_parents_uniform(cart_2d_1, (-1, 1))
        rel_2d = Relations(
            Geometry.parent_to_children_uniform(cart_2d_1, (2, 2)),
            Geometry.child_to_parents_uniform(cart_2d_2, (2, 2)),
        )

        function children(expected, rel)
            for k in keys(expected)
                val = collect(get_children(rel, k))
                if val != expected[k]
                    println(stderr, "Element $(k): expected $(expected[k]), got $(val)")
                    return false
                end
            end

            return true
        end

        function parents(expected, rel)
            for k in keys(expected)
                val = collect(get_parents(rel, k))
                if val != expected[k]
                    println(stderr, "Element $(k): expected $(expected[k]), got $(val)")
                    return false
                end
            end

            return true
        end

        children_expected = Dict(
            1 => [1, 2, 5, 6],
            2 => [3, 4, 7, 8],
            3 => [9, 10, 13, 14],
            4 => [11, 12, 15, 16],
        )
        @test children(children_expected, rel_2d)

        parents_expected = Dict(
            1 => [1],
            2 => [1],
            3 => [2],
            4 => [2],
            5 => [1],
            6 => [1],
            7 => [2],
            8 => [2],
            9 => [3],
            10 => [3],
            11 => [4],
            12 => [4],
            13 => [3],
            14 => [3],
            15 => [4],
            16 => [4],
        )
        @test parents(parents_expected, rel_2d)
    end
end

end
