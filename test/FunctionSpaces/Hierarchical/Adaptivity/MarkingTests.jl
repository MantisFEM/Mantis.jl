module MarkingTests

using Mantis
using Mantis.Hierarchical
using Test
include("../../../TestHelpers.jl")

const MarkingStrategy = FunctionSpaces.MarkingStrategy
const IdentityMarking = FunctionSpaces.IdentityMarking
const MaximumMarking = FunctionSpaces.MaximumMarking
const SupportMarking = FunctionSpaces.SupportMarking
const DorflerMarking = FunctionSpaces.DorflerMarking
const QBoxMarking = FunctionSpaces.QBoxMarking

ref_space_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
ref_geo_h2(s) = Geometry.refinement_uniform(s, 2)
scal_space_h2 = FunctionSpaces.scaling_matrix_uniform
const SelectionStandard = FunctionSpaces.SelectionStandard
const HB = FunctionSpaces.HB

############################################################################################
#                                       Test Helpers                                       #
############################################################################################

# Squared errors are [9.0, 1.0, 4.0, 0.25], so the total squared error is 14.25.
const errors = [3.0, 1.0, 2.0, 0.5]

struct DummyMarking <: MarkingStrategy end

############################################################################################
#                                          Tests                                           #
############################################################################################

@testset "MarkingStrategy" verbose = true begin
    # Strategies have to implement marking for a single object.
    @test_throws MethodError DummyMarking()(1)
    # Selecting no objects marks no objects.
    @test isempty(IdentityMarking()(Int[]))
end

@testset "IdentityMarking" verbose = true begin
    marking = IdentityMarking()
    @test @expected Dict(1 => [1], 2 => [2], 3 => [3], 4 => [4]) marking
    @test marking([1, 3]) == [1, 3]
end

@testset "MaximumMarking" verbose = true begin
    @testset "Cutoff" verbose = true begin
        # Marks errors larger or equal than 1.5.
        marking = MaximumMarking(errors, 0.5)
        @test @expected Dict(1 => [1], 2 => Int[], 3 => [3], 4 => Int[]) marking
        @test marking(1:4) == [1, 3]

        # Only the maximum error is marked.
        @test MaximumMarking(errors, 1.0)(1:4) == [1]
        # All errors are marked.
        @test MaximumMarking(errors, 0.0)(1:4) == 1:4
    end

    @testset "Invalid Arguments" verbose = true begin
        @test_throws ArgumentError MaximumMarking([1.0, -1.0], 0.5)
        @test_throws ArgumentError MaximumMarking(errors, -0.1)
        @test_throws ArgumentError MaximumMarking(errors, 1.1)
    end
end

@testset "DorflerMarking" verbose = true begin
    @testset "Cutoff" verbose = true begin
        # The target is 0.8 * 14.25 = 11.4, so the two largest errors are needed.
        marking = DorflerMarking(errors, 0.8)
        @test @expected Dict(1 => [1], 2 => Int[], 3 => [3], 4 => Int[]) marking
        @test marking(1:4) == [1, 3]

        # The full error requires all objects.
        @test DorflerMarking(errors, 1.0)(1:4) == 1:4
        # No error has to be reached, so nothing is marked.
        @test isempty(DorflerMarking(errors, 0.0)(1:4))
        # There is no error, so nothing is marked.
        @test isempty(DorflerMarking(zeros(4), 0.5)(1:4))
    end

    @testset "Ties" verbose = true begin
        # Two objects suffice, but all objects tied at the threshold are marked.
        @test DorflerMarking(ones(4), 0.5)(1:4) == 1:4
    end

    @testset "Integer Errors" verbose = true begin
        # The target is 0.5 * 14 = 7, so the largest error suffices.
        @test DorflerMarking([3, 1, 2], 0.5)(1:3) == [1]
    end

    @testset "Alias" verbose = true begin
        @test FunctionSpaces.DörflerMarking === DorflerMarking
    end

    @testset "Invalid Arguments" verbose = true begin
        @test_throws ArgumentError DorflerMarking([1.0, -1.0], 0.5)
        @test_throws ArgumentError DorflerMarking(errors, -0.1)
        @test_throws ArgumentError DorflerMarking(errors, 1.1)
    end
end

@testset "SupportMarking" verbose = true begin
    #=
        Linear B-splines on 4 elements. Basis function `b` is supported on elements `b - 1`
        and `b`, so marking element `e` extends to elements `e - 1:e + 1`.

        Elements: |---1---|---2---|---3---|---4---|
    =#
    space = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))

    @testset "IdentityMarking" verbose = true begin
        marking = SupportMarking(space, errors, IdentityMarking())
        expected = Dict(1 => [1, 2], 2 => [1, 2, 3], 3 => [2, 3, 4], 4 => [3, 4])
        @test @expected expected marking issetequal
        @test issetequal(marking([1, 4]), 1:4)
    end

    @testset "MaximumMarking" verbose = true begin
        # Only element 1 is marked by the element marking.
        marking = SupportMarking(space, errors, MaximumMarking(errors, 1.0))
        expected = Dict(1 => [1, 2], 2 => Int[], 3 => Int[], 4 => Int[])
        @test @expected expected marking issetequal
        @test issetequal(marking(1:4), [1, 2])
    end
end

@testset "QBoxMarking" verbose = true begin
    #=
        Linear B-splines on 4 elements, with qboxes of 2 elements.

        Elements: |---1---|---2---|---3---|---4---|
        QBoxes:   |-------1-------|-------2-------|
    =#
    space_1d = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))

    @testset "1D" verbose = true begin
        marking = QBoxMarking(space_1d, (2,), IdentityMarking())
        expected = Dict(1 => [1, 2], 2 => [1, 2], 3 => [3, 4], 4 => [3, 4])
        @test @expected expected marking issetequal
        @test marking([1, 4]) == 1:4
    end

    @testset "2D" verbose = true begin
        #=
            Linear B-splines on 4x2 elements, with qboxes of 2x2 elements.

            Elements:            QBoxes:
            |---|---|---|---|    |-------|-------|
            | 5 | 6 | 7 | 8 |    |       |       |
            |---|---|---|---|    |   1   |   2   |
            | 1 | 2 | 3 | 4 |    |       |       |
            |---|---|---|---|    |-------|-------|
        =#
        space = FunctionSpaces.create_bspline_space(
            (0.0, 0.0), (1.0, 1.0), (4, 2), (1, 1), (0, 0)
        )
        marking = QBoxMarking(space, (2, 2), IdentityMarking())
        expected = Dict(
            1 => [1, 2, 5, 6],
            2 => [1, 2, 5, 6],
            5 => [1, 2, 5, 6],
            6 => [1, 2, 5, 6],
            3 => [3, 4, 7, 8],
            4 => [3, 4, 7, 8],
            7 => [3, 4, 7, 8],
            8 => [3, 4, 7, 8],
        )
        @test @expected expected marking issetequal

        # Heterogenous q-box size
        marking = QBoxMarking(space, (4, 1), IdentityMarking())
        @test marking(2) == [1, 2, 3, 4]
        @test marking(7) == [5, 6, 7, 8]
    end

    @testset "Tensor Product" verbose = true begin
        # The same 4x2 elements and qboxes as in the 2D case.
        space = FunctionSpaces.TensorProductSpace((
            FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,)),
            FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (1,), (0,)),
        ))
        marking = QBoxMarking(space, (2, 2), IdentityMarking())
        expected = Dict(
            1 => [1, 2, 5, 6],
            2 => [1, 2, 5, 6],
            5 => [1, 2, 5, 6],
            6 => [1, 2, 5, 6],
            3 => [3, 4, 7, 8],
            4 => [3, 4, 7, 8],
            7 => [3, 4, 7, 8],
            8 => [3, 4, 7, 8],
        )
        @test @expected expected marking issetequal

        # Heterogenous q-box size
        marking = QBoxMarking(space, (4, 1), IdentityMarking())
        @test marking(2) == [1, 2, 3, 4]
        @test marking(7) == [5, 6, 7, 8]

        # The number of elements is not a multiple of the qbox size.
        @test_throws ArgumentError QBoxMarking(space, (3, 2), IdentityMarking())
    end

    @testset "MaximumMarking" verbose = true begin
        # Only element 1 is marked by the element marking.
        marking = QBoxMarking(space_1d, (2,), MaximumMarking(errors, 1.0))
        expected = Dict(1 => [1, 2], 2 => Int[], 3 => Int[], 4 => Int[])
        @test @expected expected marking issetequal
        @test marking(1:4) == [1, 2]
    end

    @testset "Hierarchical" verbose = true begin
        # Level Geometries
        geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
        geo_l2 = Refinement(geo_l1, ref_geo_h2)()
        geo_l3 = Refinement(geo_l2, ref_geo_h2)()
        relations = Relations(
            Geometry.parent_to_children_uniform(2), Geometry.child_to_parents_uniform(2)
        )
        geo_scal_1 = Scaling(geo_l1, geo_l2, relations)
        geo_scal_2 = Scaling(geo_l2, geo_l3, relations)

        # Level FESpaces
        bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))
        bsp_l2 = Refinement(bsp_l1, ref_space_h2)()
        bsp_l3 = Refinement(bsp_l2, ref_space_h2)()
        bsp_scal_1 = MatrixScaling(bsp_l1, bsp_l2, scal_space_h2)
        bsp_scal_2 = MatrixScaling(bsp_l2, bsp_l3, scal_space_h2)
        scalings = (bsp_scal_1, bsp_scal_2)

        @testset "QBox Refinement" verbose = true begin
            #=
                Only whole qboxes of 2 elements are refined. Hierarchical element ids are
                shown in the final geometry.

                x inactive
                - active

                Level 1: |-------|-------|xxxxxxx|xxxxxxx|
                Level 2: |xxx|xxx|xxx|xxx|---|---|xxx|xxx|
                Level 3: |x|x|x|x|x|x|x|x|x|x|x|x|-|-|-|-|

                Final G: |---1---|---2---|-3-|-4-|5|6|7|8|
            =#
            active = ActiveInfo([[1, 2], [5, 6], [13, 14, 15, 16]])
            hgeo = Geometry.HierarchicalGeometry(
                TreeHierarchy(active, geo_scal_1, geo_scal_2)
            )
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            marking = QBoxMarking(space, (2,), IdentityMarking())
            expected = Dict(
                1 => [1, 2],
                2 => [1, 2],
                3 => [3, 4],
                4 => [3, 4],
                5 => [5, 6],
                6 => [5, 6],
                7 => [7, 8],
                8 => [7, 8],
            )
            @test @expected expected marking issetequal

            # The number of elements is not a multiple of the qbox size on every level.
            @test_throws ArgumentError QBoxMarking(space, (3,), IdentityMarking())
        end

        @testset "Non-QBox Refinement" verbose = true begin
            #=
                Element 5 of level 2 is refined, but it is in the same qbox as the active
                element 6 of level 2 (hierarchical element 3).

                x inactive
                - active

                Level 1: |-------|-------|xxxxxxx|xxxxxxx|
                Level 2: |xxx|xxx|xxx|xxx|xxx|---|xxx|xxx|
                Level 3: |x|x|x|x|x|x|x|x|-|-|x|x|-|-|-|-|

                Final G: |---1---|---2---|4|5|-3-|6|7|8|9|
            =#
            active = ActiveInfo([[1, 2], [6], [9, 10, 13, 14, 15, 16]])
            hgeo = Geometry.HierarchicalGeometry(
                TreeHierarchy(active, geo_scal_1, geo_scal_2)
            )
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            marking = QBoxMarking(space, (2,), IdentityMarking())
            # Elements with complete qboxes are still marked.
            @test issetequal(marking(1), [1, 2])
            @test issetequal(marking(4), [4, 5])
            # The qbox of element 3 contains an inactive element.
            @test_throws ArgumentError marking(3)
        end
    end

    @testset "Invalid Arguments" verbose = true begin
        # The number of elements is not a multiple of the qbox size.
        @test_throws ArgumentError QBoxMarking(space_1d, (3,), IdentityMarking())
        @test_throws ArgumentError QBoxMarking(space_1d, (0,), IdentityMarking())
        # The qbox size does not match the dimension of the geometry.
        @test_throws ArgumentError QBoxMarking(space_1d, (2, 2), IdentityMarking())
    end
end

end
