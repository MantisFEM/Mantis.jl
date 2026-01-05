module GeometryInferenceTests

import Pkg

using Mantis

using Test

# We need JET for the tests in this file, but JET does not always work for every julia
# version. These tests were made on 1.12, so we start there.
@static if Base.VERSION >= v"1.12"
    using JET
end

@static if Base.VERSION >= v"1.12"
    # We can only run the tests if JET was loaded.
    if any(x -> x.name == "JET", values(Pkg.dependencies()))
        # Test spaces
        # Reduction test, single-patch, single element, 1D, Cartesian, degree 1 BSplines.
        geometry1 = Geometry.CartesianGeometry(([-1, 1],))
        B1 = FunctionSpaces.BSplineSpace(
            geometry1, geometry1, FunctionSpaces.Bernstein(1), [-1, -1], 1, 1
        )
        # Single-patch, multi-element, 1D, Cartesian, degree 6 maximally smooth BSplines.
        geometry1multi = Geometry.CartesianGeometry((LinRange(-0.34, 1.56, 26),))
        B1multi = FunctionSpaces.BSplineSpace(geometry1multi, 6, 5)
        # # Multi-patch input. 100 patches, 1D. Cartesian.
        # geometryMP100 = Geometry.CartesianGeometry(
        #     ntuple(100) do i
        #         return (LinRange((i - 1) * 1.0, i * 1.0, i + 1),)
        #     end,
        # )
        # # Single-patch, 4D.
        # geometry1p4D = Geometry.CartesianGeometry((
        #     LinRange(0.5, 2.5, 5),
        #     LinRange(-0.75, 0.75, 3),
        #     LinRange(1.5, 2.5, 4),
        #     LinRange(10.5, 20.5, 6),
        # ))
        # # 2D, two-patch, heterogeneous input.
        # geometryMP = Geometry.CartesianGeometry((
        #     (LinRange(0.5, 2.5, 5), [-0.75, 0.1, 0.75]),
        #     (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 6)),
        # ))

        const spaces = (
            B1,
            B1multi,
            # geometryMP100,
            # geometry1p4D,
            # geometryMP,
        )

        const xi_1D = Points.CartesianPoints(([0.0, 1.0],))
        # const xi_2D = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
        # const xi_4D = Points.CartesianPoints((
        #     [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        # ))

        const element_id = 1
        const component_id = 1
        const nderivatives = 2

        foreach(spaces) do space
            # Note that JET only uses the types of the inputs, so which numbers we pick
            # here is irrelevant.

            # Methods on the type.
            @test_opt FunctionSpaces.get_manifold_dim(space)
            @test_opt FunctionSpaces.get_num_components(space)
            @test_opt FunctionSpaces.get_num_patches(space)
            # Methods with have a general fallback.
            @test_opt FunctionSpaces.get_component_spaces(space)
            @test_opt FunctionSpaces.get_extraction_operator(space)
            @test_opt FunctionSpaces.get_extraction(space, element_id, component_id)
            @test_opt FunctionSpaces.get_extraction_coefficients(
                space, element_id, component_id
            )
            @test_opt FunctionSpaces.get_basis_indices(space, element_id)
            @test_opt FunctionSpaces.get_basis_permutation(space, element_id, component_id)
            @test_opt FunctionSpaces.get_num_basis(space)
            @test_opt FunctionSpaces.get_num_basis(space, element_id)
            @test_opt FunctionSpaces.get_dof_partition(space)
            @test_opt FunctionSpaces.get_max_local_dim(space)
            @test_opt FunctionSpaces.get_geometry(space)
            @test_opt FunctionSpaces.get_parametric_geometry(space)
            # Methods specific to some spaces.
            if typeof(space) <: FunctionSpaces.BSplineSpace
                @test_opt FunctionSpaces.get_polynomials(space)
                @test_opt FunctionSpaces.get_polynomial_degree(space)
                @test_opt FunctionSpaces.get_multiplicity_vector(space)
                @test_opt FunctionSpaces.get_support(space, element_id)
            end

            if FunctionSpaces.get_manifold_dim(space) == 1
                @test_opt FunctionSpaces.get_local_basis(
                    space, element_id, xi_1D, nderivatives
                )
                @test_opt FunctionSpaces.evaluate(space, element_id, xi_1D, nderivatives)
                @test_opt FunctionSpaces.evaluate(
                    space,
                    element_id,
                    xi_1D,
                    nderivatives,
                    ones(FunctionSpaces.get_num_basis(space)),
                )

                # elseif Geometry.get_manifold_dim(geo) == 2
                #     @test_opt Geometry.evaluate(geo, 5, xi_2D)
                #     @test_opt Geometry.jacobian(geo, 5, xi_2D)
                #     @test_opt Geometry.hessian(geo, 5, xi_2D)
                #     @test_opt Geometry.inv_metric(geo, 5, xi_2D)
                #     @test_opt Geometry.metric_derivatives(geo, 5, xi_2D)
                # elseif Geometry.get_manifold_dim(geo) == 3
                #     @test_opt Geometry.evaluate(geo, 5, xi_3D)
                #     @test_opt Geometry.jacobian(geo, 5, xi_3D)
                #     @test_opt Geometry.hessian(geo, 5, xi_3D)
                #     @test_opt Geometry.inv_metric(geo, 5, xi_3D)
                #     @test_opt Geometry.metric_derivatives(geo, 5, xi_3D)

                #     if typeof(geo) <: Geometry.TensorProductGeometry
                #         # Also test the method for non-cartesian points. This is only a
                #         # different method for tensor product geometries.
                #         @test_opt Geometry.evaluate(geo, 5, xi_3D_set)
                #         @test_opt Geometry.jacobian(geo, 5, xi_3D_set)
                #         @test_opt Geometry.hessian(geo, 5, xi_3D_set)
                #     end
                # elseif Geometry.get_manifold_dim(geo) == 4
                #     @test_opt Geometry.evaluate(geo, 5, xi_4D)
                #     @test_opt Geometry.jacobian(geo, 5, xi_4D)
                #     @test_opt Geometry.hessian(geo, 5, xi_4D)
                #     @test_opt Geometry.inv_metric(geo, 5, xi_4D)
                #     @test_opt Geometry.metric_derivatives(geo, 5, xi_4D)
                # elseif Geometry.get_manifold_dim(geo) == 11
                #     # println(@code_warntype Geometry.jacobian(geo, 5, xi_11D))
                #     # error()
                #     @test_opt Geometry.evaluate(geo, 5, xi_11D)
                #     @test_opt Geometry.jacobian(geo, 5, xi_11D)
                #     @test_opt Geometry.hessian(geo, 5, xi_11D)
                #     @test_opt Geometry.inv_metric(geo, 5, xi_11D)
                #     # The 11D metric_derivative is not type stable, so we avoid testing it for
                #     # now.
                #     # @test_opt Geometry.metric_derivatives(geo, 5, xi_11D)

                #     if typeof(geo) <: Geometry.TensorProductGeometry
                #         # Also test the method for non-cartesian points. This is only a
                #         # different method for tensor product geometries.
                #         @test_opt Geometry.evaluate(geo, 5, xi_11D_set)
                #         @test_opt Geometry.jacobian(geo, 5, xi_11D_set)
                #         @test_opt Geometry.hessian(geo, 5, xi_11D_set)
                #     end
            else
                @warn "FunctionSpacesInference: This space was not tested: $(space)"
            end
        end

    else
        println("Skipped function spaces inference tests. JET was not loaded.")
    end
else
    println("Skipped function spaces inference tests. Not on Julia 1.12.")
end

end
