module GeometryInferenceTests

import Pkg

using Mantis

using Test

# We need JET for the tests in this file, but JET does not always work for every julia
# version. These tests were made on 1.12, so we start there.
@static if Base.VERSION >= v"1.12"
    using JET
end

if Base.VERSION >= v"1.12"
    # We can only run the tests if JET was loaded.
    if any(x -> x.name == "JET", values(Pkg.dependencies()))
        # Test geometries
        # Reduction test, single-patch, single element, 1D. Cartesian.
        geometry1 = Geometry.CartesianGeometry(([-1, 1],))
        # Multi-patch input. 100 patches, 1D. Cartesian.
        geometryMP100 = Geometry.CartesianGeometry(
            ntuple(100) do i
                return (LinRange((i - 1) * 1.0, i * 1.0, i + 1),)
            end,
        )
        # Single-patch, 4D.
        geometry1p4D = Geometry.CartesianGeometry((
            LinRange(0.5, 2.5, 5),
            LinRange(-0.75, 0.75, 3),
            LinRange(1.5, 2.5, 4),
            LinRange(10.5, 20.5, 6),
        ))
        # 2D, single-patch, heterogeneous input.
        geometryMP = Geometry.CartesianGeometry((
            (LinRange(0.5, 2.5, 5), [-0.75, 0.1, 0.75]),
            (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 6)),
        ))
        # 3D tensor product
        cg1d = Geometry.CartesianGeometry((
            (LinRange(0.0, 1.0, 2),), (LinRange(1.0, 2.0, 3),), (LinRange(2.0, 3.0, 4),)
        ))
        cg2d = Geometry.CartesianGeometry((
            (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)), # First patch
            (LinRange(1.0, 2.0, 6), LinRange(0.0, 1.0, 7)), # Second patch
        ))
        tpgeo = Geometry.TensorProductGeometry((cg2d, cg1d))
        # 11D, TensorProduct, mixed
        tpgeo11D = Geometry.TensorProductGeometry((
            tpgeo, geometry1p4D, cg2d, cg1d, geometry1
        ))
        # 2D, Mapped
        # Mappings to create the deformed geometries. The mappings are defined with reference
        # to the unit square [0,1]x[0,1] as parametric domain.
        function mapping_patch_1_slant(x::AbstractVector{Float64}, slant_factor=0.25)
            return [x[1] + slant_factor * x[1] * x[2], x[2]]
        end
        function dmapping_patch_1_slant(x::AbstractVector{Float64}, slant_factor=0.25)
            return [
                [1.0 + slant_factor * x[2] slant_factor * x[1]]
                [0.0 1.0]
            ]
        end
        mapping_patch_1_slanted = Geometry.Mapping(
            (2, 2), mapping_patch_1_slant, dmapping_patch_1_slant
        )
        function mapping_patch_2_slant(x::AbstractVector{Float64}, slant_factor=0.25)
            return [x[1] + 1.0 + slant_factor * (1.0 - x[1]) * x[2], x[2]]
        end
        function dmapping_patch_2_slant(x::AbstractVector{Float64}, slant_factor=0.25)
            return [
                [1.0 - slant_factor * x[2] slant_factor * (1.0 - x[1])]
                [0.0 1.0]
            ]
        end
        mapping_patch_2_slanted = Geometry.Mapping(
            (2, 2), mapping_patch_2_slant, dmapping_patch_2_slant
        )
        num_elements_per_dim_per_patch = ((4, 4), (5, 6))
        geom_cart_patch_1 = Geometry.CartesianGeometry((
            0.0:(1.0 / num_elements_per_dim_per_patch[1][1]):1.0,
            0.0:(1.0 / num_elements_per_dim_per_patch[1][2]):1.0,
        ))
        geom_cart_patch_2 = Geometry.CartesianGeometry((
            0.0:(1.0 / num_elements_per_dim_per_patch[2][1]):1.0,
            0.0:(1.0 / num_elements_per_dim_per_patch[2][2]):1.0,
        ))
        # Mapped, explicit geometry and mapping per patch.
        geom_slanted_1patch = Geometry.MappedGeometry(
            (geom_cart_patch_1,), (mapping_patch_1_slanted,)
        )
        geom_slanted_1patch2 = Geometry.MappedGeometry(
            (geom_cart_patch_2,), (mapping_patch_2_slanted,)
        )
        geom_slanted_2patch = Geometry.MappedGeometry(
            (geom_cart_patch_1, geom_cart_patch_2),
            (mapping_patch_1_slanted, mapping_patch_2_slanted),
        )
        # Mapped, one parametric domain with multiple mappings.
        geom_slanted_2patch_oneref = Geometry.MappedGeometry(
            Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 7)),)),
            (mapping_patch_1_slanted, mapping_patch_2_slanted),
        )
        # Mapped, multiple patches with one map.
        geom_slanted_2patch_onemap = Geometry.MappedGeometry(
            (
                Geometry.CartesianGeometry(((
                    LinRange(0.0, 0.5, 5), LinRange(0.0, 1.0, 7)
                ),)),
                Geometry.CartesianGeometry(((
                    LinRange(0.5, 1.0, 4), LinRange(0.0, 1.0, 7)
                ),)),
            ),
            mapping_patch_1_slanted,
        )
        # Mapped, single mapping, single patch.
        geom_slanted_2patch_11 = Geometry.MappedGeometry(
            geom_cart_patch_1, mapping_patch_1_slanted
        )

        # Unstructured
        geom_unstr = Geometry.UnstructuredGeometry((
            Geometry.TensorProductGeometry((
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
            )),
            Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5))),
            geom_slanted_1patch,
        ))
        geom_unstr2 = Geometry.UnstructuredGeometry((
            Geometry.TensorProductGeometry((
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
            )),
            Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5))),
            geom_slanted_1patch,
            Geometry.TensorProductGeometry((
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 4),))),
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 4),))),
            )),
        ))
        geom_unstr3 = Geometry.UnstructuredGeometry((
            Geometry.TensorProductGeometry((
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 2),))),
            )),
            Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5))),
            geom_slanted_1patch,
            Geometry.TensorProductGeometry((
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 4),))),
                Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 4),))),
            )),
            geom_slanted_1patch2,
        ))

        const geos = (
            geometry1,
            geometryMP100,
            geometry1p4D,
            geometryMP,
            tpgeo,
            tpgeo11D,
            geom_slanted_2patch,
            geom_slanted_2patch_oneref,
            geom_slanted_2patch_onemap,
            geom_slanted_2patch_11,
            geom_unstr,
            geom_unstr2,
        )

        const xi_1D = Points.CartesianPoints(([0.0, 1.0],))
        const xi_2D = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
        const xi_3D = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
        const xi_3D_set = Points.PointSet(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
        const xi_4D = Points.CartesianPoints((
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        ))
        const xi_11D = Points.CartesianPoints((
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ))
        const xi_11D_set = Points.PointSet((
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ))

        foreach(geos) do geo
            # Note that JET only uses the types of the inputs, so which numbers we pick here is
            # irrelevant.
            @test_opt Geometry.get_patch_id(geo, 14)

            @test_opt Geometry.get_patch_and_local_element_id(geo, 14)

            @test_opt Geometry.get_global_element_id(geo, 14, 12)

            @test_opt Geometry.get_num_elements(geo)

            @test_opt Geometry.get_num_elements(geo, 58)

            @test_opt Geometry.get_num_elements_per_patch(geo)

            @test_opt Geometry.get_element_vertices(geo, 24)

            @test_opt Geometry.get_element_lengths(geo, 24)

            @test_opt Geometry.get_element_measure(geo, 24)

            if Geometry.get_manifold_dim(geo) == 1
                @test_opt Geometry.Geometry.evaluate(geo, 5, xi_1D)
                @test_opt Geometry.Geometry.jacobian(geo, 5, xi_1D)
                @test_opt Geometry.Geometry.inv_metric(geo, 5, xi_1D)
            elseif Geometry.get_manifold_dim(geo) == 2
                @test_opt Geometry.Geometry.evaluate(geo, 5, xi_2D)
                @test_opt Geometry.Geometry.jacobian(geo, 5, xi_2D)
                @test_opt Geometry.Geometry.inv_metric(geo, 5, xi_2D)
            elseif Geometry.get_manifold_dim(geo) == 3
                @test_opt Geometry.Geometry.evaluate(geo, 5, xi_3D)
                @test_opt Geometry.Geometry.jacobian(geo, 5, xi_3D)
                @test_opt Geometry.Geometry.inv_metric(geo, 5, xi_3D)

                if typeof(geo) <: Geometry.TensorProductGeometry
                    # Also test the method for non-cartesian points. This is only a different method
                    # for tensor product geometries.
                    @test_opt Geometry.Geometry.evaluate(geo, 5, xi_3D_set)
                    @test_opt Geometry.Geometry.jacobian(geo, 5, xi_3D_set)
                end
            elseif Geometry.get_manifold_dim(geo) == 4
                @test_opt Geometry.Geometry.evaluate(geo, 5, xi_4D)
                @test_opt Geometry.Geometry.jacobian(geo, 5, xi_4D)
                @test_opt Geometry.Geometry.inv_metric(geo, 5, xi_4D)
            elseif Geometry.get_manifold_dim(geo) == 11
                # println(@code_warntype Geometry.Geometry.jacobian(geo, 5, xi_11D))
                # error()
                @test_opt Geometry.Geometry.evaluate(geo, 5, xi_11D)
                @test_opt Geometry.Geometry.jacobian(geo, 5, xi_11D)
                @test_opt Geometry.Geometry.inv_metric(geo, 5, xi_11D)

                if typeof(geo) <: Geometry.TensorProductGeometry
                    # Also test the method for non-cartesian points. This is only a different method
                    # for tensor product geometries.
                    @test_opt Geometry.Geometry.evaluate(geo, 5, xi_11D_set)
                    @test_opt Geometry.Geometry.jacobian(geo, 5, xi_11D_set)
                end
            else
                @warn "GeometryInference: This geometry was not tested: $(geo)"
            end
        end

    else
        println("Skipped geometry inference tests. JET was not loaded.")
    end
else
    println("Skipped geometry inference tests. Not on Julia 1.12.")
end

end
