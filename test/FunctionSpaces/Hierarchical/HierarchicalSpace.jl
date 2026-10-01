# Test according to the example in Fig 9. of https://doi.org/10.1007/s11831-022-09752-5

using Mantis, Test
using Mantis.Hierarchical
include("../../TestHelpers.jl")

const THB = FunctionSpaces.THB
const HB = FunctionSpaces.HB
const SelectionStandard = FunctionSpaces.SelectionStandard

@testset "Polynomial Representation" verbose=true begin
    # Create the hierarhical space
    p = 3
    nlevels = 3
    subdiv = 2
    bsp_1 = FunctionSpaces.create_bspline_space(0.0, 1.0, 6, 3, 2)
    scal_1 = FunctionSpaces.scaling_uniform(bsp_1, 2)
    bsp_2 = get_child(scal_1)
    scal_2 = FunctionSpaces.scaling_uniform(bsp_2, 2)
    bsp_3 = get_child(scal_2)
    geo_active = ActiveInfo([[1, 6], [3, 9, 10], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    geo_scal_1 = Geometry.scaling_uniform(
        FunctionSpaces.get_geometry(bsp_1), FunctionSpaces.get_geometry(bsp_2), 2
    )
    geo_scal_2 = Geometry.scaling_uniform(
        FunctionSpaces.get_geometry(bsp_2), FunctionSpaces.get_geometry(bsp_3), 2
    )
    active_basis = Dict{Int, Vector{Int}}(
        1 => [1, 2, 3, 4, 6, 7, 8, 9], 2 => [6, 9, 10], 3 => collect(10:16)
    )

    for basis_type in (HB, THB)
        hgeo = Geometry.HierarchicalGeometry(geo_active, (geo_scal_1, geo_scal_2))
        hier_space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, (scal_1, scal_2), SelectionStandard, basis_type
        )
        basis = FunctionSpaces.get_basis(hier_space)

        # sanity check the active basis functions
        # NOTE: we sort because the ordering of the basis is an implementation detail
        @test @expected active_basis level ->
            sort(FunctionSpaces.get_level_ids(basis, level))

        # test polynomial representation
        nxi = 20
        points = LinRange(0, 1, nxi)
        xi = Points.PointSet((points,))
        nx = Geometry.get_num_elements(hgeo) * nxi
        x_to_degree = [Vector{Float64}(undef, nx) for _ in 0:p]
        A = zeros(nx, FunctionSpaces.get_num_basis(hier_space))
        for element_id in 1:FunctionSpaces.get_num_elements(hier_space)
            level, element_level_id = Geometry.convert_to_level_and_level_id(
                hgeo, element_id
            )
            vertices = Geometry.get_element_vertices(hgeo, element_id)[1]
            x = Geometry.affine_map.(points, vertices[2] - vertices[1], vertices[1])
            ids = ((element_id - 1) * nxi + 1):(element_id * nxi)
            for degree in eachindex(x_to_degree)
                x_to_degree[degree][ids] .= x .^ (degree - 1)
            end

            h_eval, h_inds = FunctionSpaces.evaluate(hier_space, element_id, xi, 0)
            A[ids, h_inds] = h_eval[1][1][1]
        end

        for (degree_vals, tol) in zip(x_to_degree, (1e-12, 1e-14, 1e-14, 1e-14))
            coeffs = A \ degree_vals
            @test all(v -> isapprox(v, 0.0; atol=tol), A * coeffs .- degree_vals)
        end
    end
end
