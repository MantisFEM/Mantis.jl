module HierarchicalMultiComponentTests

using Mantis

using Test

function create_hier_ds(parent_splines)
    bspline_DS = FunctionSpaces.DirectSumSpace(parent_splines)
    two_scale_operators_DS = Vector{FunctionSpaces.TwoScaleOperator}(undef, nlevels - 1)
    bsplines_DS = Vector{FunctionSpaces.DirectSumSpace{2, 2, 1}}(undef, nlevels)
    bsplines_DS[1] = bspline_DS
    for l in 1:(nlevels - 1)
        ts_operator, bspline_l = FunctionSpaces.build_two_scale_operator(
            bsplines_DS[l], subdiv
        )
        two_scale_operators_DS[l] = ts_operator
        bsplines_DS[l + 1] = bspline_l
    end
    hier_space = FunctionSpaces.HierarchicalFiniteElementSpace(
        bsplines_DS, two_scale_operators_DS, refined_domains, subdiv
    )

    return hier_space
end

function create_ds_hier(parent_splines)
    two_scale_operators = [
        Vector{FunctionSpaces.TensorProductTwoScaleOperator}(undef, nlevels - 1) for
        _ in 1:2
    ]
    bsplines = [Vector{typeof(parent_splines[c])}(undef, nlevels) for c in 1:2]
    for c in 1:2
        bsplines[c][1] = parent_splines[c]
        for l in 1:(nlevels - 1)
            ts_operator, bspline_l = FunctionSpaces.build_two_scale_operator(
                bsplines[c][l], subdiv
            )
            two_scale_operators[c][l] = ts_operator
            bsplines[c][l + 1] = bspline_l
        end
    end

    # hierarchical space built component wise
    hier_space = [
        FunctionSpaces.HierarchicalFiniteElementSpace(
            bsplines[c], two_scale_operators[c], refined_domains, subdiv
        ) for c in 1:2
    ]
    hier_space = FunctionSpaces.DirectSumSpace(tuple(hier_space...))

    return hier_space
end

p = (3, 2)
k1 = (2, 1)
k2 = (1, 0)
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (5, 5)
nlevels = 3
subdiv = (2, 2)
∫ₐ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, (4, 4))[1]
bsplines = (
    FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k1),
    FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k2),
)
const refined_domains = FunctionSpaces.HierarchicalActiveInfo([
    collect(1:25),
    [100, 99, 98, 97, 90, 89, 88, 87, 80, 79, 78, 77, 70, 69, 68, 67],
    [400, 399, 398, 397, 380, 379, 378, 377, 360, 359, 358, 357, 340, 339, 338, 337],
])
hier_ds = create_hier_ds(bsplines)
ds_hier = create_ds_hier(bsplines)
geo = Geometry.HierarchicalGeometry(FunctionSpaces.get_component_spaces(ds_hier)[1])
fs_hier_ds = Forms.FormSpace(1, geo, hier_ds, "HDS")
fs_ds_hier = Forms.FormSpace(1, geo, ds_hier, "DSH")
c1s = rand(4)
c2s = rand(3)
sol_expr =
    x -> [
        sum(c1s[i + 1] * x[:, 1] .^ i for i in 0:p[1]) +
        sum(c2s[i + 1] * x[:, 2] .^ i for i in 0:p[2]),
    ]
sol = Forms.AnalyticalFormField(1, sol_expr, geo, "sol")
dΩ = Quadrature.StandardQuadrature(∫ₐ, Geometry.get_num_elements(geo))
hier_ds_sol = Assemblers.solve_L2_projection(fs_hier_ds, sol, dΩ)
ds_hier_sol = Assemblers.solve_L2_projection(fs_ds_hier, sol, dΩ)
@test isapprox(Analysis.L2_norm(hier_ds_sol - sol, dΩ), 0.0; atol=1e-14)
@test isapprox(Analysis.L2_norm(ds_hier_sol - sol, dΩ), 0.0; atol=1e-14)
@test all(isapprox.(sort(hier_ds_sol.coefficients), sort(ds_hier_sol.coefficients)))

end
