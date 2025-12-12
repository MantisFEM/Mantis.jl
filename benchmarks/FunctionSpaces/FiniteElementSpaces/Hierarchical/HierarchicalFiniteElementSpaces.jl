module EvalHierarchicalFiniteElementSpacesBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

function run_problem(manifold_dim::Int, p, ns, truncated::Bool)
    starting_point = ntuple(_ -> 0.0, manifold_dim)
    box_size = ntuple(_ -> 1.0, manifold_dim)
    num_elements = ntuple(_ -> 4, manifold_dim)
    deg = ntuple(_ -> p, manifold_dim)
    reg = ntuple(_ -> p - 1, manifold_dim)
    bsp_p = FunctionSpaces.create_bspline_space(
        starting_point, box_size, num_elements, deg, reg
    )
    nsub = ntuple(_ -> ns, manifold_dim)
    op_p, bsp_c = FunctionSpaces.build_two_scale_operator(bsp_p, nsub)
    op_c, bsp_cc = FunctionSpaces.build_two_scale_operator(bsp_c, nsub)
    bsplines = [bsp_p, bsp_c, bsp_cc]
    ops = [op_p, op_c]
    refined_domains = [
        vcat(
            FunctionSpaces.get_support(bsplines[l], 2),
            FunctionSpaces.get_support(
                bsplines[l], FunctionSpaces.get_num_basis(bsplines[l]) - 2
            ),
        ) for l in 1:3
    ]

    create_bench = @benchmarkable FunctionSpaces.HierarchicalFiniteElementSpace(
        $bsplines, $ops, $refined_domains, $nsub, $truncated
    ) samples = 500 evals = 1 seconds = Inf

    HB = FunctionSpaces.HierarchicalFiniteElementSpace(
        bsplines, ops, refined_domains, nsub, truncated
    )
    eval_points = Points.CartesianPoints(
        ntuple(dim -> LinRange(0.0, 1.0, 25), manifold_dim)
    )
    dim = FunctionSpaces.get_num_basis(HB)
    eval_bench = @benchmarkable FunctionSpaces.evaluate($HB, 1, $eval_points) samples = 500 evals =
        1 seconds = Inf

    return dim, create_bench, eval_bench
end

############################################################################################
#                                       Run problems                                        #
############################################################################################

group = BenchmarkGroup()
manifold_dims = [2, 3]
ps = [1, 2]
n_sub = [2, 3]
for ns in n_sub
    for manifold_dim in manifold_dims
        if ns == 3 && manifold_dim == 3
            continue
        end

        sub_group = BenchmarkGroup()
        for p in ps
            dim, create_bench, eval_bench = run_problem(manifold_dim, p, ns, false)
            name = "hb-num_basis=$(dim)-p=$(p)-k=$(p-1)"
            sub_group["create-" * name] = create_bench
            sub_group["eval-" * name] = eval_bench
            dim, create_bench, eval_bench = run_problem(manifold_dim, p, ns, true)
            name = "thb-num_basis=$(dim)-p=$(p)-k=$(p-1)"
            sub_group["create-" * name] = create_bench
            sub_group["eval-" * name] = eval_bench
        end

        group["$(manifold_dim)D-num_sub=$(ns)"] = sub_group
    end
end

end
