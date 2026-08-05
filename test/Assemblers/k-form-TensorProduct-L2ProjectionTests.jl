module KFormTensorProductL2ProjectionTests

using Mantis

using Test
using LinearAlgebra
using SparseArrays
using DelimitedFiles

include("./AssemblerTestsHelpers.jl")

# PROBLEM PARAMETERS -------------------------------------------------------------------
# sub-directory for data
sub_dir = "k-form-TensorProduct-L2Projection"
# manifold dimensions
manifold_dim = 2
# mesh types to be used
mesh_type = ["cartesian", "curvilinear"]
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 4
num_elements = num_el_0 .* tuple([1 for _ in 1:manifold_dim]...)
# origin of the parametric domain in each direction
origin = (0.0, 0.0)
# length of the domain in each direction
L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [2, 3]
# type of section spaces to use
θ = 2 * pi
α = 10.0
section_space_type = [
    FunctionSpaces.Bernstein,
    FunctionSpaces.Lagrange,
    FunctionSpaces.GeneralizedTrigonometric,
    FunctionSpaces.GeneralizedExponential,
]
# print info?
verbose = false

# exact solution for the problem
function sinusoidal_solution(
    form_rank::Int, geo::Geometry.AbstractGeometry{manifold_dim}
) where {manifold_dim}
    n_form_components = binomial(manifold_dim, form_rank)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
        y = @. sin(ω * x)
        return repeat([vec(prod(y; dims=2))], n_form_components)
    end
    return Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end

# RUN L2 PROJECTION PROBLEM -------------------------------------------------------------------
errors = zeros(
    Float64, length(p⁰), length(section_space_type), length(mesh_type), 1 + manifold_dim
)
for (mesh_idx, mesh) in enumerate(mesh_type)
    for (p_idx, p) in enumerate(p⁰)
        for (ss_idx, section_space) in enumerate(section_space_type)
            if verbose
                @info(
                    "Running L2 projection tests for p = $p, section_space = $section_space, mesh = $mesh"
                )
            end

            # section space degrees
            degree = (p, p)

            # function space regularities
            regularities = degree .- 1

            # section spaces
            if section_space == FunctionSpaces.Lagrange
                nodes = ntuple(manifold_dim) do i
                    return Points.get_input_points(
                        Quadrature.get_nodes(
                            Quadrature.gauss_lobatto(degree[i]+1)
                        )
                    )[1]
                end
                section_spaces = map(section_space, nodes)
                regularities = tuple([0 for _ in 1:manifold_dim]...)
                dq⁰ = (2, 2)
            elseif section_space == FunctionSpaces.GeneralizedTrigonometric
                section_spaces = map(section_space, degree, (θ, θ), L ./ num_elements)
                dq⁰ = 2 .* degree
            elseif section_space == FunctionSpaces.GeneralizedExponential
                section_spaces = map(section_space, degree, (α, α), L ./ num_elements)
                dq⁰ = 3 .* degree
            else
                section_spaces = map(section_space, degree)
                dq⁰ = (2, 2)
            end

            # complex
            if mesh == "cartesian"
                X = Forms.create_tensor_product_bspline_de_rham_complex(
                    origin, L, num_elements, section_spaces, regularities
                )
            else
                X = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
                    origin, L, num_elements, section_spaces, regularities
                )
            end

            # quadrature rule
            canonical_qrule = Quadrature.tensor_product_rule(
                degree .+ dq⁰, Quadrature.gauss_legendre
            )
            # global quadrature rule
            geometry = Forms.get_geometry(X[1])
            dΩ = Quadrature.StandardQuadrature(
                canonical_qrule, Geometry.get_num_elements(geometry)
            )

            for form_rank in 0:manifold_dim
                n_dofs = Forms.get_num_basis(X[form_rank + 1])
                if verbose
                    display("   Form rank = $form_rank, n_dofs = $n_dofs")
                end
                # exact solution for the problem
                fₑ = sinusoidal_solution(form_rank, geometry)

                # solve the problem
                fₕ = Assemblers.solve_L2_projection(X[form_rank + 1], fₑ, dΩ)

                # read reference data and compare
                if section_space == FunctionSpaces.Lagrange
                    ref_coeffs = read_data(sub_dir, "$p-Mantis.FunctionSpaces.LobattoLegendre-$mesh-$form_rank.txt")
                else
                    ref_coeffs = read_data(sub_dir, "$p-$section_space-$mesh-$form_rank.txt")
                end
                @test all(
                    isapprox.(fₕ.coefficients, ref_coeffs, atol=atol * 20, rtol=rtol * 20)
                )

                # compute error
                err = Analysis.L2_norm(fₕ - fₑ, dΩ)
                errors[p_idx, ss_idx, mesh_idx, form_rank + 1] = err

                if verbose
                    display("   Error: $err")
                end
            end
            if verbose
                println("...done!")
            end
        end
    end
end

ref_errors = read_data(sub_dir, "errors.txt")
for i in eachindex(errors)
    @test isapprox(errors[i], ref_errors[i], atol=atol, rtol=rtol)
end

end
