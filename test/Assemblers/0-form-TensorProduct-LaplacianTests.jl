module ZeroFormTensorProductLaplacianTests

using Mantis

using Test
using LinearAlgebra
using SparseArrays
using DelimitedFiles

include("./AssemblerTestsHelpers.jl")

# exact solution for the 0-form problem
function sinusoidal_solution(
    geo::Geometry.AbstractGeometry{manifold_dim}
) where {manifold_dim}
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # u = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = @. sin(ω * x)
        return [vec(prod(y; dims=2))]
    end
    function grad_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = [ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        y = sin.(ω .* x)
        z = ω .* cos.(ω .* x)
        w = Vector{Vector{Float64}}(undef, size(x, 2))
        for i in 1:size(x, 2)
            w[i] = z[:, i] .* prod(y[:, setdiff(1:size(x, 2), i)]; dims=2)[:, 1]
        end
        return w
    end
    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x); dims=2)
        y = @. manifold_dim * ω * ω * y
        return [vec(y)]
    end
    u⁰ = Forms.AnalyticalFormField(0, my_sol, geo, "u")
    du⁰ = Forms.AnalyticalFormField(1, grad_my_sol, geo, "du")
    f⁰ = Forms.AnalyticalFormField(0, laplace_my_sol, geo, "f")
    return u⁰, du⁰, f⁰
end

# RUN POISSON PROBLEM -------------------------------------------------------------------
function test_0form_hodge_laplacian(
    manifold_dim,
    p⁰,
    section_space_type,
    mesh_type,
    origin,
    L,
    num_elements;
    verbose=false,
    test=true,
)
    errors = zeros(Float64, length(p⁰), length(section_space_type), length(mesh_type), 2)
    for (mesh_idx, mesh) in enumerate(mesh_type)
        for (p_idx, p) in enumerate(p⁰)
            for (ss_idx, section_space) in enumerate(section_space_type)
                if verbose
                    @info(
                        "Running $(manifold_dim)D zero-form Hodge-Laplace tests for p = $p, section_space = $section_space, mesh = $mesh"
                    )
                end

                # section space degrees
                degree = ntuple(manifold_dim) do _
                    return p
                end

                # function space regularities
                regularities = degree .- 1
                if section_space == FunctionSpaces.LobattoLegendre
                    regularities = tuple([0 for _ in 1:manifold_dim]...)
                end

                # section spaces
                if section_space == FunctionSpaces.GeneralizedTrigonometric
                    section_spaces = map(section_space, degree, (θ, θ), L ./ num_elements)
                    dq⁰ = 2 .* degree
                elseif section_space == FunctionSpaces.GeneralizedExponential
                    section_spaces = map(section_space, degree, (α, α), L ./ num_elements)
                    dq⁰ = 3 .* degree
                else
                    section_spaces = map(section_space, degree)
                    dq⁰ = ntuple(i -> 2, manifold_dim)
                end

                if mesh == "cartesian"
                    X = Forms.create_tensor_product_bspline_de_rham_complex(
                        origin, L, num_elements, section_spaces, regularities
                    )
                else
                    X = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
                        origin, L, num_elements, section_spaces, regularities; crazy_c=0.1
                    )
                end

                # quadrature rule
                canonical_qrule = Quadrature.tensor_product_rule(
                    degree .+ dq⁰, Quadrature.gauss_legendre
                )
                geometry = Forms.get_geometry(X[1])
                # global quadrature rule
                dΩ = Quadrature.StandardQuadrature(
                    canonical_qrule, Geometry.get_num_elements(geometry)
                )

                # number of dofs
                n_dofs = Forms.get_num_basis(X[1])
                if verbose
                    display("   n_dofs = $n_dofs")
                end
                # exact solution for the problem
                uₑ, duₑ, fₑ = sinusoidal_solution(geometry)

                # solve the problem
                uₕ = Assemblers.solve_zero_form_hodge_laplacian(X[1], fₑ, dΩ)
                if test
                    ref_coeffs = read_data(sub_dir, "$p-$section_space-$mesh.txt")
                    @test all(
                        isapprox.(
                            uₕ.coefficients, ref_coeffs, atol=atol * 10, rtol=rtol * 10
                        ),
                    )
                end

                # compute error
                error = Analysis.L2_norm(uₕ - uₑ, dΩ)
                derror = Analysis.L2_norm(Forms.ExteriorDerivative(uₕ) - duₑ, dΩ)
                errors[p_idx, ss_idx, mesh_idx, 1] = error
                errors[p_idx, ss_idx, mesh_idx, 2] = derror

                if verbose
                    display("   L2-Error: $error")
                    display("   H1-Error: $derror")
                end
                if verbose
                    println("...done!")
                end
            end
        end
    end

    if test
        ref_errors = read_data(sub_dir, "errors.txt")
        for i in eachindex(errors)
            @test isapprox(errors[i], ref_errors[i], atol=atol, rtol=rtol)
        end
    end
end

# 2D Problem Parameters
# manifold dimensions
manifold_dim_2D = 2
# mesh types to be used
mesh_type_2D = ["cartesian", "curvilinear"]
# type of section spaces to use
const θ = 2 * pi
const α = 10.0
section_space_type_2D = (
    FunctionSpaces.Bernstein,
    FunctionSpaces.LobattoLegendre,
    FunctionSpaces.GeneralizedTrigonometric,
    FunctionSpaces.GeneralizedExponential,
)
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 4
num_elements_2D = num_el_0 .* tuple([1 for _ in 1:manifold_dim_2D]...)
# origin of the parametric domain in each direction
origin_2D = ntuple(i -> 0.0, manifold_dim_2D)
# length of the domain in each direction
L_2D = ntuple(i -> 1.0, manifold_dim_2D)

# 3D Problem Parameters
# manifold dimensions
manifold_dim_3D = 3
# mesh types to be used
mesh_type_3D = ["cartesian"]
# type of section spaces to use
section_space_type_3D = (FunctionSpaces.Bernstein, FunctionSpaces.LobattoLegendre)
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 4
num_elements_3D = num_el_0 .* tuple([1 for _ in 1:manifold_dim_3D]...)
# origin of the parametric domain in each direction
origin_3D = ntuple(i -> 0.0, manifold_dim_3D)
# length of the domain in each direction
L_3D = ntuple(i -> 1.0, manifold_dim_3D)

# General Problem Parameters
# sub-directory for data
sub_dir = "0-form-TensorProduct-Laplacian"
# print info?
verbose = false
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [2, 3]

# Run 2D case with tests (we have the data).
test_0form_hodge_laplacian(
    manifold_dim_2D,
    p⁰,
    section_space_type_2D,
    mesh_type_2D,
    origin_2D,
    L_2D,
    num_elements_2D;
    verbose=verbose,
    test=true,
)
# Run the 3D case, but without tests, since we do not have the data.
test_0form_hodge_laplacian(
    manifold_dim_3D,
    p⁰,
    section_space_type_3D,
    mesh_type_3D,
    origin_3D,
    L_3D,
    num_elements_3D;
    verbose=verbose,
    test=false,
)

end
