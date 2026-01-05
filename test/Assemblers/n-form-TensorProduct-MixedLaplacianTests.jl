module NFormTensorProductMixedLaplacianTests

using Mantis

using Test
using LinearAlgebra
using SparseArrays
using DelimitedFiles

include("./AssemblerTestsHelpers.jl")

# PROBLEM PARAMETERS -------------------------------------------------------------------
# sub-directory for data
sub_dir = "n-form-TensorProduct-MixedLaplacian"
# manifold dimensions
manifold_dim = 2
# mesh types to be used
mesh_type = ["cartesian", "curvilinear"]
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 5
num_elements = num_el_0 .* tuple([1 for _ in 1:manifold_dim]...)
# origin of the parametric domain in each direction
origin = (0.0, 0.0)
# length of the domain in each direction
L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [3]
# type of section spaces to use
θ = 2 * pi
α = 10.0
section_space_type = [
    FunctionSpaces.Bernstein,
    FunctionSpaces.LobattoLegendre,
    FunctionSpaces.GeneralizedTrigonometric,
    FunctionSpaces.GeneralizedExponential,
]
# print info?
verbose = false

# exact solution for the 0-form problem
function sinusoidal_solution(geo::Geometry.AbstractGeometry)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # [u] = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
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

    function flux_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = ⋆[ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        w = grad_my_sol(x)
        if size(x, 2) == 1
            # (a) -> (a)
            return [w[1]]

        elseif size(x, 2) == 2
            # (a, b) -> (b, -a)
            w = [w[2], -w[1]]
            return w

        elseif size(x, 2) == 3
            # (a, b, c) -> (a, b, c)
            return [w[1], w[2], w[3]]

        else
            throw(ArgumentError("Dimension not supported."))
        end
    end

    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x); dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end

    ϕ² = Forms.AnalyticalFormField(2, my_sol, geo, "ϕ")
    δϕ² = Forms.AnalyticalFormField(1, flux_my_sol, geo, "δϕ")
    f² = Forms.AnalyticalFormField(2, laplace_my_sol, geo, "f")

    return ϕ², δϕ², f²
end

# RUN POISSON PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, length(p⁰), length(section_space_type), length(mesh_type), 2)
for (mesh_idx, mesh) in enumerate(mesh_type)
    for (p_idx, p) in enumerate(p⁰)
        for (ss_idx, section_space) in enumerate(section_space_type)
            if verbose
                @info("""Running volume-form Hodge-Laplace tests for p = $p, \
                      section_space = $section_space, mesh = $mesh.""")
            end

            # section space degrees
            degree = (p, p)

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
                dq⁰ = (2, 2)
            end

            # quadrature rule
            canonical_qrule = Quadrature.tensor_product_rule(
                degree .+ dq⁰, Quadrature.gauss_legendre
            )
            dΩ = Quadrature.StandardQuadrature(canonical_qrule, prod(num_elements))

            # geometry
            if mesh == "cartesian"
                # create tensor-product B-spline complex
                X = Forms.create_tensor_product_bspline_de_rham_complex(
                    origin, L, num_elements, section_spaces, regularities
                )
            else
                X = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
                    origin, L, num_elements, section_spaces, regularities; crazy_c=0.1
                )
            end

            # number of dofs
            n_dofs = Forms.get_num_basis(X[2]) + Forms.get_num_basis(X[3])
            if verbose
                display("   n_dofs = $n_dofs")
            end

            geometry = Forms.get_geometry(X[1])
            # exact solution for the problem
            ϕₑ, δϕₑ, fₑ = sinusoidal_solution(geometry)

            # solve the problem
            uₕ, ϕₕ = Assemblers.solve_volume_form_hodge_laplacian(X[2], X[3], fₑ, dΩ)
            # if verbose
            #     display("   cond_num = $cond_num")
            # end
            # display([n_dofs cond_num])

            ref_coeffs = read_data(sub_dir, "$p-$section_space-$mesh-uh.txt")
            @show section_space
            @show mesh
            @show maximum(ref_coeffs - uₕ.coefficients)
            if maximum(ref_coeffs - uₕ.coefficients) > 1
                Plot.export_form_fields_to_vtk((ϕₕ, ϕₑ), "broken-test")
				exit()
            end

            @test all(
                isapprox.(uₕ.coefficients, ref_coeffs, atol=atol * 50, rtol=rtol * 50)
            )
            ref_coeffs = read_data(sub_dir, "$p-$section_space-$mesh-phih.txt")
            @test all(
                isapprox.(ϕₕ.coefficients, ref_coeffs, atol=atol * 10, rtol=rtol * 10)
            )

            # compute error
            error = Analysis.L2_norm(ϕₕ - ϕₑ, dΩ)
            δerror = Analysis.L2_norm(uₕ - δϕₑ, dΩ)
            errors[p_idx, ss_idx, mesh_idx, 1] = error
            errors[p_idx, ss_idx, mesh_idx, 2] = δerror

            if verbose
                display("   L2-Error in ϕ: $error")
                display("   L2-Error in δϕ: $δerror")
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
