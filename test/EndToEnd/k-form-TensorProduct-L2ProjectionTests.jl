module KFormTensorProductL2ProjectionTests

using Mantis

using Test
using LinearAlgebra
using SparseArrays
using DelimitedFiles

include("./AssemblerTestsHelpers.jl")

# PROBLEM PARAMETERS -------------------------------------------------------------------
# sub-directory for data
const sub_dir = "k-form-TensorProduct-L2Projection"
# mesh types to be used
const mesh_type = (
    ("cartesian", Geometry.CartesianGeometry), ("curvilinear", Geometry.MappedGeometry)
)
# number of elements in each direction
const num_elements = (4, 4)
# origin of the parametric domain in each direction
const origin = (0.0, 0.0)
# length of the domain in each direction
const L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
const p⁰ = (2, 3)
# type of section spaces to use
const θ = 2 * pi
const α = 10.0
const section_space_type = (
    FunctionSpaces.Bernstein,
    FunctionSpaces.Lagrange,
    FunctionSpaces.GeneralizedTrigonometric,
    FunctionSpaces.GeneralizedExponential,
)

# exact solution for the problem
function my_sol1(x::Matrix{Float64})
    # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
    y = @. sin(2.0 * pi * x)
    return [vec(prod(y; dims=2))]
end
function my_sol2(x::Matrix{Float64})
    # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
    y = @. sin(2.0 * pi * x)
    return repeat([vec(prod(y; dims=2))], 2)
end
function sinusoidal_solution(form::Forms.AbstractForm{2}, geo::Geometry.AbstractGeometry{2})
    form_rank = Forms.get_form_rank(form)
    return Forms.AnalyticalFormField(form_rank, my_sol1, geo, "f")
end
function sinusoidal_solution(::Forms.AbstractForm{2, 1}, geo::Geometry.AbstractGeometry{2})
    return Forms.AnalyticalFormField(1, my_sol2, geo, "f")
end

function create_section_spaces(
    section_space::Type{FunctionSpaces.Lagrange}, degree, θ, α, L, num_elements
)
    nodes = ntuple(2) do i
        return Points.get_input_points(
            Quadrature.get_nodes(Quadrature.gauss_lobatto(degree[i]+1))
        )[1]
    end
    section_spaces = map(section_space, nodes)
    regularities = ntuple(i -> 0, 2)
    dq⁰ = (2, 2)

    return section_spaces, regularities, dq⁰
end
function create_section_spaces(
    section_space::Type{FunctionSpaces.GeneralizedTrigonometric},
    degree,
    θ,
    α,
    L,
    num_elements,
)
    section_spaces = map(section_space, degree, (θ, θ), L ./ num_elements)
    regularities = degree .- 1
    dq⁰ = 2 .* degree

    return section_spaces, regularities, dq⁰
end
function create_section_spaces(
    section_space::Type{FunctionSpaces.GeneralizedExponential},
    degree,
    θ,
    α,
    L,
    num_elements,
)
    section_spaces = map(section_space, degree, (α, α), L ./ num_elements)
    regularities = degree .- 1
    dq⁰ = 3 .* degree

    return section_spaces, regularities, dq⁰
end
function create_section_spaces(
    section_space::Type{FunctionSpaces.Bernstein}, degree, θ, α, L, num_elements
)
    section_spaces = map(section_space, degree)
    regularities = degree .- 1
    dq⁰ = (2, 2)

    return section_spaces, regularities, dq⁰
end

function setup_complex(
    ::Type{Geometry.CartesianGeometry},
    origin,
    L,
    num_elements,
    section_spaces,
    regularities,
)
    return Forms.create_tensor_product_bspline_de_rham_complex(
        origin, L, num_elements, section_spaces, regularities
    )
end
function setup_complex(
    ::Type{Geometry.MappedGeometry}, origin, L, num_elements, section_spaces, regularities
)
    return Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
        origin, L, num_elements, section_spaces, regularities
    )
end

# RUN L2 PROJECTION PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, length(p⁰), length(section_space_type), length(mesh_type), 3)
for (mesh_idx, (mesh, mesh_T)) in pairs(mesh_type)
    for (p_idx, p) in pairs(p⁰)
        for (ss_idx, section_space) in pairs(section_space_type)
            degree = (p, p)
            section_spaces, regularities, dq⁰ = create_section_spaces(
                section_space, degree, θ, α, L, num_elements
            )

            X = setup_complex(mesh_T, origin, L, num_elements, section_spaces, regularities)

            canonical_qrule = Quadrature.tensor_product_rule(
                degree .+ dq⁰, Quadrature.gauss_legendre
            )
            geometry = Forms.get_geometry(first(X))
            dΩ = Quadrature.StandardQuadrature(
                canonical_qrule, Geometry.get_num_elements(geometry)
            )

            for (form_rank_plus1, k_form_space) in pairs(X)
                form_rank = form_rank_plus1 - 1

                fₑ = sinusoidal_solution(k_form_space, geometry)

                fₕ = Assemblers.solve_L2_projection(k_form_space, fₑ, dΩ)

                if section_space == FunctionSpaces.Lagrange
                    ref_coeffs = read_data(
                        sub_dir,
                        "$p-Mantis.FunctionSpaces.LobattoLegendre-$mesh-$form_rank.txt",
                    )
                else
                    ref_coeffs = read_data(
                        sub_dir, "$p-$section_space-$mesh-$form_rank.txt"
                    )
                end

                @test all(
                    isapprox.(fₕ.coefficients, ref_coeffs, atol=atol * 20, rtol=rtol * 20)
                )

                err = Analysis.L2_norm(fₕ - fₑ, dΩ)
                errors[p_idx, ss_idx, mesh_idx, form_rank + 1] = err
            end
        end
    end
end

ref_errors = read_data(sub_dir, "errors.txt")
for i in eachindex(errors)
    @test isapprox(errors[i], ref_errors[i], atol=atol, rtol=rtol)
end

end
