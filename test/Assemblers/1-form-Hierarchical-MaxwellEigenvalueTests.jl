module HierarchicalBsplinesMaxwellEigenvalueTests

using Mantis

using Test

# Refer to the following file for method and variable definitions
include("../../examples/HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
const starting_point = (0.0, 0.0)
const box_size = (fpi, fpi)
num_elements = (2, 2) .^ 3 # Initial mesh size.

# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = true
simplified = false
num_steps = 3 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
dorfler_parameter = 0.2
Lchains = true # Decide if Lchains are added to fix inexact refinements.
eigenfunc = 1 # Eigenfunction to use for adaptive refinement.

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
qrule_assembly, qrule_error = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ = Quadrature.StandardQuadrature(qrule_assembly, prod(num_elements))
dΩₑ = Quadrature.StandardQuadrature(qrule_error, prod(num_elements))

# Number of eigenvalues to compute
const num_eig = 5
# Scaling form maxwell eigenfunctions.
scale_factors = ntuple(2) do k
    return pi / (box_size[k] - starting_point[k])
end

############################################################################################
#                                       Run problem                                        #
############################################################################################
function run_problems(
    complex::C,
    dΩₐ::Quadrature.StandardQuadrature{manifold_dim},
    num_steps::Int,
    dorfler_parameter::Float64,
    dΩₑ::Quadrature.StandardQuadrature{manifold_dim},
    Lchains::Bool,
    eigenfunction::Int,
    num_eig::Int,
    scale_factors::NTuple{manifold_dim, Float64};
    verbose::Bool=false,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}
    compt_eigvals, compt_eigfuncs = Assemblers.solve_maxwell_eig(
        complex,
        dΩₐ,
        num_steps,
        dorfler_parameter,
        dΩₑ,
        Lchains,
        eigenfunction,
        num_eig,
        scale_factors;
        verbose=verbose,
    )
    geo = Forms.get_geometry(compt_eigfuncs...)
    dΩₑ = Quadrature.StandardQuadrature(
        Quadrature.get_canonical_quadrature_rule(dΩₑ), Geometry.get_num_elements(geo)
    )
    exact_eigvals, exact_eigfuncs = Assemblers.get_analytical_maxwell_eig(
        num_eig, geo, scale_factors
    )
    eigval_errors = compt_eigvals - exact_eigvals
    eigfunc_errors =
        Analysis.compute_error_total.(compt_eigfuncs, exact_eigfuncs, Ref(dΩₑ), Ref("L2"))

    return eigval_errors, eigfunc_errors
end

function runt_tests(eigval_errors)
    for error in eigval_errors
        @test isapprox(abs(error), 0.0; atol=2e-2)
    end

    return nothing
end

R_complex = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
eigval_errors, eigfunc_errors = run_problems(
    R_complex,
    dΩₐ,
    num_steps,
    dorfler_parameter,
    dΩₑ,
    Lchains,
    eigenfunc,
    num_eig,
    scale_factors,
)
runt_tests(eigval_errors)

end
