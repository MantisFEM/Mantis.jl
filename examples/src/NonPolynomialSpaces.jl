# # Non-polynomial function spaces

# In `Mantis` the *section spaces* underlying the finite elements need not be polynomial. By
# swapping the canonical (per-direction) space, the same machinery that produces classical
# B-splines produces their trigonometric or exponential generalisations, the *extended complete
# Tchebycheffian* (ECT) B-splines. These can represent functions such as sinusoids or
# exponentials with fewer degrees of freedom than polynomials. In this example we project a
# sinusoidal target onto ``0``-form spaces built from three different section spaces and compare
# the accuracy.

# ## Section spaces

# Recall (see the [FunctionSpaces](@ref) module) that a finite element
# space is assembled from element-local *canonical* spaces. `Mantis` provides, among others:
#
# * [`FunctionSpaces.Bernstein`](@ref) for the classical polynomial choice (B-splines),
# * [`FunctionSpaces.GeneralizedTrigonometric`](@ref) for trigonometric ECT B-splines,
# * [`FunctionSpaces.GeneralizedExponential`](@ref) for exponential ECT B-splines.
#
# The trigonometric and exponential spaces take, in addition to the degree, a shape parameter
# and the element length per direction.

# ## Implementation

using Mantis

manifold_dim = 2
origin       = (0.0, 0.0)
box_size     = (1.0, 1.0)
num_elements = (8, 8)

degree       = (3, 3)
regularities = degree .- 1                # maximally smooth

element_lengths = box_size ./ num_elements
θ = 2.0 * pi                              # shape parameter for the trigonometric space
α = 10.0                                  # shape parameter for the exponential space

# We over-integrate (relative to assembly) so that the reported errors reflect the
# approximation quality of the spaces rather than quadrature error. Non-polynomial bases need
# somewhat richer quadrature than polynomials, so we are generous here.
canonical_qrule = Quadrature.tensor_product_rule(4 .* degree, Quadrature.gauss_legendre)

# The exact solution is a product of sines, a ``0``-form (a function).
function sinusoidal_target(geo)
    ω = 2.0 * pi
    expression(x::Matrix{Float64}) = [vec(prod((@. sin(ω * x)); dims=2))]
    return Forms.AnalyticalFormField(0, expression, geo, "f")
end

# Given a tuple of section spaces, this helper builds the de Rham complex, projects the target
# onto its ``0``-form space, and returns the ``L^2`` error of the projection.
function project_and_measure(section_spaces)
    X = Forms.create_tensor_product_bspline_de_rham_complex(
        origin, box_size, num_elements, section_spaces, regularities
    )
    geometry = Forms.get_geometry(X[1])
    dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

    fₑ = sinusoidal_target(geometry)
    fₕ = Assemblers.solve_L2_projection(X[1], fₑ, dΩ)

    return Analysis.L2_norm(fₕ - fₑ, dΩ)
end

# We build one set of section spaces per family. Note how `Bernstein` only needs the degree,
# while the ECT spaces also take a shape parameter and the element length.
bernstein_spaces     = map(FunctionSpaces.Bernstein, degree)
trigonometric_spaces = map(
    FunctionSpaces.GeneralizedTrigonometric, degree, (θ, θ), element_lengths
)
exponential_spaces   = map(
    FunctionSpaces.GeneralizedExponential, degree, (α, α), element_lengths
)

println("L² projection error of sin(2πx)·sin(2πy), degree $degree, $num_elements elements:")
println("  Bernstein (polynomial)    : ", project_and_measure(bernstein_spaces))
println("  Generalized trigonometric : ", project_and_measure(trigonometric_spaces))
println("  Generalized exponential   : ", project_and_measure(exponential_spaces))

# All three families approximate the target well, but the trigonometric space, built to
# represent sinusoids, is the most accurate here. No space is best in general; the point is
# that `Mantis` lets you choose the section space to match the problem, through the same de
# Rham-complex interface used everywhere else in the library.
