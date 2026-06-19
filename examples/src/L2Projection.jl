# # L2 projection

# The ``L^2`` projection is the simplest problem one can pose in `Mantis`: given a target
# form ``f^k``, find the discrete form ``f^k_h`` in a finite element form space that is
# closest to it in the ``L^2`` sense. It is a good first example because it exercises the whole
# `Mantis` pipeline (geometry, function space, form space, quadrature, assembly, and solve)
# without the complications of boundary conditions.

# ## Formulation

# Given ``f^k \in L^2\Lambda^k(\Omega)``, the ``L^2`` projection ``f^k_h \in \Lambda^k_h``
# is defined by the weak problem
# ```math
# \int_\Omega v^k_h \wedge \star f^k_h = \int_\Omega v^k_h \wedge \star f^k \quad
# \forall\, v^k_h \in \Lambda^k_h \;.
# ```
# The left-hand side is the (mass-matrix) inner product on ``\Lambda^k_h`` and the
# right-hand side tests the target against the basis. `Mantis` already provides this weak
# form, so we will use the convenience solver
# [`Assemblers.solve_L2_projection`](@ref Mantis.Assemblers.solve_L2_projection).

# ## Implementation

# We work on the unit square and build a *whole de Rham complex* in one call, so that we can
# project onto ``0``-, ``1``- and ``2``-form spaces with the same code.

using Mantis

manifold_dim = 2
origin       = (0.0, 0.0)
box_size     = (1.0, 1.0)
num_elements = (8, 8)

p = (3, 3)  # B-spline degree per direction
k = (2, 2)  # regularity per direction (C²)

# `create_tensor_product_bspline_de_rham_complex` returns a tuple of `manifold_dim + 1` form
# spaces, `X[1]` being the ``0``-form space, `X[2]` the ``1``-form space and `X[3]` the
# ``2``-form (top) space.
X = Forms.create_tensor_product_bspline_de_rham_complex(origin, box_size, num_elements, p, k)
geometry = Forms.get_geometry(X[1])

# We use a Gauss-Legendre rule with `p + 1` points per direction for the projection, and a
# finer rule for measuring the error so that quadrature error does not pollute the result.
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

canonical_qrule_error = Quadrature.tensor_product_rule(2 .* p, Quadrature.gauss_legendre)
dΩ_error = Quadrature.StandardQuadrature(
    canonical_qrule_error, Geometry.get_num_elements(geometry)
)

# As a target we use a sinusoid in every component. A ``k``-form in ``n`` dimensions has
# ``\binom{n}{k}`` components, so the exact solution adapts to the form rank.
function sinusoidal_target(form_rank::Int, geo)
    num_components = binomial(manifold_dim, form_rank)
    ω = 2.0 * pi
    function expression(x::Matrix{Float64})
        values = vec(prod((@. sin(ω * x)); dims=2))
        return repeat([values], num_components)
    end
    return Forms.AnalyticalFormField(form_rank, expression, geo, "f")
end

# We now project the target onto each space in the complex and report the ``L^2`` error.
for form_rank in 0:manifold_dim
    Xᵏ = X[form_rank + 1]
    fₑ = sinusoidal_target(form_rank, geometry)

    fₕ = Assemblers.solve_L2_projection(Xᵏ, fₑ, dΩ)

    error = Analysis.L2_norm(fₕ - fₑ, dΩ_error)
    println("form rank $form_rank: dim(Λᵏₕ) = $(Forms.get_num_basis(Xᵏ)), L² error = $error")
end

# The errors are small and decrease as the mesh is refined or the degree is raised; a
# convergence study (as in the [Biharmonic](@ref) example) would confirm the expected
# ``\mathcal{O}(h^{p+1})`` rate.
