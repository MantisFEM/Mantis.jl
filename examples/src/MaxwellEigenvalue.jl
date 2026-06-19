# # Maxwell eigenvalue problem

# The Maxwell eigenvalue problem is the prototypical example where naive finite element
# discretisations fail and a *structure-preserving* one succeeds. It asks for the resonant
# modes of a cavity, and getting it right requires the discrete ``1``-form space to sit in a
# discrete de Rham complex, which is what `Mantis` provides.

# ## Formulation

# On a domain ``\Omega \subset \mathbb{R}^2`` we look for non-trivial ``1``-forms ``u^1`` and
# eigenvalues ``\omega^2`` such that
# ```math
# \int_\Omega \mathrm{d} v^1 \wedge \star\, \mathrm{d} u^1
#   = \omega^2 \int_\Omega v^1 \wedge \star\, u^1
#   \quad \forall\, v^1 \in \Lambda^1_{h,0}(\Omega)\;.
# ```
# Written with form spaces ``\Lambda^0_h \xrightarrow{\mathrm{d}} \Lambda^1_h
# \xrightarrow{\mathrm{d}} \Lambda^2_h`` this is a *generalised* eigenvalue problem
# ``A u = \omega^2 B u``. The discrete kernel of ``\mathrm{d}`` (the gradients of
# ``\Lambda^0_h``) produces spurious zero eigenvalues; because the spaces form a compatible
# complex, `Mantis` can identify and skip exactly this nullspace. The bookkeeping is handled
# by the solver [`Assemblers.solve_maxwell_eig`](@ref Mantis.Assemblers.solve_maxwell_eig).

# ## Implementation

# We solve on a rectangle. The side lengths are chosen so that the analytical eigenvalues
# ``\omega^2_{m,n} = (m\,\pi/L_x)^2 + (n\,\pi/L_y)^2`` are non-degenerate, which makes them
# easy to match up one-to-one with the computed ones.

using Mantis

starting_point = (0.0, 0.0)
box_size       = (Float64(pi), 1.0)  # incommensurate sides ⇒ simple (non-repeated) eigenvalues
num_elements   = (8, 8)

p = (3, 3)       # B-spline degree
k = p .- 1       # maximally-smooth (Cᵖ⁻¹) B-splines

num_eig = 5      # number of eigenvalues to compute

# We build the de Rham complex and pull out the ``0``- and ``1``-form spaces, which are the
# two spaces the Maxwell solver needs.
R = Forms.create_tensor_product_bspline_de_rham_complex(starting_point, box_size, num_elements, p, k)
R⁰, R¹ = R[1], R[2]
geometry = Forms.get_geometry(R⁰)

# A Gauss-Legendre rule with `p + 1` points per direction integrates the mass and stiffness
# blocks exactly.
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

# Solve the discrete eigenvalue problem...
computed_eigenvalues, computed_eigenfunctions = Assemblers.solve_maxwell_eig(
    R⁰, R¹, dΩ, num_eig
)

# ...and compare against the analytical eigenvalues and eigenfunctions. The scale factors
# encode the domain size in the analytical formula.
scale_factors = ntuple(2) do d
    return pi / (box_size[d] - starting_point[d])
end
exact_eigenvalues, exact_eigenfunctions = Assemblers.get_analytical_maxwell_eig(
    num_eig, geometry, scale_factors
)

# We report the error in each computed eigenvalue. (The matching eigenfunctions are returned
# in `computed_eigenfunctions` and can be exported to VTK with the [Plot](@ref) module; we do
# not measure their error here, because an eigenfunction is only defined up to sign and
# normalisation.)
for i in 1:num_eig
    eigenvalue_error = abs(computed_eigenvalues[i] - exact_eigenvalues[i])
    println(
        "mode $i: ω² ≈ $(round(computed_eigenvalues[i]; digits=4)) " *
        "(exact $(round(exact_eigenvalues[i]; digits=4)), error $(round(eigenvalue_error; sigdigits=2)))",
    )
end

# Because the discretisation is structure-preserving, the computed spectrum is free of the
# spurious modes that appear in naive (primal) discretisations, and the eigenvalues converge to
# the exact ones as the mesh is refined. The [Adaptive refinement](@ref) example revisits
# this same problem, but drives the refinement adaptively with hierarchical B-splines.
