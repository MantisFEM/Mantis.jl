# # Incompressible Navier–Stokes equations in 2D

# ## [Introduction](@id NS2DIntroduction)
# In this example we build a solver for the two-dimensional incompressible Navier–Stokes
# equations on a doubly periodic domain. The discretization follows the *mass-, kinetic
# energy- and helicity-conserving* formulation of [Zhang2022](@cite), which is written with
# vector calculus in 3D. We restate it in the language of differential forms, the language
# `Mantis` is built on, and restrict it to 2D. Along the way we will:
#
# 1. build a periodic **discrete de Rham complex** from periodic B-splines;
# 2. translate every term of the Navier–Stokes equations into differential forms, including
#    the nonlinear term, which becomes an **interior product** (a building block of the Lie
#    derivative);
# 3. discretize in time with the **implicit midpoint rule** and solve the resulting
#    nonlinear system with a **Newton–Raphson** method whose Jacobian we derive by hand;
# 4. **verify** the implementation with the Taylor–Green vortex, which has an analytical
#    solution, and check that the fully discrete scheme conserves kinetic energy and
#    enstrophy (or dissipates them at exactly the right rates);
# 5. simulate the **roll-up of a double shear layer**.
#
# We assume some familiarity with the finite element method, differential forms (exterior
# derivative ``\mathrm{d}``, Hodge star ``\star``, wedge product ``\wedge``) and with the
# basic `Mantis` workflow; see, for instance, the [Hodge Laplacian](HodgeLaplacian.md)
# example.

# ## [The Navier–Stokes equations in rotational form](@id NS2DRotationalForm)
# Consider a periodic domain ``\Omega = [0, 2\pi)^2``. Given an initial velocity field
# ``\boldsymbol{u}_0``, the (dimensionless) incompressible Navier–Stokes equations ask for a
# velocity ``\boldsymbol{u}`` and a pressure ``p`` such that
# ```math
# \frac{\partial \boldsymbol{u}}{\partial t} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u}
# - \nu \Delta \boldsymbol{u} + \nabla p = \boldsymbol{f}\,, \qquad
# \nabla\cdot\boldsymbol{u} = 0\,,
# ```
# with ``\nu = 1/\mathrm{Re}`` the kinematic viscosity and ``\boldsymbol{f}`` a body force
# (zero in this example). Several analytically equivalent forms of the convective term
# exist, and at the discrete level they lead to schemes with very different properties.
# Following [Zhang2022](@cite), we use the **rotational (or Lamb) form**
# ```math
# (\boldsymbol{u}\cdot\nabla)\boldsymbol{u} = \boldsymbol{\omega}\times\boldsymbol{u}
# + \tfrac{1}{2}\nabla|\boldsymbol{u}|^2\,, \qquad
# \boldsymbol{\omega} = \nabla\times\boldsymbol{u}\,,
# ```
# absorb the kinetic energy term into the **total (Bernoulli) pressure**
# ``P = p + \tfrac{1}{2}|\boldsymbol{u}|^2``, and write the viscous term as
# ``-\nu\Delta\boldsymbol{u} = \nu\nabla\times\boldsymbol{\omega}`` (valid because
# ``\nabla\cdot\boldsymbol{u}=0``). This gives
# ```math
# \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{\omega}\times\boldsymbol{u}
# + \nu\nabla\times\boldsymbol{\omega} + \nabla P = \boldsymbol{f}\,, \qquad
# \boldsymbol{\omega} = \nabla\times\boldsymbol{u}\,, \qquad
# \nabla\cdot\boldsymbol{u} = 0\,.
# ```
# In 2D the vorticity has a single component, ``\boldsymbol{\omega} = \omega\,\boldsymbol{e}_z``
# with ``\omega = \partial_x u_y - \partial_y u_x``, and
# ``\nabla\times\omega = (\partial_y\omega, -\partial_x\omega)``.
#
# In the absence of viscosity and forcing, these equations conserve (among other things)
# the total kinetic energy ``K = \tfrac{1}{2}\int_\Omega|\boldsymbol{u}|^2``. The proof
# only needs two facts: ``(\boldsymbol{\omega}\times\boldsymbol{u})\cdot\boldsymbol{u}=0``
# pointwise, and ``\int_\Omega\nabla P\cdot\boldsymbol{u} = -\int_\Omega P\,\nabla\cdot
# \boldsymbol{u} = 0``. With viscosity, ``\frac{\mathrm{d}K}{\mathrm{d}t} = -\nu\int_\Omega
# \omega^2 = -2\nu\mathcal{E}``, where ``\mathcal{E} = \tfrac{1}{2}\int_\Omega\omega^2`` is
# the enstrophy. The goal is a discretization in which *both facts remain true exactly*,
# so that the discrete kinetic energy behaves exactly like the continuous one.
#
# Kinetic energy is not the only invariant. In 3D, the formulation of [Zhang2022](@cite)
# also conserves helicity ``H = \int_\Omega\boldsymbol{u}\cdot\boldsymbol{\omega}``. In 2D,
# ``\boldsymbol{\omega}\perp\boldsymbol{u}``, so helicity is identically zero. Its role is
# taken by the enstrophy: both are Casimirs of the Hamiltonian structure of the Euler
# equations [Zhang2022](@cite).

# ## [From vector calculus to differential forms](@id NS2DForms)
# ### [The unknowns as differential forms](@id NS2DUnknowns)
# A vector field can be represented by a differential form in more than one way. The
# natural choice depends on which differential operator acts on it. Mass conservation,
# ``\nabla\cdot\boldsymbol{u} = 0``, is the constraint we want to satisfy exactly, so we
# represent velocity as a **flux 1-form** (the ``H(\mathrm{div})`` representation):
# ```math
# u^{1} := \star u^\flat = -u_y\,\mathrm{d}x + u_x\,\mathrm{d}y\,,
# \qquad \text{so that} \qquad
# \mathrm{d}u^{1} = (\partial_x u_x + \partial_y u_y)\,\mathrm{d}x\wedge\mathrm{d}y
# = (\nabla\cdot\boldsymbol{u})\,\mathrm{vol}\,.
# ```
# Here ``u^\flat = u_x\,\mathrm{d}x + u_y\,\mathrm{d}y`` is the circulation 1-form of
# ``\boldsymbol{u}``, and the integral of ``u^{1}`` over a curve is the volumetric flux
# through it. The remaining unknowns follow from the de Rham complex:
# - the **vorticity** is the 0-form ``w^{0} := \delta u^{1}``, where
#   ``\delta = (-1)^{n(k+1)+1}\star\mathrm{d}\star`` is the codifferential. Using
#   ``\star\star = -1`` on 1-forms in 2D, ``\delta u^1 = -\star\mathrm{d}\star u^1 =
#   \star\mathrm{d}u^\flat = \partial_x u_y - \partial_y u_x = \omega``;
# - the **total pressure** is the 2-form ``P^{2} := \star P = P\,\mathrm{vol}``.
#
# With this choice, all differential operators become ``\mathrm{d}`` or ``\delta``:
#
# | vector calculus | differential forms (flux representation) |
# |:--- |:--- |
# | ``\nabla\cdot\boldsymbol{u}`` | ``\mathrm{d}u^{1}`` |
# | ``\omega = \nabla\times\boldsymbol{u}`` | ``w^{0} = \delta u^{1}`` |
# | ``\nabla\times\omega`` | ``\mathrm{d}w^{0}`` |
# | ``\nabla P`` | ``-\delta P^{2}`` |
#
# The one term not covered by this table is the nonlinear term
# ``\boldsymbol{\omega}\times\boldsymbol{u}``, which we treat next.

# ### [The nonlinear term: an interior product](@id NS2DLamb)
# The convective term is where differential geometry pays off. In the circulation
# representation, the convective derivative is related to the **Lie derivative**
# ``\mathcal{L}_{\boldsymbol{u}}`` through
# ```math
# \big((\boldsymbol{u}\cdot\nabla)\boldsymbol{u}\big)^\flat
# = \mathcal{L}_{\boldsymbol{u}}u^\flat - \tfrac{1}{2}\mathrm{d}|\boldsymbol{u}|^2
# = \iota_{\boldsymbol{u}}\mathrm{d}u^\flat + \tfrac{1}{2}\mathrm{d}|\boldsymbol{u}|^2\,,
# ```
# where we used Cartan's formula
# ``\mathcal{L}_{\boldsymbol{u}} = \iota_{\boldsymbol{u}}\mathrm{d} + \mathrm{d}\iota_{\boldsymbol{u}}``
# and ``\iota_{\boldsymbol{u}}u^\flat = |\boldsymbol{u}|^2``. `Mantis` does not implement
# the Lie derivative directly, but it does implement its building blocks. The exact term
# ``\tfrac{1}{2}\mathrm{d}|\boldsymbol{u}|^2`` goes into the total pressure. What remains is
# the interior product ``\iota_{\boldsymbol{u}}\mathrm{d}u^\flat``, which is precisely the
# Lamb vector ``(\boldsymbol{\omega}\times\boldsymbol{u})^\flat``.
#
# Written in terms of the unknown ``u^{1}``, this term is
# ``\iota_{(\star u^{1})^\sharp}\,\mathrm{d}\star u^{1}``: since ``\star u^{1} = -u^\flat``,
# both minus signs cancel. We want to express it with operations `Mantis` provides
# (``\star``, ``\wedge``). For a ``k``-form ``\alpha`` in ``n`` dimensions,
# [Hirani2003](@cite) gives
# ```math
# \iota_{X}\alpha = (-1)^{k(n-k)}\star(\star\alpha\wedge X^\flat)\,.
# ```
# Take ``X = (\star u^{1})^\sharp`` (so ``X^\flat = \star u^{1}``). For the 2-form
# ``\alpha = \star w^{0}`` (``k=n=2``, and ``\star\star w^{0} = w^{0}``) this reads
# ```math
# \iota_{(\star u^{1})^\sharp}\star w^{0} = \star\big(w^{0}\wedge\star u^{1}\big)\,.
# ```
# One sign needs care: ``\mathrm{d}\star u^{1} = -\mathrm{d}u^\flat = -\star w^{0}`` when
# ``w^{0}=\delta u^{1}`` is the physical vorticity. (Defining the vorticity instead as
# ``\star\mathrm{d}\star u^{1}`` flips its sign and removes the minus.) Hence the Lamb
# vector in circulation form is
# ``\iota_{(\star u^{1})^\sharp}\mathrm{d}\star u^{1} = -\star(w^{0}\wedge\star u^{1})``.
# Our momentum equation is written for the flux form ``u^{1} = \star u^\flat``, so we apply
# one more Hodge star (``\star\star=-1`` on 1-forms):
# ```math
# \big(\boldsymbol{\omega}\times\boldsymbol{u}\big)\;\longleftrightarrow\;
# \star\,\iota_{\boldsymbol{u}}\mathrm{d}u^\flat = w^{0}\wedge\star u^{1}\,.
# ```
# As a sanity check in components:
# ``w^{0}\wedge\star u^{1} = -\omega u_x\,\mathrm{d}x - \omega u_y\,\mathrm{d}y``. As a flux
# form, this is the vector ``(-\omega u_y, \omega u_x) = \omega\boldsymbol{e}_z\times
# \boldsymbol{u}``.

# ### [The equations in differential forms](@id NS2DFormEquations)
# Collecting everything, the rotational Navier–Stokes equations become: find
# ``(u^{1}, w^{0}, P^{2})`` such that
# ```math
# \begin{aligned}
# \frac{\partial u^{1}}{\partial t} + w^{0}\wedge\star u^{1} + \nu\,\mathrm{d}w^{0}
# - \delta P^{2} &= f^{1}\,,\\
# w^{0} &= \delta u^{1}\,,\\
# \mathrm{d}u^{1} &= 0\,.
# \end{aligned}
# ```

# ## [Weak formulation](@id NS2DWeakFormulation)
# We multiply by test forms and integrate, using the ``L^2`` inner product of
# ``k``-forms, ``(\alpha^k, \beta^k)_\Omega = \int_\Omega \alpha^k\wedge\star\beta^k``.
# Because the domain is periodic, ``\delta`` is the exact adjoint of ``\mathrm{d}``:
# ``(\delta P^{2}, v^{1})_\Omega = (P^{2}, \mathrm{d}v^{1})_\Omega`` and
# ``(\delta u^{1}, \xi^{0})_\Omega = (u^{1}, \mathrm{d}\xi^{0})_\Omega``. The weak
# formulation reads: find ``(u^{1}, w^{0}, P^{2}) \in H\Lambda^{1}\times H\Lambda^{0}\times
# L^2\Lambda^{2}`` such that
# ```math
# \begin{aligned}
# \Big(\frac{\partial u^{1}}{\partial t}, v^{1}\Big)_\Omega
# + \big(w^{0}\wedge\star u^{1}, v^{1}\big)_\Omega
# + \nu\big(\mathrm{d}w^{0}, v^{1}\big)_\Omega
# - \big(P^{2}, \mathrm{d}v^{1}\big)_\Omega &= \big(f^{1}, v^{1}\big)_\Omega
# &&\forall v^{1}\in H\Lambda^{1}\,,\\
# \big(w^{0}, \xi^{0}\big)_\Omega - \big(u^{1}, \mathrm{d}\xi^{0}\big)_\Omega &= 0
# &&\forall \xi^{0}\in H\Lambda^{0}\,,\\
# \big(\mathrm{d}u^{1}, q^{2}\big)_\Omega &= 0 &&\forall q^{2}\in L^2\Lambda^{2}\,.
# \end{aligned}
# ```
# This is the 2D version of the ``(u_2, \omega_1, P_3)`` subsystem, equations (13d)–(13f),
# of [Zhang2022](@cite). In 2D, ``H\Lambda^1 = H(\mathrm{div})``,
# ``H\Lambda^0 = H^1`` and ``L^2\Lambda^2 = L^2``.
#
# The convective term has a particularly telling structure:
# ```math
# \big(w^{0}\wedge\star u^{1}, v^{1}\big)_\Omega
# = \int_\Omega v^{1}\wedge\star\big(w^{0}\wedge\star u^{1}\big)
# = \int_\Omega w^{0}\wedge u^{1}\wedge v^{1}\,.
# ```
# Because ``u^{1}\wedge v^{1} = -v^{1}\wedge u^{1}``, it is **antisymmetric** in
# ``(u^{1}, v^{1})`` for *any* ``w^{0}``. This is the discrete counterpart of
# ``(\boldsymbol{\omega}\times\boldsymbol{u})\cdot\boldsymbol{u} = 0``. It holds pointwise,
# independently of the basis functions and even of the quadrature rule.

# ## [Discretization in space: a periodic de Rham complex](@id NS2DComplex)
# The conservation properties rely on the discrete spaces forming a **discrete de Rham
# complex**:
# ```math
# \mathbb{R} \hookrightarrow \Lambda^{0}_h \xrightarrow{\ \mathrm{d}\ } \Lambda^{1}_h
# \xrightarrow{\ \mathrm{d}\ } \Lambda^{2}_h \rightarrow 0\,,
# ```
# that is, ``\mathrm{d}\Lambda^{0}_h\subset\Lambda^{1}_h`` and
# ``\mathrm{d}\Lambda^{1}_h\subset\Lambda^{2}_h``. The second inclusion is what makes the
# continuity equation hold *pointwise*: ``\mathrm{d}u^{1}_h \in \Lambda^{2}_h`` is orthogonal
# to all of ``\Lambda^{2}_h``, including itself, so ``\mathrm{d}u^{1}_h = 0`` exactly.
#
# With B-splines, such a complex is built from univariate spline spaces ``B`` of degree
# ``p`` and their derivative spaces ``D = \frac{\mathrm{d}}{\mathrm{d}x}B`` of degree
# ``p-1``. In 2D:
# ```math
# \Lambda^{0}_h = B_x\otimes B_y\,,\qquad
# \Lambda^{1}_h = (D_x\otimes B_y)\,\mathrm{d}x \;\oplus\; (B_x\otimes D_y)\,\mathrm{d}y\,,\qquad
# \Lambda^{2}_h = (D_x\otimes D_y)\,\mathrm{d}x\wedge\mathrm{d}y\,.
# ```
# For a periodic domain we use *periodic* univariate splines. In `Mantis` these are
# `GTBSplineSpace`s built from a single B-spline patch glued to itself with ``C^{p-1}``
# smoothness [Hiemstra2020](@cite). Their derivative spaces are again periodic splines, of
# one degree lower.
#
# !!! note "The flux interpretation of the 1-forms"
#     The ``\mathrm{d}y`` component of ``u^1`` is the horizontal velocity ``u_x``. It
#     lives in ``B_x\otimes D_y``: continuous in ``x``, the direction in which it is a
#     normal flux, and one degree lower in ``y``. This is exactly the structure of a
#     Raviart–Thomas-type ``H(\mathrm{div})`` space.

# ## [Implementation](@id NS2DImplementation)
# ### [Packages](@id NS2DPackages)
# Besides `Mantis`, we use `GLMakie` for plotting, and the standard libraries
# `LinearAlgebra`, `SparseArrays` and `Printf` for the linear algebra and printing.

using Mantis
using GLMakie
using DisplayAs #hide
using LinearAlgebra
using SparseArrays
using Printf

# ### [The periodic de Rham complex](@id NS2DComplexCode)
# The function below builds ``\Lambda^0_h, \Lambda^1_h, \Lambda^2_h`` exactly as described
# above. The order of the components of the 1-form space (first ``\mathrm{d}x``, then
# ``\mathrm{d}y``) is the order `Mantis` uses for 1-forms.

function create_periodic_de_rham_complex(
    starting_point::NTuple{2, Float64},
    box_size::NTuple{2, Float64},
    num_elements::NTuple{2, Int},
    degree::NTuple{2, Int},
)
    ## Univariate periodic splines of degree p and maximal smoothness C^{p-1}: an open
    ## B-spline space whose two ends are glued together with C^{p-1} continuity.
    B = ntuple(2) do i
        open_space = FunctionSpaces.create_bspline_space(
            starting_point[i], box_size[i], num_elements[i], degree[i], degree[i] - 1
        )
        return FunctionSpaces.GTBSplineSpace((open_space,), [degree[i] - 1])
    end
    ## Their derivative spaces: periodic splines of degree p-1 and smoothness C^{p-2}.
    D = map(FunctionSpaces.get_derivative_space, B)

    Λ⁰ = Forms.FormSpace(0, FunctionSpaces.TensorProductSpace((B[1], B[2])), "ξ⁰")
    Λ¹ = Forms.FormSpace(
        1,
        FunctionSpaces.DirectSumSpace((
            FunctionSpaces.TensorProductSpace((D[1], B[2])), # dx-component
            FunctionSpaces.TensorProductSpace((B[1], D[2])), # dy-component
        )),
        "v¹",
    )
    Λ² = Forms.FormSpace(2, FunctionSpaces.TensorProductSpace((D[1], D[2])), "q²")

    return Λ⁰, Λ¹, Λ²
end

# ### [Assembling operators](@id NS2DOperators)
# All matrices below are assembled with the standard `Mantis` workflow: define the test and
# trial spaces, write the integrand with forms, and call `Assemblers.assemble`. The helper
# takes the integrand as a function of the test and trial forms (in that order; `Mantis`
# uses the first basis in an expression for the rows and the second for the columns).

function assemble_operator(test_space, trial_space, integrand)
    weak_form_inputs = Assemblers.WeakFormInputs(test_space, trial_space)
    test_form = Assemblers.get_test_form(weak_form_inputs)
    trial_form = Assemblers.get_trial_form(weak_form_inputs)
    lhs_expressions = ((integrand(test_form, trial_form),),)
    rhs_expressions = ((0,),) # No right-hand side needed.
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    matrix, _ = Assemblers.assemble(weak_form)

    return matrix
end

# Denote the bases of ``\Lambda^0_h, \Lambda^1_h, \Lambda^2_h`` by
# ``\{\psi_i\}, \{\varphi_i\}, \{\chi_i\}``. The linear operators of the weak formulation
# are the mass matrices ``\mathsf{M}^0_{ij} = (\psi_j, \psi_i)_\Omega`` and
# ``\mathsf{M}^1_{ij} = (\varphi_j, \varphi_i)_\Omega``, and the two operators
# ```math
# \mathsf{C}_{ij} = (\mathrm{d}\psi_j, \varphi_i)_\Omega\,, \qquad
# \mathsf{D}_{ij} = (\mathrm{d}\varphi_j, \chi_i)_\Omega\,,
# ```
# the discrete "curl" (``\mathrm{d}: \Lambda^0_h\to\Lambda^1_h``) and "divergence"
# (``\mathrm{d}: \Lambda^1_h\to\Lambda^2_h``). We use the same names as eq. (46) of
# [Zhang2022](@cite).
#
# The nonlinear term needs two more matrices, which change during the simulation. Given a
# vorticity field ``w_h`` and a velocity field ``u_h``, let
# ```math
# \mathsf{R}(w_h)_{ij} = (w_h\wedge\star\varphi_j, \varphi_i)_\Omega\,, \qquad
# \mathsf{Q}(u_h)_{ij} = (\psi_j\wedge\star u_h, \varphi_i)_\Omega\,.
# ```
# The convective term ``(w_h\wedge\star u_h, \varphi_i)_\Omega`` is *bilinear* in
# ``(u_h, w_h)``. It can therefore be written as ``\mathsf{R}(w_h)\,\mathbf{u} =
# \mathsf{Q}(u_h)\,\mathbf{w}``, with bold symbols denoting coefficient vectors.
# By the antisymmetry noted above, ``\mathsf{R}(w_h)`` is a skew-symmetric matrix.
#
# ### [The pressure gauge](@id NS2DGauge)
# On a periodic domain, ``\int_\Omega\mathrm{d}u^{1}_h = 0`` for every ``u^{1}_h``, so the
# constant 2-form is orthogonal to ``\mathrm{d}\Lambda^1_h``. The pressure enters the weak
# formulation only through ``(P^2, \mathrm{d}v^1)_\Omega``, so it is determined only up to
# a constant. Correspondingly, one row of the continuity equation
# ``\mathsf{D}\mathbf{u} = 0`` is a linear combination of the others. We therefore
# replace the first continuity equation by ``P_1 = 0``: the matrix `D_gauged` is
# ``\mathsf{D}`` with its first row set to zero, and the matrix `gauge` has a single entry
# ``1`` at position ``(1, 1)``, so that the continuity rows read
# `D_gauged * u + gauge * P = 0`. The constraint ``\mathrm{d}u^1_h = 0`` still holds exactly.
# Whenever we compare pressures, we compare them up to their mean.
#
# ### [Setting up the problem](@id NS2DSetup)
# In `Mantis`, the integrand ``v^{1}\wedge\star(w^{0}_h\wedge\star u^{1})`` is written
# literally as `v¹ ∧ ★(w⁰ₕ ∧ ★(u¹))`. When one factor is a `FormField` (a known field) and
# the other a `FormSpace` (a basis), the result is again a (bi)linear form in the basis
# functions. The first function below assembles everything that does not change in time
# and collects it in a `NamedTuple`. The second assembles the two convection matrices for
# given fields.

function setup_navier_stokes(
    num_elements::NTuple{2, Int}, degree::NTuple{2, Int}; box_size=(2π, 2π)
)
    Λ⁰, Λ¹, Λ² = create_periodic_de_rham_complex((0.0, 0.0), box_size, num_elements, degree)
    geometry = Forms.get_geometry(Λ⁰)

    ## The convective integrand has degree at most 3p per direction; Gauss–Legendre with
    ## ⌊3p/2⌋ + 1 points per direction integrates it exactly.
    canonical_qrule = Quadrature.tensor_product_rule(
        3 .* degree .÷ 2 .+ 1, Quadrature.gauss_legendre
    )
    dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

    M⁰ = assemble_operator(Λ⁰, Λ⁰, (ξ⁰, w⁰) -> ∫(ξ⁰ ∧ ★(w⁰), dΩ))
    M¹ = assemble_operator(Λ¹, Λ¹, (v¹, u¹) -> ∫(v¹ ∧ ★(u¹), dΩ))
    C = assemble_operator(Λ¹, Λ⁰, (v¹, w⁰) -> ∫(v¹ ∧ ★(d(w⁰)), dΩ))
    D = assemble_operator(Λ², Λ¹, (q², u¹) -> ∫(q² ∧ ★(d(u¹)), dΩ))

    ## The pressure gauge.
    num_P = Forms.get_num_basis(Λ²)
    D_gauged = copy(D)
    D_gauged[1, :] .= 0.0
    dropzeros!(D_gauged)
    gauge = sparse([1], [1], [1.0], num_P, num_P)

    return (;
        Λ⁰,
        Λ¹,
        Λ²,
        geometry,
        dΩ,
        M⁰,
        M¹,
        C,
        D,
        D_gauged,
        gauge,
        box_size,
        num_elements,
        num_w=Forms.get_num_basis(Λ⁰),
        num_u=Forms.get_num_basis(Λ¹),
        num_P,
    )
end

function convection_operators(problem, u¹ₕ, w⁰ₕ)
    (; Λ⁰, Λ¹, dΩ) = problem
    R = assemble_operator(Λ¹, Λ¹, (v¹, u¹) -> ∫(v¹ ∧ ★(w⁰ₕ ∧ ★(u¹)), dΩ))
    Q = assemble_operator(Λ¹, Λ⁰, (v¹, w⁰) -> ∫(v¹ ∧ ★(w⁰ ∧ ★(u¹ₕ)), dΩ))

    return R, Q
end

# Before using these operators, let us check that the complex behaves as promised on a
# small mesh. The dimensions satisfy
# ``\dim\Lambda^0_h - \dim\Lambda^1_h + \dim\Lambda^2_h = 0``, the Euler characteristic of
# the torus. The discrete ``\mathrm{d}\circ\mathrm{d} = 0``: the coefficients of
# ``\mathrm{d}\psi_j`` in ``\Lambda^1_h`` are the columns of
# ``(\mathsf{M}^1)^{-1}\mathsf{C}``, and applying ``\mathsf{D}`` to them must give zero.
# Finally, ``\mathsf{R}(w_h)`` is skew-symmetric and
# ``\mathsf{R}(w_h)\mathbf{u} = \mathsf{Q}(u_h)\mathbf{w}``.

test_problem = setup_navier_stokes((4, 4), (2, 2))
dims = (test_problem.num_w, test_problem.num_u, test_problem.num_P)
println(
    "dim Λ⁰ₕ, Λ¹ₕ, Λ²ₕ = ", dims, ",   Euler characteristic = ", dims[1] - dims[2] + dims[3]
)
println(
    "‖D (M¹)⁻¹ C‖ = ", norm(test_problem.D * (test_problem.M¹ \ Matrix(test_problem.C)))
)

w_test = Forms.FormField(test_problem.Λ⁰, randn(test_problem.num_w), "w")
u_test = Forms.FormField(test_problem.Λ¹, randn(test_problem.num_u), "u")
R_test, Q_test = convection_operators(test_problem, u_test, w_test)
println("‖R + Rᵀ‖ / ‖R‖ = ", norm(R_test + R_test') / norm(R_test))
println(
    "‖R(w)u - Q(u)w‖ = ", norm(R_test * u_test.coefficients - Q_test * w_test.coefficients)
)

# ## [Discretization in time: the implicit midpoint rule](@id NS2DTime)
# As in [Zhang2022](@cite), we use the implicit midpoint rule (the lowest-order Gauss
# integrator). With ``\Delta t`` the time step,
# ``\mathbf{u}^{n+\frac{1}{2}} = \tfrac{1}{2}(\mathbf{u}^{n+1}+\mathbf{u}^{n})`` and
# similarly for ``\mathbf{w}``, one step from ``t^n`` to ``t^{n+1}`` requires solving
# ``\mathbf{F}(\mathbf{x}) = \mathbf{0}`` for
# ``\mathbf{x} = (\mathbf{u}^{n+1}, \mathbf{w}^{n+1}, \mathbf{P}^{n+\frac{1}{2}})``, with
# ```math
# \mathbf{F}(\mathbf{x}) =
# \begin{bmatrix}
# \mathsf{M}^1\dfrac{\mathbf{u}^{n+1}-\mathbf{u}^{n}}{\Delta t}
# + \mathsf{R}(w^{n+\frac{1}{2}}_h)\,\mathbf{u}^{n+\frac{1}{2}}
# + \nu\,\mathsf{C}\,\mathbf{w}^{n+\frac{1}{2}} - \mathsf{D}^{\mathsf{T}}\mathbf{P}^{n+\frac{1}{2}}\\[2mm]
# \mathsf{M}^0\,\mathbf{w}^{n+1} - \mathsf{C}^{\mathsf{T}}\mathbf{u}^{n+1}\\[1mm]
# \mathsf{D}\,\mathbf{u}^{n+1}
# \end{bmatrix}.
# ```
# The pressure is a Lagrange multiplier for the divergence constraint and naturally lives
# at the half step.
#
# ### [Conservation properties of the fully discrete scheme](@id NS2DConservation)
# **Mass.** The last block row states ``(\mathrm{d}u^{n+1}_h, q^2)_\Omega = 0`` for all
# ``q^2\in\Lambda^2_h``. Since ``\mathrm{d}u^{n+1}_h\in\Lambda^2_h``, we may take
# ``q^2 = \mathrm{d}u^{n+1}_h``, which gives ``\mathrm{d}u^{n+1}_h = 0`` *pointwise*.
#
# **Kinetic energy.** Take the inner product of the first block row with
# ``\mathbf{u}^{n+\frac{1}{2}}``.
# - The time derivative gives ``(K^{n+1}-K^{n})/\Delta t``, with
#   ``K = \tfrac{1}{2}\mathbf{u}^{\mathsf{T}}\mathsf{M}^1\mathbf{u}``.
# - The convective term vanishes because ``\mathsf{R}`` is skew-symmetric.
# - The pressure term vanishes because
#   ``\mathsf{D}\mathbf{u}^{n} = \mathsf{D}\mathbf{u}^{n+1} = 0``.
# - By the second row (which holds at ``n`` and ``n+1``, hence at ``n+\frac{1}{2}``), the
#   viscous term equals ``\nu\,(\mathbf{w}^{n+\frac{1}{2}})^{\mathsf{T}}\mathsf{M}^0
#   \mathbf{w}^{n+\frac{1}{2}}``.
#
# Therefore
# ```math
# \frac{K^{n+1}-K^{n}}{\Delta t} = -\nu\,\big\|w^{n+\frac{1}{2}}_h\big\|^2_{L^2}\,,
# ```
# the exact discrete analogue of ``\frac{\mathrm{d}K}{\mathrm{d}t} = -2\nu\mathcal{E}``. For
# ``\nu=0`` the kinetic energy is conserved exactly, for any mesh and any time step. Since
# the skew-symmetry of ``\mathsf{R}`` holds pointwise, this is even independent of the
# quadrature rule.
#
# **Enstrophy (the 2D counterpart of helicity).** The vorticity equation is linear, so
# ``\mathsf{M}^0(\mathbf{w}^{n+1}-\mathbf{w}^{n}) = \mathsf{C}^{\mathsf{T}}(\mathbf{u}^{n+1}-\mathbf{u}^{n})``.
# Hence, with
# ``\mathcal{E} = \tfrac{1}{2}\mathbf{w}^{\mathsf{T}}\mathsf{M}^0\mathbf{w}``,
# ```math
# \mathcal{E}^{n+1}-\mathcal{E}^{n}
# = (\mathbf{w}^{n+\frac{1}{2}})^{\mathsf{T}}\mathsf{M}^0(\mathbf{w}^{n+1}-\mathbf{w}^{n})
# = \big(\mathrm{d}w^{n+\frac{1}{2}}_h,\, u^{n+1}_h - u^{n}_h\big)_\Omega\,.
# ```
# Because the complex is exact, ``\mathrm{d}w^{n+\frac{1}{2}}_h`` belongs to
# ``\Lambda^1_h`` and is a valid test function in the momentum equation. Testing with it:
# - the pressure term vanishes because ``\mathrm{d}\mathrm{d} = 0``;
# - the convective term becomes ``\int_\Omega w\, u\wedge\mathrm{d}w = \int_\Omega u\wedge
#   \mathrm{d}\big(\tfrac{1}{2}w^2\big) = \int_\Omega \tfrac{1}{2}w^2\,\mathrm{d}u = 0``,
#   where we integrated by parts on the periodic domain and used ``\mathrm{d}u^{n+
#   \frac{1}{2}}_h = 0`` *pointwise*.
#
# What remains is
# ```math
# \frac{\mathcal{E}^{n+1}-\mathcal{E}^{n}}{\Delta t}
# = -\nu\,\big\|\mathrm{d}w^{n+\frac{1}{2}}_h\big\|^2_{L^2}\,,
# ```
# the discrete analogue of ``\frac{\mathrm{d}\mathcal{E}}{\mathrm{d}t} = -\nu\int_\Omega
# |\nabla\omega|^2``. For ``\nu = 0`` the enstrophy is conserved. This argument, unlike the
# energy argument, needs the integrals to be computed exactly. This is the case here: the
# mesh is Cartesian and the quadrature rule is exact for the polynomial integrands. The
# argument mirrors the helicity proof of [Zhang2022](@cite), with enstrophy in place of
# helicity. All three properties rest on the discrete de Rham complex.
#
# **Nonlinear vs. linear.** [Zhang2022](@cite) solve two such subsystems, one for each
# representation of the velocity, staggered in time: each borrows the vorticity from the
# other, so both are linear. Here we solve only the ``H(\mathrm{div})`` subsystem. The
# vorticity in the convective term is then an unknown, and the system is nonlinear. We solve
# it with Newton–Raphson.

# ## [Solving the nonlinear system: Newton–Raphson](@id NS2DNewton)
# Given an iterate ``\mathbf{x}_k``, Newton's method solves
# ``\mathsf{J}(\mathbf{x}_k)\,\Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k)`` and sets
# ``\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x}``. The only nonlinear term is
# ``\mathsf{R}(w^{n+\frac{1}{2}}_h)\mathbf{u}^{n+\frac{1}{2}}``. It is bilinear, so its
# derivative with respect to ``\mathbf{u}^{n+1}`` is ``\tfrac{1}{2}\mathsf{R}(w^{n+\frac{1}{2}}_h)``
# (the ``\tfrac{1}{2}`` comes from ``\partial\mathbf{u}^{n+\frac{1}{2}}/\partial
# \mathbf{u}^{n+1}``), and with respect to ``\mathbf{w}^{n+1}`` it is
# ``\tfrac{1}{2}\mathsf{Q}(u^{n+\frac{1}{2}}_h)``. The Jacobian is
# ```math
# \mathsf{J} =
# \begin{bmatrix}
# \dfrac{1}{\Delta t}\mathsf{M}^1 + \dfrac{1}{2}\mathsf{R}(w^{n+\frac{1}{2}}_h)
# & \dfrac{1}{2}\mathsf{Q}(u^{n+\frac{1}{2}}_h) + \dfrac{\nu}{2}\mathsf{C}
# & -\mathsf{D}^{\mathsf{T}}\\[2mm]
# -\mathsf{C}^{\mathsf{T}} & \mathsf{M}^0 & 0\\[1mm]
# \mathsf{D} & 0 & 0
# \end{bmatrix}.
# ```
# Compare with the (linear) systems of [Zhang2022](@cite), eq. (46). The new ingredient is
# the block ``\tfrac{1}{2}\mathsf{Q}``, which couples the momentum equation to the unknown
# vorticity. The solution at the previous time step is an excellent initial guess, so Newton
# typically converges quadratically in two to four iterations. In the code, the last block
# row uses the gauged continuity equation of [The pressure gauge](@ref NS2DGauge).
#
function midpoint_newton_step(
    problem, uⁿ, wⁿ, P_guess, Δt, ν; tolerance=1e-11, max_iterations=15, verbose=false
)
    (; Λ⁰, Λ¹, M⁰, M¹, C, D, D_gauged, gauge, num_u, num_w, num_P) = problem
    iu, iw, iP = 1:num_u, num_u .+ (1:num_w), (num_u + num_w) .+ (1:num_P)

    x = [uⁿ; wⁿ; P_guess] # initial guess: the previous time level
    residual_norms = Float64[]
    for iteration in 1:max_iterations
        uⁿ⁺¹, wⁿ⁺¹, P = x[iu], x[iw], x[iP]
        u_mid = 0.5 .* (uⁿ⁺¹ .+ uⁿ)
        w_mid = 0.5 .* (wⁿ⁺¹ .+ wⁿ)
        R, Q = convection_operators(
            problem, Forms.FormField(Λ¹, u_mid, "u"), Forms.FormField(Λ⁰, w_mid, "w")
        )

        ## Residual: momentum, vorticity and (gauged) continuity equations.
        F_u = M¹ * (uⁿ⁺¹ .- uⁿ) ./ Δt .+ R * u_mid .+ ν .* (C * w_mid) .- D' * P
        F_w = M⁰ * wⁿ⁺¹ .- C' * uⁿ⁺¹
        F_P = D_gauged * uⁿ⁺¹ .+ gauge * P
        F = [F_u; F_w; F_P]
        push!(residual_norms, norm(F))
        converged = residual_norms[end] < tolerance
        if verbose
            @printf(
                "    Newton iteration %2d: ‖F‖ = %.3e%s\n",
                iteration - 1,
                residual_norms[end],
                converged ? "  (converged)" : "",
            )
        end
        if converged
            break
        end

        ## Jacobian: the only nonlinear blocks are ∂F_u/∂u and ∂F_u/∂w.
        J_uu = M¹ ./ Δt .+ 0.5 .* R
        J_uw = 0.5 .* Q .+ (0.5 * ν) .* C
        ## Block columns are ordered as the unknowns: (u, w, P).
        J = [
            J_uu J_uw -D'
            -C' M⁰ spzeros(num_w, num_P)
            D_gauged spzeros(num_P, num_w) gauge
        ]
        x .-= J \ F
    end
    if residual_norms[end] ≥ tolerance
        @warn "Newton did not converge" residual_norms
    end

    return x[iu], x[iw], x[iP], residual_norms
end

# ### [Initial conditions: a discrete Leray projection](@id NS2DInitialConditions)
# The initial velocity must be divergence free *at the discrete level*, or the first time
# step would have to absorb the difference. We therefore project the given velocity
# ``u^1_0`` onto the discretely divergence-free subspace of ``\Lambda^1_h``: find
# ``(u^1_h, \phi^2_h)`` such that
# ``(u^1_h, v^1)_\Omega - (\phi^2_h, \mathrm{d}v^1)_\Omega = (u^1_0, v^1)_\Omega`` and
# ``(\mathrm{d}u^1_h, q^2)_\Omega = 0``. The right-hand side is exactly that of an ``L^2``
# projection, for which `Mantis` provides `Assemblers.L2_projection`. The initial vorticity
# then follows from the second equation of the weak formulation:
# ``\mathsf{M}^0\mathbf{w} = \mathsf{C}^{\mathsf{T}}\mathbf{u}``.
#
# Analytical 1-forms are given to `Mantis` by their components in the ``(\mathrm{d}x,
# \mathrm{d}y)`` basis. For the flux form of a velocity ``(u_x, u_y)`` these are
# ``(-u_y, u_x)``.

function project_velocity(problem, velocity::Function)
    (; Λ¹, geometry, dΩ, M¹, D, D_gauged, gauge, num_P) = problem
    function flux_form(x)
        u_x, u_y = velocity(x)
        return [-u_y, u_x]
    end
    u¹₀ = Forms.AnalyticalFormField(1, flux_form, geometry, "u₀")

    weak_form_inputs = Assemblers.WeakFormInputs(Λ¹, u¹₀)
    lhs_expressions, rhs_expressions = Assemblers.L2_projection(weak_form_inputs, dΩ)
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    _, b = Assemblers.assemble(weak_form)

    A = [M¹ -D'; D_gauged gauge]
    x = A \ [b; zeros(num_P)]

    return x[1:problem.num_u]
end

compute_vorticity(problem, u) = problem.M⁰ \ (problem.C' * u)

# ### [Diagnostics](@id NS2DDiagnostics)
# We monitor the kinetic energy ``K = \tfrac{1}{2}(u^1_h, u^1_h)_\Omega``, the enstrophy
# ``\mathcal{E} = \tfrac{1}{2}(w^0_h, w^0_h)_\Omega``, and the ``L^2`` norm of
# ``\mathrm{d}u^1_h`` (the divergence). The divergence is computed by applying `d` to the
# `FormField` itself, independently of the matrices. The term
# ``\|\mathrm{d}w^0_h\|^2`` of the enstrophy balance is computed in the same way.

kinetic_energy(problem, u) = 0.5 * dot(u, problem.M¹ * u)
enstrophy(problem, w) = 0.5 * dot(w, problem.M⁰ * w)
function vorticity_gradient_squared(problem, w)
    return Analysis.L2_norm(d(Forms.FormField(problem.Λ⁰, w, "w")), problem.dΩ)^2
end
function divergence_norm(problem, u)
    return Analysis.L2_norm(d(Forms.FormField(problem.Λ¹, u, "u")), problem.dΩ)
end

# ### [The time loop](@id NS2DTimeLoop)
# Finally, a small driver advances the solution and records the diagnostics. It also
# records the residuals of the two discrete balance laws derived above,
# ```math
# \frac{K^{n+1}-K^n}{\Delta t} + \nu\|w^{n+\frac{1}{2}}_h\|^2
# \qquad\text{and}\qquad
# \frac{\mathcal{E}^{n+1}-\mathcal{E}^n}{\Delta t} + \nu\|\mathrm{d}w^{n+\frac{1}{2}}_h\|^2\,,
# ```
# which should both vanish up to the Newton tolerance. Optionally, the driver stores copies
# of the solution at requested time steps.
#
# To follow a simulation, set `verbose = true`. At every time step the driver then prints
# the step number and time, followed by the residual ``\|\mathbf{F}\|`` of each Newton
# iteration (iteration 0 is the residual of the initial guess). With
# `show_conservation = true`, it also prints, after each step, the relative change of the
# kinetic energy and the enstrophy since ``t = 0`` and the residuals of the two discrete
# balance laws.

function print_conservation(history)
    K₀, ℰ₀ = history.kinetic_energy[1], history.enstrophy[1]
    @printf(
        "    ΔK/K₀ = %+.3e   Δℰ/ℰ₀ = %+.3e   K balance = %+.3e   ℰ balance = %+.3e\n",
        (history.kinetic_energy[end] - K₀) / K₀,
        (history.enstrophy[end] - ℰ₀) / ℰ₀,
        history.energy_balance[end],
        history.enstrophy_balance[end],
    )
    return nothing
end

function run_navier_stokes(
    problem,
    velocity::Function,
    ν,
    Δt,
    num_steps;
    snapshot_steps=Int[],
    verbose=false,
    show_conservation=false,
)
    u = project_velocity(problem, velocity)
    w = compute_vorticity(problem, u)
    P = zeros(problem.num_P)

    history = (
        t=[0.0],
        kinetic_energy=[kinetic_energy(problem, u)],
        enstrophy=[enstrophy(problem, w)],
        energy_balance=Float64[],
        enstrophy_balance=Float64[],
        newton_residuals=Vector{Float64}[],
    )
    snapshots = Dict{Int, NamedTuple}()
    0 in snapshot_steps && (snapshots[0] = (; t=0.0, u=copy(u), w=copy(w), P=copy(P)))
    for step in 1:num_steps
        if verbose
            @printf(
                "Time step %d/%d: t = %.4f → %.4f\n",
                step,
                num_steps,
                (step - 1) * Δt,
                step * Δt
            )
        end
        uⁿ⁺¹, wⁿ⁺¹, P, residuals = midpoint_newton_step(
            problem, u, w, P, Δt, ν; verbose=verbose
        )

        w_mid = 0.5 .* (w .+ wⁿ⁺¹)
        Kⁿ, Kⁿ⁺¹ = history.kinetic_energy[end], kinetic_energy(problem, uⁿ⁺¹)
        ℰⁿ, ℰⁿ⁺¹ = history.enstrophy[end], enstrophy(problem, wⁿ⁺¹)
        push!(history.energy_balance, (Kⁿ⁺¹ - Kⁿ) / Δt + 2ν * enstrophy(problem, w_mid))
        push!(
            history.enstrophy_balance,
            (ℰⁿ⁺¹ - ℰⁿ) / Δt + ν * vorticity_gradient_squared(problem, w_mid),
        )
        push!(history.t, step * Δt)
        push!(history.kinetic_energy, Kⁿ⁺¹)
        push!(history.enstrophy, ℰⁿ⁺¹)
        push!(history.newton_residuals, residuals)
        verbose && show_conservation && print_conservation(history)

        u, w = uⁿ⁺¹, wⁿ⁺¹
        if step in snapshot_steps
            snapshots[step] = (; t=step * Δt, u=copy(u), w=copy(w), P=copy(P))
        end
    end

    return u, w, P, history, snapshots
end

# ### [Plotting helpers](@id NS2DPlotting)
# To plot a field, we evaluate it on a uniform grid of points inside each element and
# assemble the element-wise values into one global array. Two operators from
# `Mantis.Forms` recover physical quantities. The Hodge star turns the 2-form ``P^2`` into
# the scalar ``P``. The sharp gives back the velocity vector:
# ``\boldsymbol{u} = (u^\flat)^\sharp = -(\star u^1)^\sharp``, which
# `Forms.evaluate_sharp_pushforward` evaluates in physical coordinates.

function sample_on_grid(problem, evaluate_values; num_points_per_element=6)
    (; geometry, box_size) = problem
    n = num_points_per_element
    ξ = collect(range(0.5 / n, 1 - 0.5 / n; length=n)) # cell-centred points in [0, 1]
    points = Points.TensorProductPoints((ξ, ξ))

    all_x, all_y, all_values = Float64[], Float64[], Float64[]
    for element_id in 1:Geometry.get_num_elements(geometry)
        x = Geometry.evaluate(geometry, element_id, points)
        append!(all_x, x[:, 1])
        append!(all_y, x[:, 2])
        append!(all_values, evaluate_values(element_id, points))
    end
    ## The points lie on a uniform grid: recover each point's grid index from its position.
    num_x, num_y = n .* problem.num_elements
    Δx, Δy = box_size[1] / num_x, box_size[2] / num_y
    grid = zeros(num_x, num_y)
    for (x, y, value) in zip(all_x, all_y, all_values)
        grid[floor(Int, x / Δx) + 1, floor(Int, y / Δy) + 1] = value
    end
    x_grid = collect(range(Δx / 2, box_size[1] - Δx / 2; length=num_x))
    y_grid = collect(range(Δy / 2, box_size[2] - Δy / 2; length=num_y))

    return x_grid, y_grid, grid
end

function sample_vorticity(problem, w; kwargs...)
    w⁰ₕ = Forms.FormField(problem.Λ⁰, w, "w")
    return sample_on_grid(
        problem,
        (element_id, points) -> vec(Forms.evaluate(w⁰ₕ, element_id, points)[1][1]);
        kwargs...,
    )
end

function sample_total_pressure(problem, P; kwargs...)
    P²ₕ = Forms.FormField(problem.Λ², P, "P")
    x, y, values = sample_on_grid(
        problem,
        (element_id, points) -> vec(Forms.evaluate(★(P²ₕ), element_id, points)[1][1]);
        kwargs...,
    )
    return x, y, values .- sum(values) / length(values) # remove the (arbitrary) mean
end

function sample_speed(problem, u; kwargs...)
    u¹ₕ = Forms.FormField(problem.Λ¹, u, "u")
    velocity = -★(u¹ₕ) # (u^♭) = -⋆u¹
    return sample_on_grid(
        problem,
        (element_id, points) -> begin
            u_vec, _ = Forms.evaluate_sharp_pushforward(velocity, element_id, points)
            return vec(sqrt.(u_vec[1] .^ 2 .+ u_vec[2] .^ 2))
        end;
        kwargs...,
    )
end

# ## [Verification: the Taylor–Green vortex](@id NS2DTaylorGreen)
# The 2D Taylor–Green vortex is an exact solution of the Navier–Stokes equations on
# ``[0, 2\pi)^2``:
# ```math
# \boldsymbol{u} = \big(\sin x\cos y,\, -\cos x\sin y\big)\,F(t)\,,\qquad
# \omega = 2\sin x\sin y\,F(t)\,,\qquad
# p = \tfrac{1}{4}(\cos 2x + \cos 2y)\,F(t)^2\,,
# ```
# with ``F(t) = e^{-2\nu t}``. The total pressure is
# ``P = p + \tfrac{1}{2}|\boldsymbol{u}|^2 = \tfrac{1}{4}\big(1 + \cos 2x + \cos 2y -
# \cos 2x\cos 2y\big)F(t)^2``. The nonlinear term does not vanish here; it is exactly
# balanced by the pressure gradient. In the rotational form, a sign error in the
# convective term could be compensated by a different total pressure. Comparing the
# computed *pressure* with the exact one therefore checks the convective term as well.
const ν_tg = 0.01
taylor_green_velocity(x) =
    [(@. sin(x[:, 1]) * cos(x[:, 2])), (@. -cos(x[:, 1]) * sin(x[:, 2]))]
decay(t) = exp(-2ν_tg * t)
exact_flux_form(t) =
    x -> [
        (@. cos(x[:, 1]) * sin(x[:, 2]) * decay(t)),
        (@. sin(x[:, 1]) * cos(x[:, 2]) * decay(t)),
    ]
exact_vorticity(t) = x -> [@. 2 * sin(x[:, 1]) * sin(x[:, 2]) * decay(t)]
exact_total_pressure(t) =
    x -> [
        @. 0.25 *
            (1 + cos(2x[:, 1]) + cos(2x[:, 2]) - cos(2x[:, 1]) * cos(2x[:, 2])) *
            decay(t)^2
    ]

# To measure the pressure error up to a constant, we subtract the mean of the error:
# ``\|e - \bar e\|^2 = \|e\|^2 - \big(\int_\Omega e\big)^2/|\Omega|``.

function integrate(form, dΩ)
    integral = ∫(form, dΩ)
    return sum(el -> Forms.evaluate(integral, el)[1][1], 1:Forms.get_num_elements(form))
end

function taylor_green_errors(num_elements, degree; Δt=0.05, T=1.0)
    problem = setup_navier_stokes(num_elements, degree)
    num_steps = round(Int, T / Δt)
    u, w, P, history, _ = run_navier_stokes(
        problem, taylor_green_velocity, ν_tg, Δt, num_steps
    )

    ## A more accurate quadrature rule for the error computation.
    qrule = Quadrature.tensor_product_rule(degree .+ 4, Quadrature.gauss_legendre)
    dΩₑ = Quadrature.StandardQuadrature(qrule, Geometry.get_num_elements(problem.geometry))
    geometry = problem.geometry

    u_error =
        Forms.FormField(problem.Λ¹, u, "u") -
        Forms.AnalyticalFormField(1, exact_flux_form(T), geometry, "u")
    w_error =
        Forms.FormField(problem.Λ⁰, w, "w") -
        Forms.AnalyticalFormField(0, exact_vorticity(T), geometry, "w")
    ## The pressure lives at the half step t = T - Δt/2.
    P_error =
        Forms.FormField(problem.Λ², P, "P") -
        Forms.AnalyticalFormField(2, exact_total_pressure(T - Δt / 2), geometry, "P")
    area = prod(problem.box_size)
    P_error_norm = sqrt(Analysis.L2_norm(P_error, dΩₑ)^2 - integrate(P_error, dΩₑ)^2 / area)

    return (
        u=Analysis.L2_norm(u_error, dΩₑ),
        w=Analysis.L2_norm(w_error, dΩₑ),
        P=P_error_norm,
        divergence=divergence_norm(problem, u),
        history=history,
    )
end

# ### [Newton convergence and discrete balance laws](@id NS2DTGNewton)
# We first run a single configuration and look at the Newton residuals of the first time
# step, and at the discrete energy and enstrophy balances over all steps.
N = 16
p = 3
Δt = 0.05
T = 1.0

@printf("========================================================== 
= Verification: Taylor-Green vortex                      =
==========================================================\n")

@printf(
    "\n==> Max energy and enstrophy balance
      N  = %d x %d
      p  = %d 
      Δt = %.2f 
      T  = %.1f\n",
    N,
    N,
    p,
    Δt,
    T
)

tg = taylor_green_errors((N, N), (p, p); Δt=Δt, T=T)
println("Newton residuals, first time step:")
foreach(r -> @printf("    %.3e\n", r), tg.history.newton_residuals[1])
@printf(
    "max |discrete energy balance|    = %.3e\n", maximum(abs, tg.history.energy_balance)
)
@printf(
    "max |discrete enstrophy balance| = %.3e\n", maximum(abs, tg.history.enstrophy_balance)
)
@printf("‖du‖ at the final time            = %.3e\n", tg.divergence)

# The residual drops quadratically, which confirms that the Jacobian is correct. Both
# discrete balance laws hold up to the Newton tolerance. The kinetic energy and the
# enstrophy therefore decay at exactly the rates dictated by the viscous terms, and the
# velocity is divergence free to machine precision.

# ### [Convergence study](@id NS2DTGConvergence)
# Next, we refine the mesh for splines of degree ``p = 2`` and ``p = 3``. The time step is
# small enough that the spatial error dominates. We expect the 1-form velocity and the
# 2-form pressure, whose spaces contain polynomials of degree ``p-1`` in some directions,
# to converge as ``\mathcal{O}(h^{p})``. The 0-form vorticity, of degree ``p``, should
# converge as ``\mathcal{O}(h^{p+1})``.
Δt = 0.05
T = 1.0

@printf(
    "\n==> Convergence tests
      Δt = %.2f 
      T  = %.1f\n",
    Δt,
    T
)

mesh_sizes = [4, 8, 16, 32]
errors = Dict(
    p => [taylor_green_errors((n, n), (p, p); Δt=Δt, T=T) for n in mesh_sizes] for
    p in (2, 3)
)

fig_convergence = Figure(; size=(1000, 380))
fields = ((:u, "velocity u¹"), (:w, "vorticity w⁰"), (:P, "total pressure P²"))
for (column, (field, label)) in enumerate(fields)
    ax = Axis(
        fig_convergence[1, column];
        title=label,
        xlabel="elements per direction N  (h = 2π/N)",
        ylabel="L² error",
        xscale=log2,
        yscale=log10,
        xticks=mesh_sizes,
    )
    for (p, color) in ((2, Makie.wong_colors()[1]), (3, Makie.wong_colors()[2]))
        e = [getfield(err, field) for err in errors[p]]
        scatterlines!(ax, mesh_sizes, e; color=color, label="p = $p")
        ## Reference slope, anchored at the finest mesh.
        rate = field == :w ? p + 1 : p
        reference = e[end] .* (mesh_sizes[end] ./ mesh_sizes) .^ rate
        lines!(
            ax,
            mesh_sizes,
            reference;
            color=color,
            linestyle=:dash,
            label=L"\mathcal{O}(h^{%$rate})",
        )
    end
    axislegend(ax; position=:lb)
end
display(GLMakie.Screen(), fig_convergence) #src
DisplayAs.Text(DisplayAs.PNG(fig_convergence)) #hide

# All fields converge at the expected rates; only the pressure errors on the coarsest mesh
# are still outside the asymptotic range. For completeness, here are the computed
# vorticity, speed ``|\boldsymbol{u}|`` and total pressure (with its mean removed) at
# ``t = 1`` on the ``16\times 16``, ``p=3`` mesh.
N = 16
p = 3
Δt = 0.05
T = 1.0

@printf(
    "\n==> Snapshots at t = 1
      N  = %d x %d
      p  = %d 
      Δt = %.2f 
      T  = %.1f",
    N,
    N,
    p,
    Δt,
    T
)

problem_tg = setup_navier_stokes((N, N), (p, p))
u_tg, w_tg, P_tg, _, _ = run_navier_stokes(
    problem_tg, taylor_green_velocity, ν_tg, Δt, round(Int, T / Δt)
)

fig_tg = Figure(; size=(1200, 360))
panels = (
    ("vorticity w⁰", sample_vorticity(problem_tg, w_tg), :balance),
    ("speed |u|", sample_speed(problem_tg, u_tg), :viridis),
    ("total pressure ⋆P²", sample_total_pressure(problem_tg, P_tg), :plasma),
)
for (column, (title, (x, y, values), colormap)) in enumerate(panels)
    ax = Axis(
        fig_tg[1, 2column - 1]; title=title, aspect=DataAspect(), xlabel="x", ylabel="y"
    )
    hm = heatmap!(ax, x, y, values; colormap=colormap)
    Colorbar(fig_tg[1, 2column], hm)
end
display(GLMakie.Screen(), fig_tg) #src
DisplayAs.Text(DisplayAs.PNG(fig_tg)) #hide

# ## [Application: roll-up of a double shear layer](@id NS2DShearLayer)
# A classical test for the robustness of incompressible flow solvers is the double shear
# layer [Minion1997](@cite), also used for the 2D predecessor of the present scheme
# [Palha2017](@cite). Two thin shear layers of thickness ``\delta`` are perturbed by a
# small vertical velocity:
# ```math
# u_x = \begin{cases}\tanh\big((y - \pi/2)/\delta\big) & y\le\pi\\
# \tanh\big((3\pi/2 - y)/\delta\big) & y > \pi\end{cases}\,,\qquad
# u_y = \epsilon\sin x\,,
# ```
# with ``\delta = \pi/15`` and ``\epsilon = 0.05``. The shear layers are unstable and roll
# up into vortices that wind up progressively thinner filaments of vorticity. We solve the
# **inviscid** problem, ``\nu = 0`` (the Euler equations). The scheme should then conserve
# both the kinetic energy and the enstrophy exactly, while the vorticity filaments
# eventually become too thin for any fixed mesh to resolve.

const δ_sl = π / 15
const ϵ_sl = 0.05
function shear_layer_velocity(x)
    y = x[:, 2]
    u = @. ifelse(y <= π, tanh((y - π / 2) / δ_sl), tanh((3π / 2 - y) / δ_sl))
    v = @. ϵ_sl * sin(x[:, 1])
    return [u, v]
end

# We use cubic splines (``p = 3``) on a ``30\times 30`` mesh and a time step
# ``\Delta t = 0.2``. Because the implicit midpoint rule conserves energy, it is
# unconditionally stable, so the time step is limited by accuracy only. With
# ``|\boldsymbol{u}| \lesssim 1`` and element size ``h = 2\pi/30``, the CFL number
# ``|\boldsymbol{u}|\Delta t/h`` is about ``1``, or about ``3`` when measured against the
# resolution ``h/p`` of the cubic splines. This mesh is deliberately coarse, so that the
# example runs quickly; it is enough to see the roll-up.
#
# Set `show_progress = true` to print, at every time step, the Newton residuals and the
# conservation diagnostics (see [The time loop](@ref NS2DTimeLoop)). To keep this page
# short, it is switched off here.

show_progress = false # set to true to print the diagnostics of every time step
show_progress = true # when running this file directly, always print them #src
N = 30  # there will be N² elements
p = 3   # polynomial degree of basis functions (with maximal regularity) 
Δt_sl = 0.2  # time step size
@printf("\n\n========================================================== 
= Shear layer roll up                                    =
==========================================================\n")

@printf(
    "\n==> Full simulation
      N  = %d x %d
      p  = %d 
      Δt = %.2f",
    N,
    N,
    p,
    Δt_sl
)

problem_sl = setup_navier_stokes((N, N), (p, p))
snapshot_steps = round.(Int, [0, 4, 6, 8] ./ Δt_sl) # t = 0, 4, 6, 8
u_sl, w_sl, P_sl, history_sl, snapshots_sl = run_navier_stokes(
    problem_sl,
    shear_layer_velocity,
    0.0,
    Δt_sl,
    snapshot_steps[end];
    snapshot_steps=snapshot_steps,
    verbose=show_progress,
    show_conservation=show_progress,
)

# The vorticity at ``t = 0, 4, 6, 8``:

fig_sl = Figure(; size=(900, 800))
for (i, step) in enumerate(snapshot_steps)
    snapshot = snapshots_sl[step]
    x, y, values = sample_vorticity(problem_sl, snapshot.w)
    ax = Axis(
        fig_sl[fld1(i, 2), mod1(i, 2)];
        title=@sprintf("t = %.1f", snapshot.t),
        aspect=DataAspect(),
        xlabel="x",
        ylabel="y",
    )
    heatmap!(ax, x, y, values; colormap=:balance, colorrange=(-5, 5))
end
Colorbar(fig_sl[:, 3]; colormap=:balance, limits=(-5, 5), label="vorticity")
display(GLMakie.Screen(), fig_sl) #src
DisplayAs.Text(DisplayAs.PNG(fig_sl)) #hide

# The shear layers roll up into two vortices connected by thin braids of vorticity. On
# this coarse mesh, the braids soon become thinner than the mesh can represent:
# grid-scale oscillations are already visible at ``t = 4``. By ``t = 6`` they fill the
# domain, and by ``t = 8`` the braids break up into small spurious vortices. On a finer
# mesh (for example ``60\times 60`` elements with ``p = 2``) the roll-up stays clean
# until about ``t = 6``. This is the expected behaviour of a scheme without
# numerical dissipation on an under-resolved flow, not an instability. The enstrophy that
# the roll-up transfers to ever smaller scales cannot leave the resolved scales, so it
# appears as small-scale noise. Energy and enstrophy remain bounded by construction.
# Refining the mesh delays this. Physical viscosity (``\nu > 0``), or a sub-grid model
# acting on the smallest scales such as the one proposed in [Zhang2022](@cite), removes
# it.
#
# Finally, the conservation properties. Kinetic energy and enstrophy are both conserved up
# to the Newton tolerance and round-off, and the divergence of the velocity is zero to
# machine precision. The size of any remaining drift is set by the Newton tolerance.

K₀, ℰ₀ = history_sl.kinetic_energy[1], history_sl.enstrophy[1]
fig_conservation = Figure(; size=(900, 350))
ax_K = Axis(
    fig_conservation[1, 1]; xlabel="t", ylabel="(K - K₀) / K₀", title="kinetic energy"
)
lines!(ax_K, history_sl.t, (history_sl.kinetic_energy .- K₀) ./ K₀)
ax_E = Axis(fig_conservation[1, 2]; xlabel="t", ylabel="(ℰ - ℰ₀) / ℰ₀", title="enstrophy")
lines!(ax_E, history_sl.t, (history_sl.enstrophy .- ℰ₀) ./ ℰ₀)
display(GLMakie.Screen(), fig_conservation) #src
DisplayAs.Text(DisplayAs.PNG(fig_conservation)) #hide

#
relative_change(values) = maximum(abs, (values .- values[1]) ./ values[1])
num_newton_iterations =
    sum(length, history_sl.newton_residuals) - length(history_sl.newton_residuals)
@printf(
    "max relative kinetic energy change = %.3e\n",
    relative_change(history_sl.kinetic_energy)
)
@printf(
    "max relative enstrophy change      = %.3e\n", relative_change(history_sl.enstrophy)
)
@printf(
    "‖du‖ at t = %.1f                    = %.3e\n",
    history_sl.t[end],
    divergence_norm(problem_sl, u_sl)
)
@printf(
    "Newton iterations per time step    = %.2f\n",
    num_newton_iterations / length(history_sl.newton_residuals)
)

# ## [The same scheme with the `TimeIntegrators` module](@id NS2DTimeIntegrators)
# So far we wrote the time loop by hand. `Mantis` also provides a general
# [TimeIntegrators](@ref) module (see the [time integration tutorial](GetStartTimeIntegration.md)).
# In this section we use it to advance the same discretization in time, and check that
# it gives the same result.
#
# `TimeIntegrators` separates two ingredients:
# - the **ODE**, i.e. the problem, described by problem-specific functions;
# - the **time integration scheme**, described by its Butcher tableau.
#
# The same description of the ODE can be combined with any scheme of the right class. We
# first explain the class of schemes we use, then what the ODE description must contain,
# and only then choose the scheme.
#
# ### [Diagonally implicit Runge–Kutta methods](@id NS2DTIDIRK)
# Consider a general ODE
# ```math
# \frac{\mathrm{d}\mathbf{y}}{\mathrm{d}t} = \mathbf{g}(t, \mathbf{y})\,, \qquad
# \mathbf{y}(0) = \mathbf{y}^0\,.
# ```
# An ``s``-stage Runge–Kutta method with Butcher tableau ``(a_{ij}, b_i, c_i)`` advances the
# solution from ``t^n`` to ``t^{n+1} = t^n + \Delta t`` by first computing ``s`` *stage
# values* ``\mathbf{Y}_1, \dots, \mathbf{Y}_s``, at the stage times
# ``t_i = t^n + c_i\,\Delta t``, and then the new solution:
# ```math
# \mathbf{Y}_i = \mathbf{y}^n + \Delta t\sum_{j=1}^{s} a_{ij}\,\mathbf{g}(t_j, \mathbf{Y}_j)\,,
# \quad i = 1, \dots, s\,, \qquad
# \mathbf{y}^{n+1} = \mathbf{y}^n + \Delta t\sum_{i=1}^{s} b_i\,\mathbf{g}(t_i, \mathbf{Y}_i)\,.
# ```
# Many texts write the method instead in terms of the *stage derivatives*
# ``\mathbf{k}_i = \mathbf{g}(t_i, \mathbf{Y}_i)``:
# ```math
# \mathbf{k}_i = \mathbf{g}\Big(t^n + c_i\,\Delta t,\ \mathbf{y}^n + \Delta t\sum_{j=1}^{s}
# a_{ij}\,\mathbf{k}_j\Big)\,, \qquad
# \mathbf{y}^{n+1} = \mathbf{y}^n + \Delta t\sum_{i=1}^{s} b_i\,\mathbf{k}_i\,.
# ```
# Both forms are equivalent: inserting ``\mathbf{k}_j = \mathbf{g}(t_j, \mathbf{Y}_j)`` into
# the first form gives ``\mathbf{Y}_i = \mathbf{y}^n + \Delta t\sum_j a_{ij}\,\mathbf{k}_j``,
# and applying ``\mathbf{g}(t_i, \cdot)`` to both sides gives the second. `TimeIntegrators`
# works with the stage values ``\mathbf{Y}_i``, so we use the first form.
#
# The method is *diagonally implicit* if ``a_{ij} = 0`` for ``j > i``. Then stage ``i``
# involves only ``\mathbf{Y}_i`` itself and the stages before it, and can be written as
# ```math
# \mathbf{Y}_i = \mathbf{y}^n + \Delta t\sum_{j=1}^{s} a_{ij}\,\mathbf{g}(t_j, \mathbf{Y}_j) =
# \mathbf{y}^n + \Delta t\sum_{j=1}^{i-1}a_{ij}\,\mathbf{g}(t_j, \mathbf{Y}_j) + \Delta t\, a_{ii}\,\mathbf{g}(t_i, \mathbf{Y}_i)\,,
# \quad i = 1, \dots, s\,,
# ```
# since the method is *diagonally implicit* (``a_{ij} = 0`` for ``j > i``). If we now introduce
# ```math
# \mathbf{x}_i = \mathbf{y}^n + \Delta t\sum_{j=1}^{i-1} a_{ij}\,\mathbf{g}(t_j,
# \mathbf{Y}_j)\,,
# ```
# we can rewrite the stages as
# ```math
# \mathbf{Y}_i = \mathbf{x}_i + \Delta t\, a_{ii}\,\mathbf{g}(t_i, \mathbf{Y}_i)\,.
# ```
# When stage ``i`` is computed, ``\mathbf{x}_i`` is known. So a step consists
# of solving ``s`` equations of the same form, one after the other. We call them *stage
# equations*.
#
# ### [Describing the ODE: `define_diagonally_implicit_ode`](@id NS2DTISolve)
# In the algorithm above, everything is generic except one operation: solving a stage
# equation ``\mathbf{Y}_{i} - \Delta t\, a_{ii}\,\mathbf{g}(t_{i}, \mathbf{Y}_{i}) = \mathbf{x}_{i}`` for given ``\mathbf{x}_{i}``,
# ``a_{ii}``, ``\Delta t``, and stage time ``t_{i}``. Note that, for each stage we must solve
# an equation for ``\mathbf{Y}`` of the form
# ```math
# \mathbf{Y} - h\,\mathbf{g}(t, \mathbf{Y}) = \mathbf{x}\,,
# ```
# given ``h``, ``t``, and ``\mathbf{x}``. The scheme provides ``h``, ``t``, and
# ``\mathbf{x}``; how to solve this equation depends on ``\mathbf{g}``, which only we know.
# So we must provide a function that, given ``h``, ``t``, and ``\mathbf{x}``, returns the
# solution ``\mathbf{Y}`` of the stage equation.
# `TimeIntegrators.define_diagonally_implicit_ode(implicit_solve!, implicit_evaluate!)`
# collects the problem-specific functions into an object describing the ODE. Its inputs are:
# - `implicit_solve!(Y, x, h, t)` (required): overwrites `Y` with the solution of the stage
#   equation ``\mathbf{Y} - h\,\mathbf{g}(t, \mathbf{Y}) = \mathbf{x}``, where ``t`` is the
#   stage time. ``\mathbf{g}`` is not an argument: ``\mathbf{g}`` is directly included
#   inside the function. For a nonlinear ``\mathbf{g}``, `implicit_solve!(Y, x, h, t)` implements, for example,
#   Newton's method applied to the residual
#   ```math
#   \mathbf{F}(\mathbf{Y}) = \mathbf{Y} - h\,\mathbf{g}(t, \mathbf{Y}) - \mathbf{x}\,, \qquad
#   \frac{\partial\mathbf{F}}{\partial\mathbf{Y}} = \mathsf{I} - h\,\frac{\partial\mathbf{g}}{\partial\mathbf{Y}}\,,
#   ```
#   starting from the initial guess ``\mathbf{Y} = \mathbf{x}``. In summary, given 
#   ``h``, ``t``, and ``\boldsymbol{x}``, `implicit_solve!(Y, x, h, t)` must return the solution
#   to the stage equation.
# - `implicit_evaluate!(G, y, t)` (optional): overwrites `G` with
#   ``\mathbf{g}(t, \mathbf{y})``. The algorithm needs the values
#   ``\mathbf{g}(t_i, \mathbf{Y}_i)`` twice: in ``\mathbf{x}_j`` of the later stages
#   ``j > i``, and in the update of ``\mathbf{y}^{n+1}``. If ``a_{ii} \neq 0``, these values
#   can be obtained without evaluating ``\mathbf{g}``. The solution ``\mathbf{Y}_i``
#   returned by `implicit_solve!` satisfies the stage equation
#   ``\mathbf{Y}_i = \mathbf{x}_i + h_i\,\mathbf{g}(t_i, \mathbf{Y}_i)``, with
#   ``h_i = a_{ii}\,\Delta t \neq 0``, so
#   ```math
#   \mathbf{g}(t_i, \mathbf{Y}_i) = \frac{\mathbf{Y}_i - \mathbf{x}_i}{h_i}\,.
#   ```
#   This costs one vector subtraction, while evaluating ``\mathbf{g}`` may be expensive.
#   If ``a_{ii} = 0``, the stage equation reduces to ``\mathbf{Y}_i = \mathbf{x}_i``: stage
#   ``i`` is explicit, its equation does not contain ``\mathbf{g}(t_i, \mathbf{Y}_i)``, and
#   the division above is not possible. Then ``\mathbf{g}(t_i, \mathbf{Y}_i)`` must be
#   evaluated directly, and the user must provide `implicit_evaluate!`. The same holds for
#   multi-step schemes, which evaluate ``\mathbf{g}`` at the initial condition during
#   their initialisation. For schemes with ``a_{ii} \neq 0`` in all stages,
#   `implicit_evaluate!` is never called.
#
# In the examples considered here, ``\mathbf{g}`` does not depend on ``t`` (the ODE is
# *autonomous*), so from here on we drop the ``t`` argument, and the time arguments of the
# two functions are not used.
#
# Note that the ODE object contains no information about the scheme: the Butcher tableau
# only determines *which* ``\mathbf{x}``, ``h``, and ``t`` are passed to `implicit_solve!`.
# This is a setup step that is common to all time integrators that require an implicit solve.
#
# ### [Choosing the scheme: the implicit midpoint rule](@id NS2DTIMidpoint)
# The scheme is chosen when the solution is initialised, with
# `TimeIntegrators.initialise_scheme(y⁰, scheme)`. This returns a solution object holding the
# initial condition ``\mathbf{y}^0`` and the scheme. Here we use the implicit midpoint rule,
# `TimeIntegrators.IMPLICIT_MIDPOINT`: the one-stage method with ``a_{11} = \tfrac{1}{2}``,
# ``b_1 = 1``, ``c_1 = \tfrac{1}{2}``. Any other diagonally implicit scheme of
# `TimeIntegrators`, such as `TimeIntegrators.DIRK2`, could be used instead with the same ODE
# object.
#
# For the implicit midpoint rule, the general algorithm reduces to:
# - one stage equation, with ``\mathbf{x} = \mathbf{y}^n`` and ``h = \tfrac{\Delta t}{2}``:
#   ``\mathbf{Y} - \tfrac{\Delta t}{2}\,\mathbf{g}(\mathbf{Y}) = \mathbf{y}^n``;
# - the update ``\mathbf{y}^{n+1} = \mathbf{y}^n + \Delta t\,\mathbf{g}(\mathbf{Y})``. With
#   ``\mathbf{g}(\mathbf{Y}) = (\mathbf{Y} - \mathbf{y}^n)/h``, this is
#   ``\mathbf{y}^{n+1} = 2\mathbf{Y} - \mathbf{y}^n``.
#
# The last relation shows that ``\mathbf{Y} = \tfrac{1}{2}(\mathbf{y}^n + \mathbf{y}^{n+1})``
# is the midpoint value. Inserted into the update, it gives the familiar form of the rule,
# ``\mathbf{y}^{n+1} = \mathbf{y}^n + \Delta t\,\mathbf{g}\big(\tfrac{1}{2}(\mathbf{y}^n +
# \mathbf{y}^{n+1})\big)``, i.e. exactly the scheme of the previous sections. Since
# ``a_{11} \neq 0``, `implicit_evaluate!` is never called.
#
# ### [Integrating in time](@id NS2DTIScheme)
# Each call `TimeIntegrators.time_integrate!(solution, ode, tⁿ, Δt)` advances the solution
# object by one time step: for every stage it forms ``\mathbf{x}_i`` and ``h_i``, calls
# `implicit_solve!`, and finally computes ``\mathbf{y}^{n+1}``. For the implicit midpoint rule
# this is one call `implicit_solve!(Y, yⁿ, Δt/2, tⁿ + Δt/2)` followed by
# ``\mathbf{y}^{n+1} = 2\mathbf{Y} - \mathbf{y}^n``. `TimeIntegrators.get_solution(solution)`
# returns the current solution.
#
# Put together, a schematic implementation for a general ``\mathbf{g}`` reads:
# ```julia
# # Describing the ODE: the right-hand side and Newton's method for Y - h g(Y) = x.
# function implicit_evaluate!(G, y, t)
#     G .= g(y)
# end
#
# function implicit_solve!(Y, x, h, t)
#     Y .= x # initial guess
#     for iteration in 1:max_iterations
#         F = Y - h * g(Y) - x
#         if norm(F) < tolerance
#             break
#         end
#         Y .-= (I - h * dg_dy(Y)) \ F
#     end
# end
#
# ode = TimeIntegrators.define_diagonally_implicit_ode(implicit_solve!, implicit_evaluate!)
#
# # Choosing the scheme and integrating in time.
# solution = TimeIntegrators.initialise_scheme(y⁰, TimeIntegrators.IMPLICIT_MIDPOINT)
# t = 0.0
# for step in 1:num_steps
#     TimeIntegrators.time_integrate!(solution, ode, t, Δt) # yⁿ → yⁿ⁺¹
#     t += Δt
# end
# y = TimeIntegrators.get_solution(solution)[:, 1] # the solution at the final time
# ```
# Here `g(y)` and its Jacobian `dg_dy(y)` stand for the problem at hand.
#
# We now carry out these steps for the Navier–Stokes equations.
#
# ### [Navier–Stokes: the right-hand side ``\mathbf{g}``](@id NS2DTIODE)
# Before discretizing in time, the semi-discrete Navier–Stokes equations read
# ```math
# \mathsf{M}^1\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} + \mathsf{R}(\mathbf{w})\,\mathbf{u}
# + \nu\,\mathsf{C}\,\mathbf{w} - \mathsf{D}^{\mathsf{T}}\mathbf{P} = 0\,, \qquad
# \mathsf{M}^0\mathbf{w} = \mathsf{C}^{\mathsf{T}}\mathbf{u}\,, \qquad
# \mathsf{D}\,\mathbf{u} = 0\,,
# ```
# where ``\mathsf{R}(\mathbf{w})`` denotes the matrix ``\mathsf{R}(w_h)`` of the vorticity
# field ``w_h`` with coefficients ``\mathbf{w}``. Solving the first equation for
# ``\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t}``, we identify the right-hand side
# ``\mathbf{g}``:
# ```math
# \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} = \mathbf{g}(\mathbf{u}, \mathbf{w}, \mathbf{P})\,,
# \qquad
# \mathbf{g}(\mathbf{u}, \mathbf{w}, \mathbf{P}) = (\mathsf{M}^1)^{-1}\big(
# -\mathsf{R}(\mathbf{w})\,\mathbf{u} - \nu\,\mathsf{C}\,\mathbf{w}
# + \mathsf{D}^{\mathsf{T}}\mathbf{P}\big)\,,
# ```
# subject to the two remaining equations,
# ```math
# \mathsf{M}^0\mathbf{w} = \mathsf{C}^{\mathsf{T}}\mathbf{u}\,, \qquad
# \mathsf{D}\,\mathbf{u} = 0\,.
# ```
# This is the ODE of the generic description with ``\mathbf{y} = \mathbf{u}``, with two
# differences:
# - ``\mathbf{g}`` also depends on the vorticity ``\mathbf{w}`` and the pressure
#   ``\mathbf{P}``;
# - two constraints accompany the ODE: the definition of the vorticity, and the
#   divergence-free constraint ``\mathsf{D}\,\mathbf{u} = 0``.
#
# Neither ``\mathbf{w}`` nor ``\mathbf{P}`` has an evolution equation of its own. The
# vorticity is determined by the velocity through its definition. The pressure is the
# Lagrange multiplier of the divergence-free constraint, and takes the value for which the
# constraint holds.
#
# ### [Navier–Stokes: solving the stage equation](@id NS2DTIStage)
# With this ``\mathbf{g}``, the stage equation of the generic description, together with
# the two constraints, becomes, for the stage value ``\mathbf{Y}``, the stage vorticity
# ``\mathbf{w}`` and the stage pressure ``\mathbf{P}``,
# ```math
# \mathbf{Y} - h\,\mathbf{g}(\mathbf{Y}, \mathbf{w}, \mathbf{P}) = \mathbf{x}\,, \qquad
# \mathsf{M}^0\mathbf{w} = \mathsf{C}^{\mathsf{T}}\mathbf{Y}\,, \qquad
# \mathsf{D}\,\mathbf{Y} = 0\,.
# ```
# Each constraint adds one equation, which determines one additional unknown: the
# vorticity definition determines ``\mathbf{w}``, and the divergence-free constraint
# determines ``\mathbf{P}``. For the implicit midpoint rule, ``\mathbf{x} = \mathbf{u}^n``,
# ``h = \tfrac{\Delta t}{2}``, and ``\mathbf{Y} = \mathbf{u}^{n+\frac{1}{2}}`` is the
# midpoint velocity.
#
# To avoid the inverse ``(\mathsf{M}^1)^{-1}`` in ``\mathbf{g}``, we multiply the first
# equation by ``\mathsf{M}^1/h``. This gives the nonlinear system for
# ``(\mathbf{Y}, \mathbf{w}, \mathbf{P})`` that we solve:
# ```math
# \begin{aligned}
# \mathsf{M}^1\frac{\mathbf{Y} - \mathbf{x}}{h} + \mathsf{R}(\mathbf{w})\,\mathbf{Y}
# + \nu\,\mathsf{C}\,\mathbf{w} - \mathsf{D}^{\mathsf{T}}\mathbf{P} &= 0\,,\\
# \mathsf{M}^0\mathbf{w} - \mathsf{C}^{\mathsf{T}}\mathbf{Y} &= 0\,,\\
# \mathsf{D}\,\mathbf{Y} &= 0\,.
# \end{aligned}
# ```
# For the implicit midpoint rule, this is the system of the
# [Newton–Raphson section](@ref NS2DNewton), written for the midpoint values instead of the
# values at ``t^{n+1}``. We solve it with the same Newton method. As
# there, the derivative of ``\mathsf{R}(\mathbf{w})\,\mathbf{Y}`` with respect to
# ``\mathbf{Y}`` is ``\mathsf{R}(\mathbf{w})``, and with respect to ``\mathbf{w}`` it is
# ``\mathsf{Q}(\mathbf{Y})``, the matrix ``\mathsf{Q}`` of
# [Assembling operators](@ref NS2DOperators) evaluated at the velocity field with
# coefficients ``\mathbf{Y}``. The Jacobian is the same as before, without the factors
# ``\tfrac{1}{2}``:
# ```math
# \mathsf{J} =
# \begin{bmatrix}
# \dfrac{1}{h}\mathsf{M}^1 + \mathsf{R}(\mathbf{w}) & \mathsf{Q}(\mathbf{Y}) + \nu\,\mathsf{C}
# & -\mathsf{D}^{\mathsf{T}}\\[2mm]
# -\mathsf{C}^{\mathsf{T}} & \mathsf{M}^0 & 0\\[1mm]
# \mathsf{D} & 0 & 0
# \end{bmatrix}.
# ```
# The function below performs this Newton solve, starting from ``\mathbf{Y} = \mathbf{x}``.
# It returns the stage value ``\mathbf{Y}``, together with the pressure and the Newton
# residuals for later inspection. It assumes ``h > 0``, which holds for every scheme whose
# stages all have ``a_{ii} \neq 0``, such as the implicit midpoint rule.

function solve_stage_equation(problem, ν, x, h, P_guess; tolerance=1e-11, max_iterations=15)
    (; Λ⁰, Λ¹, M⁰, M¹, C, D, D_gauged, gauge, num_u, num_w, num_P) = problem
    iY, iw, iP = 1:num_u, num_u .+ (1:num_w), (num_u + num_w) .+ (1:num_P)

    z = [x; compute_vorticity(problem, x); P_guess] # initial guess: Y = x
    residual_norms = Float64[]
    for iteration in 1:max_iterations
        Y, w, P = z[iY], z[iw], z[iP]
        R, Q = convection_operators(
            problem, Forms.FormField(Λ¹, Y, "u"), Forms.FormField(Λ⁰, w, "w")
        )

        ## Residual of the stage equation, the vorticity and the (gauged) continuity.
        F_Y = M¹ * (Y .- x) ./ h .+ R * Y .+ ν .* (C * w) .- D' * P
        F_w = M⁰ * w .- C' * Y
        F_P = D_gauged * Y .+ gauge * P
        F = [F_Y; F_w; F_P]
        push!(residual_norms, norm(F))
        if residual_norms[end] < tolerance
            break
        end

        ## Jacobian, with block columns ordered as the unknowns (Y, w, P).
        J_YY = M¹ ./ h .+ R
        J_Yw = Q .+ ν .* C
        J = [
            J_YY J_Yw -D'
            -C' M⁰ spzeros(num_w, num_P)
            D_gauged spzeros(num_P, num_w) gauge
        ]
        z .-= J \ F
    end
    if residual_norms[end] ≥ tolerance
        @warn "Newton did not converge" residual_norms
    end

    return z[iY], z[iP], residual_norms
end

# ### [Navier–Stokes: describing the ODE](@id NS2DTIDefine)
# `implicit_evaluate!(G, y, t)` receives only the velocity ``\mathbf{y} = \mathbf{u}``, and
# must return ``\mathbf{G} = \mathbf{g}(\mathbf{u}, \mathbf{w}, \mathbf{P})``. As in the
# stage equation, the vorticity ``\mathbf{w}`` and the pressure ``\mathbf{P}`` follow from
# the two constraints:
# - the vorticity from its definition, ``\mathsf{M}^0\mathbf{w} =
#   \mathsf{C}^{\mathsf{T}}\mathbf{u}``;
# - the pressure from the divergence-free constraint. Since ``\mathsf{D}\,\mathbf{u} = 0``
#   at all times, also ``\mathsf{D}\,\frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} =
#   \mathsf{D}\,\mathbf{G} = 0``.
#
# Multiplying ``\mathbf{G} = \mathbf{g}(\mathbf{u}, \mathbf{w}, \mathbf{P})`` by
# ``\mathsf{M}^1``, as for the stage equation, gives the system for
# ``(\mathbf{G}, \mathbf{w}, \mathbf{P})``:
# ```math
# \begin{aligned}
# \mathsf{M}^1\mathbf{G} + \mathsf{R}(\mathbf{w})\,\mathbf{u}
# + \nu\,\mathsf{C}\,\mathbf{w} - \mathsf{D}^{\mathsf{T}}\mathbf{P} &= 0\,,\\
# \mathsf{M}^0\mathbf{w} - \mathsf{C}^{\mathsf{T}}\mathbf{u} &= 0\,,\\
# \mathsf{D}\,\mathbf{G} &= 0\,.
# \end{aligned}
# ```
# Since ``\mathbf{u}`` is given, this system is linear. The second equation gives
# ``\mathbf{w}`` directly; the first and third then form a linear saddle-point system for
# ``(\mathbf{G}, \mathbf{P})``, in which we fix the pressure with the
# [pressure gauge](@ref NS2DGauge). The function `evaluate_g` solves it. The implicit
# midpoint rule never calls `implicit_evaluate!`, but schemes with a stage that has
# ``a_{ii} = 0`` do.

function evaluate_g(problem, ν, u)
    (; Λ⁰, Λ¹, M¹, C, D, D_gauged, gauge, num_u, num_P) = problem
    w = compute_vorticity(problem, u)
    R, _ = convection_operators(
        problem, Forms.FormField(Λ¹, u, "u"), Forms.FormField(Λ⁰, w, "w")
    )

    ## Solve M¹ G - Dᵀ P = -R(w) u - ν C w together with the (gauged) D G = 0.
    A = [M¹ -D'; D_gauged gauge]
    b = [-(R * u) .- ν .* (C * w); zeros(num_P)]
    G_and_P = A \ b
    G = G_and_P[1:num_u]

    return G
end

# We now wrap `evaluate_g` and `solve_stage_equation` in the two functions expected by
# `define_diagonally_implicit_ode`, `implicit_solve!(Y, x, h, t)` and
# `implicit_evaluate!(G, y, t)`, and pass them to it. The two functions receive only these
# arguments; everything else they need (the operators, the viscosity, a pressure to start
# Newton from) is captured from the enclosing function. We also keep the latest pressure
# and the Newton residuals in a small `stage_info` container, so that they can be
# inspected after each step.

function define_navier_stokes_ode(problem, ν)
    stage_info = (P=zeros(problem.num_P), newton_residuals=Vector{Float64}[])

    ## Solve the stage equation Y - h g(Y, w, P) = x with M⁰ w = Cᵀ Y and D Y = 0.
    function implicit_solve!(Y, x, h, t)
        Y_new, P, residuals = solve_stage_equation(problem, ν, x, h, stage_info.P)
        Y .= Y_new # the first argument must be overwritten
        stage_info.P .= P
        push!(stage_info.newton_residuals, residuals)
        return nothing
    end

    ## Evaluate the right-hand side G = g(u, w, P), with w and P from the constraints.
    function implicit_evaluate!(G, y, t)
        G .= evaluate_g(problem, ν, collect(vec(y))) # some schemes pass y as a one-column matrix
        return nothing
    end

    ode = TimeIntegrators.define_diagonally_implicit_ode(
        implicit_solve!, implicit_evaluate!
    )

    return ode, stage_info
end

# ### [Navier–Stokes: choosing the scheme and integrating in time](@id NS2DTILoop)
# With the ODE described, we choose the scheme with `initialise_scheme`, passing the initial
# velocity and `IMPLICIT_MIDPOINT`, and call `time_integrate!` once per time step, as in the
# schematic implementation. The vorticity at the end follows from the velocity.

function run_navier_stokes_time_integrators(problem, velocity::Function, ν, Δt, num_steps)
    u₀ = project_velocity(problem, velocity)
    ode, stage_info = define_navier_stokes_ode(problem, ν)
    solution = TimeIntegrators.initialise_scheme(u₀, TimeIntegrators.IMPLICIT_MIDPOINT)

    t = 0.0
    for step in 1:num_steps
        TimeIntegrators.time_integrate!(solution, ode, t, Δt)
        t += Δt
    end

    u = TimeIntegrators.get_solution(solution)[:, 1]
    w = compute_vorticity(problem, u)

    return u, w, stage_info.P, stage_info
end

# ### [Comparison with the hand-written time loop](@id NS2DTICompare)
# We rerun the Taylor–Green vortex up to ``t = 1`` with both implementations and compare the
# results. Both solve the same equations with the same Newton method from the same initial
# guess, so they should agree up to round-off.

problem_ti = setup_navier_stokes((16, 16), (3, 3))
Δt_ti, num_steps_ti = 0.05, 20
u_hand, w_hand, P_hand, history_hand, _ = run_navier_stokes(
    problem_ti, taylor_green_velocity, ν_tg, Δt_ti, num_steps_ti
)
u_ti, w_ti, P_ti, stage_info_ti = run_navier_stokes_time_integrators(
    problem_ti, taylor_green_velocity, ν_tg, Δt_ti, num_steps_ti
)

relative_difference(a, b) = norm(a - b) / norm(b)
@printf("relative difference in u = %.3e\n", relative_difference(u_ti, u_hand))
@printf("relative difference in w = %.3e\n", relative_difference(w_ti, w_hand))
@printf("relative difference in P = %.3e\n", relative_difference(P_ti, P_hand))
println("Newton residuals, first time step (TimeIntegrators):")
foreach(r -> @printf("    %.3e\n", r), stage_info_ti.newton_residuals[1])

# The two implementations agree to round-off, and Newton again converges quadratically.
#
# Finally, we check that `evaluate_g` and `solve_stage_equation` describe the same ODE. For the
# stage value ``\mathbf{Y}`` of the first time step of the implicit midpoint rule
# (``\mathbf{x} = \mathbf{u}^0``, ``h = \tfrac{\Delta t}{2}``), with stage vorticity
# ``\mathbf{w}`` and stage pressure ``\mathbf{P}``, the stage equation gives
# ``\mathbf{g}(\mathbf{Y}, \mathbf{w}, \mathbf{P}) = (\mathbf{Y} - \mathbf{u}^0)/h``. We
# compare this with `evaluate_g(problem, ν, Y)`, which computes ``\mathbf{w}`` and
# ``\mathbf{P}`` from ``\mathbf{Y}`` through the constraints.

u⁰_ti = project_velocity(problem_ti, taylor_green_velocity)
h_ti = Δt_ti / 2
Y_ti, _, _ = solve_stage_equation(problem_ti, ν_tg, u⁰_ti, h_ti, zeros(problem_ti.num_P))
@printf(
    "relative difference between g(Y, w, P) and (Y - u⁰)/h = %.3e\n",
    relative_difference(evaluate_g(problem_ti, ν_tg, Y_ti), (Y_ti .- u⁰_ti) ./ h_ti)
)

# Both agree, so `implicit_evaluate!` and `implicit_solve!` describe the same ODE. Since this
# description does not depend on the scheme, using another diagonally implicit scheme whose
# stages all have ``a_{ii} \neq 0`` (for example `TimeIntegrators.DIRK2`) only requires
# replacing `IMPLICIT_MIDPOINT` in `initialise_scheme`. Note, however, that the
# conservation proofs of this example rely on the implicit midpoint rule: other schemes
# generally do not conserve energy and enstrophy exactly.

# ## [Summary and outlook](@id NS2DSummary)
# We implemented the ``H(\mathrm{div})`` part of the formulation of [Zhang2022](@cite) in
# 2D with `Mantis`:
# - velocity, vorticity and total pressure are the differential forms ``u^1, w^0, P^2`` of
#   a periodic spline de Rham complex;
# - the Lamb vector ``\boldsymbol{\omega}\times\boldsymbol{u}`` is the interior product
#   ``\iota_{\boldsymbol{u}}\mathrm{d}u^\flat``, written with Hodge stars and wedge products
#   as ``w^0\wedge\star u^1``;
# - the implicit midpoint rule with Newton–Raphson gives a scheme that is pointwise
#   divergence free. In the inviscid case it conserves kinetic energy and enstrophy (the
#   2D counterpart of helicity) exactly. In the viscous case it dissipates both at exactly
#   the rates dictated by the viscous terms.
#
# Natural extensions are:
# - the full **dual-field** scheme, which adds the circulation subsystem
#   ``(u^\flat, \star w^0, \star P^2)``, staggers the two subsystems in time so that each
#   one is linear, and conserves helicity in 3D;
# - **3D**, where the same steps apply: the vorticity becomes a 1-form, and the Lamb vector
#   is again an interior product that Hirani's identity writes with ``\star`` and
#   ``\wedge``, which `Mantis` provides in 3D as well;
# - **curvilinear geometries**, by building the complex on a mapped geometry. Mass and
#   energy conservation carry over unchanged. Enstrophy conservation then holds up to the
#   quadrature error, because the metric terms make the integrands non-polynomial.
