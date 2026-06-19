```@meta
CurrentModule = Mantis.Assemblers
```
# [Assemblers](@id DocAssemblyModule)

The `Assemblers` module turns a *weak formulation*, written in the language of [Forms](@ref),
into the matrices and vectors of a discrete linear (or generalised eigenvalue) problem. It
connects the symbolic form expressions you write to the sparse arrays you solve.

## The assembly workflow

A typical solve in `Mantis` follows the same four steps regardless of the PDE:

1. **Collect the inputs.** A [`WeakFormInputs`](@ref) bundles the trial space, the test space,
   and (optionally) a forcing term. When trial and test spaces coincide (the usual
   Bubnov-Galerkin case) you only pass it once.
2. **Describe the weak form.** Write a function that reads the trial/test/forcing objects back
   out of the inputs (with `get_trial_form`, `get_test_form` and `get_forcing`) and returns
   the left- and right-hand side as nested tuples of integral expressions, one entry per block
   of the (possibly mixed) system.
3. **Build a [`WeakForm`](@ref).** This pairs the left/right-hand side expressions with the
   inputs.
4. **Assemble.** [`assemble`](@ref) evaluates every block over the mesh and returns the global
   system, optionally applying boundary conditions.

```julia
using Mantis

# ... geometry, B-spline space B, forcing f⁰, quadrature dΩ defined as usual ...
Λ⁰  = Forms.FormSpace(0, B, "ϕ")
wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

function poisson(inputs::Assemblers.AbstractInputs, dΩ)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)

    A = ∫(d(v⁰) ∧ ★(d(u⁰)), dΩ)   # stiffness block
    b = ∫(v⁰ ∧ ★(f⁰), dΩ)         # load block
    return ((A,),), ((b,),)        # one-by-one block structure
end

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

lhs, rhs  = poisson(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs, rhs, wfi)
A, b      = Assemblers.assemble(weak_form, bc)
ϕ⁰        = Forms.build_form_field(Λ⁰, vec(A \ b))
```

### Block structure

The left- and right-hand sides are returned as *tuples of tuples* of real-valued operators
([`AbstractRealValuedOperator`](@ref Mantis.Forms.AbstractRealValuedOperator)s, typically
[`Integral`](@ref Mantis.Forms.Integral)s). The outer tuple indexes block-rows and the inner
tuple indexes block-columns, so a scalar problem is `((A,),)` while a two-field mixed problem
(e.g. the mixed Hodge-Laplacian, or the Maxwell saddle-point system) is a ``2 \times 2``
arrangement such as `((A11, A12), (A21, A22))`. This is how `Mantis` represents the compatible
mixed problems of Finite Element Exterior Calculus. [`assemble`](@ref) stitches the blocks
into a single global matrix.

### Boundary conditions

Essential (Dirichlet) boundary conditions are applied *at assembly time*. You build a boundary
condition object with the [Forms](@ref) helper
[`set_dirichlet_boundary_conditions`](@ref Mantis.Forms.set_dirichlet_boundary_conditions) and
pass it to [`assemble`](@ref); natural
conditions need no special treatment because they are already contained in the weak form. The
keyword arguments `lhs_type` / `rhs_type` let you request dense outputs (e.g.
`Matrix{Float64}`), which is convenient for the generalised eigenvalue problems below.

## Standard weak formulations

`Mantis` ships ready-made weak forms for several model problems, so you do not have to rewrite
them each time. Each comes both as a *weak-form function* (returning the block expressions) and
as a *solver* convenience function.

The L² projection is [`L2_projection`](@ref), with solver [`solve_L2_projection`](@ref); see
the [L2 projection](@ref) example. The Hodge-Laplacian is the `hodge_laplace` family for
``0``-, ``1``- and ``n``-forms. The Maxwell eigenvalue problem is [`maxwell_eigenvalue`](@ref),
solved by [`solve_maxwell_eig`](@ref) (single-mesh and adaptive variants) against the
analytical reference [`get_analytical_maxwell_eig`](@ref); see the
[Maxwell eigenvalue problem](@ref) and [Adaptive refinement](@ref) examples.

These are good starting points when writing the weak form for a new problem.

## All docstrings from Mantis.Assemblers
```@autodocs
Modules = [Main.Mantis.Assemblers]
```
