```@meta
CurrentModule = Mantis.Assemblers
```
# [Assemblers](@id DocAssemblyModule)

The assembly module contains two important pieces for assembly: the weak formulation and the assembly routine itself.
The weak formulation contains all the terms that need to be assembled, the assembler creates the final system matrix.

## [Weak Formulations](@id AssemblersWeakForms)
A weak formulation describes the problem in detail, see [The Finite Element Method (FEM)](@ref) for more information.
It consists of a few key ingredients: the given data (forcings, boundary conditions, etc.), the trial spaces, the test spaces, and the equations.
In `Mantis`, there are two structures that are used to represent all this data: `WeakFormInputs` and `WeakForm`.

### Representing problem data using `WeakFormInputs`
The data, trial spaces, and test spaces are represented by the following object.
```@docs
WeakFormInputs
```

The following getters are defined to obtain the data from a `WeakFormInputs` object.
```@docs
get_trial_forms(wfi::WeakFormInputs)
get_test_forms(wfi::WeakFormInputs)
get_forcings(wfi::WeakFormInputs)
get_trial_form(wfi::WeakFormInputs, i::Int=1)
get_test_form(wfi::WeakFormInputs, i::Int=1)
get_forcing(wfi::WeakFormInputs, i::Int=1)
get_num_trial(wfi::WeakFormInputs)
get_num_test(wfi::WeakFormInputs)
get_num_forcings(wfi::WeakFormInputs)
```

### Representing weak formulations using `WeakForm`
The complete weak formulation is represented by the following object, that takes a `WeakFormInputs` and the left-hand and right-hand sides of the equations as inputs.
```@docs
WeakForm
```

For this object, the following getters are defined.
```@docs
get_lhs_expressions
get_rhs_expressions
get_inputs
get_test_forms(wf::WeakForm)
get_trial_forms(wf::WeakForm)
get_forcings(wf::WeakForm)
get_forcing(wf::WeakForm, id::Int=1)
get_test_sizes
get_trial_sizes
get_test_size
get_trial_size
get_lhs_size
get_rhs_size
get_test_offsets
get_trial_offsets
get_estimated_nnz_per_elem
get_num_elements
get_num_evaluation_elements
```

### Example: setting up a weak form.
As an example, consider the mixed ``n``-form Hodge-Laplacian in 3D also treated in [Finite Element Exterior Calculus (FEEC)](@ref).
The weak formulation is written as:
Given ``f \in L^2(\Omega)``, find ``(\sigma_h, u_h) \in V_h^{2} \times V_h^3`` such that
```math
    \begin{equation}
        \begin{split}
            (\sigma_h, \tau_h) - (u_h, d \tau_h) &= 0, \quad \forall \tau_h \in V_h^{2},\\
            (d \sigma_h, v_h) &= (f, v_h), \quad \forall v_h \in V_h^3.
        \end{split}
    \end{equation}
```
We first create a `WeakFormInputs` object containing the test and trial spaces (which are the same in this case, so we only have to include them once), and the forcing (which we set to return one here for simplicity).
```@repl nFormHodgeLaplacianWeakForm
using Mantis
V0, V1, V2, V3 = Forms.create_tensor_product_bspline_de_rham_complex(
    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 4), (3, 3, 3), (2, 2, 2)
);

geometry = Forms.get_geometry(V2);

forcing_function(x) = [ones(size(x, 1))]
f = Forms.AnalyticalFormField(3, forcing_function, geometry, "f");

weak_form_inputs = Assemblers.WeakFormInputs((V2, V3), (f,));
```
With the `WeakFormInputs` object ready, we can create the complete `WeakForm` object.
Note that the `lhs_expressions` and `rhs_expressions` contain the integer `0` where we don't need a term.
```@repl nFormHodgeLaplacianWeakForm
canonical_qrule = Quadrature.tensor_product_rule((4, 4, 4), Quadrature.gauss_legendre);
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry));

τ_h, v_h = Assemblers.get_test_forms(weak_form_inputs);
σ_h, u_h = Assemblers.get_trial_forms(weak_form_inputs);
f = Assemblers.get_forcing(weak_form_inputs);

A_11 = ∫(τ_h ∧ ★(σ_h), dΩ);
A_12 = -∫(d(τ_h) ∧ ★(u_h), dΩ);
A_21 = ∫(v_h ∧ ★(d(σ_h)), dΩ);
lhs_expressions = ((A_11, A_12), (A_21, 0));

b_21 = ∫(v_h ∧ ★(f), dΩ);
rhs_expressions = ((0,), (b_21,));

weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs);
```

The weak form object can now be assembled using the [`assemble`](@ref) function.

### [Pre-implemented weak formulations](@id AssemblersWeakForms)
While you can always write your own weak formulation, `Mantis` does provide a few common pre-made weak formulations. 
The pre-implemented weak formulations require input spaces, forcings, and quadrature, but simplify the remaining steps.
Most pre-implemented weak formulations consist of two parts: one that encodes the equations, another that also performs the remaining setup and solve steps.

::: details ``L^2``-projection
```@docs
L2_projection
solve_L2_projection
```
:::

::: details Hodge-Laplacians
```@docs
zero_form_hodge_laplacian
solve_zero_form_hodge_laplacian
one_form_hodge_laplacian
solve_one_form_hodge_laplacian
n_form_hodge_laplacian
solve_volume_form_hodge_laplacian
```
:::

::: details Maxwell Eigenvalue Problems

In this case, next to the functions `maxwell_eigenvalue` and `solve_maxwell_eig`, `Mantis` also provides two functions to obtain the analytical eigenvalues.
```@docs
maxwell_eigenvalue
analytical_maxwell_eigenfunction
get_analytical_maxwell_eig
solve_maxwell_eig
```
:::

## Assembling a weak formulation
In `Mantis`, the actual assembly routine can be used by simply calling the following function.
```@docs
assemble
```
So, continuing the previous example, we can assemble our problem by simply calling
```@repl nFormHodgeLaplacianWeakForm
A, b = Assemblers.assemble(weak_form);
```
This gives us the left-hand side matrix `A` and right-hand side vector `b`. 
You can solve this with a solver of your choosing.
Here we pick Julia's backslash operator.
We can obtain the final solution as a `FormField` using the `build_form_fields` helper.
This helper takes the trial spaces and full solution vector as inputs, and combines the spaces and coefficients into the final solution objects.
```@repl nFormHodgeLaplacianWeakForm
sol = A \ b;
sigma_h, u_h = Forms.build_form_fields((V2, V3), sol);
```

::: details Helpers used within `assemble`

The assembly module also provides the following helper functions that are used within the assembly routine.
```@docs
get_pre_allocation
add_expression_contributions!
zero_rows!
add_bc!
set_diagonal!
build_array
```

:::
