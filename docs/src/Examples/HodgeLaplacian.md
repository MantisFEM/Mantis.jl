```@meta
EditURL = "../../../examples/src/HodgeLaplacian.jl"
```

# Hodge Laplacian

## Introduction

In this example, ww solve the Hodge-Laplacian for $0$-forms with Dirichlet boundary
conditions. The
``0``-form Hodge-Laplacian on a domain ``\Omega`` with boundary ``\partial \Omega`` is
```math
\begin{alignat*}{2}
    &\mathrm{d}^* \mathrm{d} \phi^0 = - f^0  \quad &&\text{on}\ \Omega \;, \\
    &tr(\phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;.
\end{alignat*}
```
The weak formulation is then as follows.
```math
\begin{gather*}
\text{Given}\ f^0 \in L^2 \Lambda^0 (\Omega),\ \text{find}\ \phi^0 \in \Lambda^0_h (\Omega)
\ \text{such that} \\
\int_{\Omega} \mathrm{d} \phi^0 \wedge \star \mathrm{d} \varphi^0 = \int_{\Omega} f^0
\wedge \star \varphi^0 \quad  \forall \ \varphi^0 \in \Lambda^0_{h,0} (\Omega) \;.
\end{gather*}
```

## Implementation

We start by defining the geometry and the function space. The geometry is a simple unit
square with ``4 \times 4`` elements. The function space is a B-spline space with degree
``p = (3, 3)`` and knot vector ``k = (2, 2)``.

````@example HodgeLaplacian
using Mantis

starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (4, 4)
geometry = Geometry.create_cartesian_box(
    starting_point, box_size, num_elements
)

p = (3, 3)
k = (2, 2)
B = FunctionSpaces.create_bspline_space(
    starting_point,
    box_size,
    num_elements,
    p,
    k,
)
````

Mantis works with forms, so we need to define the form space. In this case, we are
working with $0$-forms, so we define the form space as follows.

````@example HodgeLaplacian
Λ⁰ = Forms.FormSpace(0, geometry, B, "label")
````

We define the weak form inputs. The weak form inputs contain the trial and test spaces,
the forcing function, and the quadrature rule. We define the forcing function as a
function of the coordinates. In this case, we define the forcing function as
``f^0 = 8 \pi^2 \sin(2 \pi x) \sin(2 \pi y)``.

````@example HodgeLaplacian
function forcing_function(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sin(2.0 * pi * x[:,1]) * sin(2.0 * pi * x[:,2])]
end
f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, "f⁰")
````

The quadrature rule is defined as a tensor product rule of the degree of the B-spline
space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.

````@example HodgeLaplacian
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))
````

We define the weak form inputs as a `WeakFormInputs` object. The weak form inputs contain
the trial and test spaces, and the forcing function. The trial and test spaces are the
same in this case, which is the default.

````@example HodgeLaplacian
wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)
````

We define the weak form for the Hodge-Laplacian. The weak form is defined as a function
that takes the weak form inputs and the quadrature rule as arguments.

````@example HodgeLaplacian
function zero_form_hodge_laplacian(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)

    A = ∫(d(v⁰) ∧ ★(d(u⁰)), dΩ)
    lhs_expression = ((A,),)

    b = ∫(v⁰ ∧ ★(f⁰), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end
````

We can now assemble the linear system and solve it to obtain the solution. We define the
Dirichlet boundary conditions using the appropriate helper function.

````@example HodgeLaplacian
bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
````

We assemble the linear system and solve it to obtain the solution.

````@example HodgeLaplacian
lhs_expressions, rhs_expressions = zero_form_hodge_laplacian(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A, b = Assemblers.assemble(weak_form, bc)
sol = vec(A \ b)
ϕ⁰ = Forms.build_form_field(Λ⁰, sol)
````

We can now plot the solution using the `plot` function. This will write the output to a
VTK file that can be visualized using a VTK viewer, such as Paraview.

````@example HodgeLaplacian
data_folder = joinpath(dirname(dirname(pathof(Mantis))), "examples", "data")
output_data_folder = joinpath(data_folder, "output", "HodgeLaplacian")
output_filename = "HodgeLaplacian-0form-Dirichlet-$(length(p))D.vtu"
output_file = joinpath(output_data_folder, output_filename)
Plot.plot(
    ϕ⁰;
    vtk_filename = output_file,
    n_subcells = 1,
    degree = maximum(p),
    ascii = false,
    compress = false
)
````

## Extensions

### 3D
We can extend this example to 3D by defining the geometry as a unit cube with ``4 \times 4
\times 4`` elements. We define the function space as a B-spline space with degree ``p = (3
, 3, 3)`` and knot vector ``k = (2, 2, 2)``. We define the forcing function as ``f^0 = 12
\pi^2 \sin(2 \pi x) \sin(2 \pi y) \sin(2 \pi z)``.
Note that the new variables are renamed to avoid overwriting the previous variables and to
avoid type instabilities.

````@example HodgeLaplacian
starting_point_3D = (0.0, 0.0, 0.0)
box_size_3D = (1.0, 1.0, 1.0)
num_elements_3D = (4, 4, 4)
geometry_3D = Geometry.create_cartesian_box(
    starting_point_3D, box_size_3D, num_elements_3D
)

p_3D = (3, 3, 3)
k_3D = (2, 2, 2)
B_3D = FunctionSpaces.create_bspline_space(
    starting_point_3D,
    box_size_3D,
    num_elements_3D,
    p_3D,
    k_3D,
)

function forcing_function_3D(x::Matrix{Float64})
    return [
        @. 12.0 * pi^2 * sin(2.0 * pi * x[:,1]) * sin(2.0 * pi * x[:,2]) *
        sin(2.0 * pi * x[:,3])
    ]
end
````

We don't need to change any of the previous code, so we can reuse it directly:

````@example HodgeLaplacian
Λ⁰_3D = Forms.FormSpace(0, geometry_3D, FunctionSpaces.DirectSumSpace((B_3D,)), "label")

f⁰_3D = Forms.AnalyticalFormField(0, forcing_function_3D, geometry_3D, "f⁰")

canonical_qrule_3D = Quadrature.tensor_product_rule(p_3D .+ 1, Quadrature.gauss_legendre)
dΩ_3D = Quadrature.StandardQuadrature(canonical_qrule_3D, Geometry.get_num_elements(geometry_3D))

wfi_3D = Assemblers.WeakFormInputs(Λ⁰_3D, f⁰_3D)

bc_3D = Forms.set_dirichlet_boundary_conditions(Λ⁰_3D, 0.0)

lhs_expressions_3D, rhs_expressions_3D = zero_form_hodge_laplacian(wfi_3D, dΩ_3D)
weak_form_3D = Assemblers.WeakForm(lhs_expressions_3D, rhs_expressions_3D, wfi_3D)
A_3D, b_3D = Assemblers.assemble(weak_form_3D, bc_3D)
sol_3D = vec(A_3D \ b_3D)
ϕ⁰_3D = Forms.build_form_field(Λ⁰_3D, sol_3D)

output_filename_3D = "HodgeLaplacian-0form-Dirichlet-$(length(p_3D))D.vtu"
output_file_3D = joinpath(output_data_folder, output_filename_3D)
Plot.plot(
    ϕ⁰_3D;
    vtk_filename = output_file_3D,
    n_subcells = 1,
    degree = maximum(p_3D),
    ascii = false,
    compress = false
)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

