```@meta
EditURL = "../../../examples/src/Biharmonic.jl"
```

# Biharmonic

The biharmonic problem is a typical example of a higher-order PDE problem. It can appear
in, for example, elasticity and fluid flow problems. In this example, we will briefly
review what the biharmonic problem looks like, and we will implement it using Mantis.jl.

## Formulation

### The 1D case.

In 1D, ignoring form notation, the biharmonic problem is defined as:
```math
\begin{alignat*}{2}
    &\frac{\partial^4 \phi(x)}{\partial x^4} = - f(x)  \quad &&\text{for}\ x \in [0, L] \;, \\
    &\phi(0) = \phi(L) = 0 \;, \\
    &-\frac{\partial \phi}{\partial x}(0) = \frac{\partial \phi}{\partial x}(L) = 0 \;,
\end{alignat*}
```
where we have chosen the domain to be ``[0, L]`` with ``L`` some length.
The weak formulation is then as follows.
```math
\begin{gather*}
\text{Given}\ f \in L^2 ([0, L]),\ \text{find}\ \phi \in H^2_{h,0}([0, L])\
\text{such that} \\
\int_0^L \frac{\partial^2 \phi}{\partial x^2} \frac{\partial^2 \phi}{\partial x^2} dx =
\int_{\Omega} \psi f dx \quad  \forall \ \psi \in H^2_{h,0}([0, L]) \;.
\end{gather*}
```

### The differential form case in nD.

Since Mantis.jl is designed to deal with differential form, we prefer to work with the
differential form formulation of the biharmonic problem. The 1D example above is the 1D
version of the ``0``-form biharmonic problem with homogeneous Dirichlet and Neumann
boundary conditions. The ``0``-form biharmonic problem in ``n``-dimensions on domain
``\Omega \subset \mathbb{R}^n`` with boundary ``\partial \Omega`` is
```math
\begin{alignat*}{2}
    &\Delta^2 \phi^0 = - f^0  \quad &&\text{on}\ \Omega \;, \\
    &tr(\phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;, \\
    &tr(\star \mathrm{d} \phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;.
\end{alignat*}
```
The weak formulation is then as follows.
```math
\begin{gather*}
\text{Given}\ f^0 \in L^2 \Lambda^0 (\Omega),\ \text{find}\ \phi^0 \in H^2
\Lambda^0_h (\Omega)\ \text{such that} \\
\int_{\Omega} \Delta \psi^0 \wedge \star \Delta \phi^0 = \int_{\Omega} \psi^0
\wedge \star f^0 \quad  \forall \ \psi^0 \in H^2\Lambda^0_{h,0} (\Omega) \;.
\end{gather*}
```

### What is actually computed?
In most finite element codes, and Mantis.jl is no exception, the integrals in the above
weak formulations are not directly computed on the given domain. Instead, they are
pulled-back (mapped) to a reference domain.

For higher-order operators, such as the biharmonic operator, this usually causes
derivatives of the (inverse) metric to appear. The Laplacian in 1D, for example, becomes
```math
\begin{equation}
\Delta \phi^0 = \frac{1}{\sqrt{det(g)}} \left( \frac{\partial }{\partial \xi} \left (
\frac{1}{\sqrt{det(g)}} \right ) \frac{\partial^2 \phi}{\partial \xi^2} +
\frac{1}{\sqrt{det(g)}} \frac{\partial^2 \phi}{\partial \xi^2} \right )\;,
\end{equation}
```
while in 2D, it becomes
```math
\begin{equation}
\Delta \phi^0 = \frac{\partial^2 \phi^0}{\partial \xi^2} g^{1,1} +
\frac{\partial \phi^0}{\partial \xi}\frac{\partial g^{1,1}}{\partial \xi} + \frac{\partial^2 \phi^0}{\partial \xi \partial \eta} g^{1,2} +
\frac{\partial \phi^0}{\partial \eta}\frac{\partial g^{1,2}}{\partial \xi} + \frac{\partial^2 \phi^0}{\partial \eta \partial \xi} g^{2,1} +
\frac{\partial \phi^0}{\partial \xi}\frac{\partial g^{2,1}}{\partial \eta} + \frac{\partial^2 \phi^0}{\partial \eta^2} g^{2,2} +
\frac{\partial \phi^0}{\partial \eta}\frac{\partial g^{2,2}}{\partial \eta} + \frac{1}{\sqrt{det(g)}} \left (
(\frac{\partial \phi^0}{\partial \xi} g^{1,1} + \frac{\partial \phi^0}{\partial \eta} g^{1,2}) \frac{\partial}{\partial \xi}(\sqrt{det(g)}) +
(\frac{\partial \phi^0}{\partial \xi} g^{2,1} + \frac{\partial \phi^0}{\partial \eta} g^{2,2}) \frac{\partial}{\partial \eta}(\sqrt{det(g)}) \right) \;.
\end{equation}
```
As you can see, the expression for the Laplacian becomes rather involved. That is why
Mantis.jl allows you to compute these terms using automatic differentiation.

## Implementation

### The 1D case

Let's first look at the 1D version. We will start by creating a geometry (just a line in
1D) from 0.0 to 1.0 with 2 elements. On this geometry, we create a B-spline space of
maximally smooth quadratics (so, degree 2 and regularity 1). Note that since the
biharmonic problem requires the computation of the Laplacian, the regularity ``k`` must
be at least 1.

````@example Biharmonic
using Mantis
using DisplayAs #hide

starting_point = (0.0,)
box_size = (1.0,)
num_elements = (2,)
p = (2,)
k = (1,)

geometry = Geometry.create_cartesian_box(starting_point, box_size, num_elements)
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
````

Mantis works with forms, so we need to define the form space. In this case, we are
working with ``0``-forms, so we define the form space as follows.

````@example Biharmonic
Λ⁰ = Forms.FormSpace(0, geometry, B, "ϕ")
````

We define the weak form inputs. The weak form inputs contain the trial and test spaces,
the forcing function, and the quadrature rule. We define the forcing function as a
function of the coordinates. In this case, we define the forcing function as
``f^0 = 16 \pi^4 \sin(2 \pi x)``.

````@example Biharmonic
function forcing_function(x::Matrix{Float64})
    return [@. 16.0 * pi^4 * sin(2.0 * pi * x[:, 1])]
end
f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, "f⁰")
````

The quadrature rule is defined as a tensor product rule of the degree of the B-spline
space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.

````@example Biharmonic
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))
````

We define the weak form for the biharmonic problem. The weak form is defined as a function
that takes the weak form inputs and the quadrature rule as arguments. Note that this
function is dimension-agnostic. The laplacian is written as ``δd``.

````@example Biharmonic
function zero_form_biharmonic(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ψ⁰ = Assemblers.get_test_form(inputs)
    ϕ⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)

    A = ∫(δ(d(ψ⁰)) ∧ ★(δ(d(ϕ⁰))), dΩ)
    lhs_expression = ((A,),)

    b = ∫(ψ⁰ ∧ ★(f⁰), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end
````

We define the weak form inputs as a `WeakFormInputs` object. The weak form inputs contain
the trial and test spaces, and the forcing function. The trial and test spaces are the
same in this case, which is the default.

````@example Biharmonic
wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)
````

We can now assemble the linear system and solve it to obtain the solution. We define the
Dirichlet boundary conditions using the appropriate helper function.

````@example Biharmonic
bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
````

We assemble the linear system and solve it to obtain the solution. Note that we do not
need to redefine the weak form itself; it was already written in a dimension-independent
way.

````@example Biharmonic
lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A, b = Assemblers.assemble(weak_form, bc)
sol = vec(A \ b)
ϕ⁰ = Forms.build_form_field(Λ⁰, sol)
````

We want to plot the computed solution next to the exact solution, so we also create a
FormField for the exact solution.

````@example Biharmonic
function exact_solution(x::Matrix{Float64})
    return [@. sin(2.0 * pi * x[:, 1])]
end
ϕ_exact = Forms.AnalyticalFormField(0, exact_solution, geometry, "\\phi_{exact}")

fig = Mantis.Plot.plot_solution((ϕ⁰, ϕ_exact); title="Solution", ylabel=" ")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide
````

The above solution does not look very good. We only used two elements with a low
polynomial degree for our basis functions. If we use 16 elements instead, we should get a
much better approximation. So let's try that:

````@example Biharmonic
num_elements = (16,)
````

The code below is the same as before. We don't need to redefine any of the functions that
we created, those are still valid. We just rerun the rest.

````@example Biharmonic
geometry = Geometry.create_cartesian_box(starting_point, box_size, num_elements)
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)

Λ⁰ = Forms.FormSpace(0, geometry, B, "ϕ")

f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, "f⁰")

canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A, b = Assemblers.assemble(weak_form, bc)
sol = vec(A \ b)
ϕ⁰ = Forms.build_form_field(Λ⁰, sol)

ϕ_exact = Forms.AnalyticalFormField(0, exact_solution, geometry, "\\phi_{exact}")

fig = Mantis.Plot.plot_solution((ϕ⁰, ϕ_exact); title="Solution", ylabel=" ")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide
````

The above solution is indeed much closer (in the 'eyeball-norm') than before.

### The 2D case

The 2D case works in essentially the same way. All we have to do is ensure that our inputs
are now the 2D versions.

````@example Biharmonic
starting_point_2D = (0.0, 0.0)
box_size_2D = (1.0, 1.0)
num_elements_2D = (2, 2)
p_2D = (3, 3)
k_2D = (2, 2)

geometry_2D = Geometry.create_cartesian_box(starting_point_2D, box_size_2D, num_elements_2D)
B_2D = FunctionSpaces.create_bspline_space(
    starting_point_2D, box_size_2D, num_elements_2D, p_2D, k_2D
)
````

Mantis works with forms, so we need to define the form space. In this case, we are
working with ``0``-forms, so we define the form space as follows.

````@example Biharmonic
Λ⁰_2D = Forms.FormSpace(0, geometry_2D, B_2D, "label")
````

We define the weak form inputs. The weak form inputs contain the trial and test spaces,
the forcing function, and the quadrature rule. We define the forcing function as a
function of the coordinates. In this case, we define the forcing function as
``f^0 = 64 \pi^4 \sin(2 \pi x) \sin(2 \pi y)``.

````@example Biharmonic
function forcing_function_2D(x::Matrix{Float64})
    return [@. 64.0 * pi^4 * sin(2.0 * pi * x[:, 1]) * sin(2.0 * pi * x[:, 2])]
end
f⁰_2D = Forms.AnalyticalFormField(0, forcing_function_2D, geometry_2D, "f⁰")
````

The quadrature rule is defined as a tensor product rule of the degree of the B-spline
space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.

````@example Biharmonic
canonical_qrule_2D = Quadrature.tensor_product_rule(p_2D .+ 1, Quadrature.gauss_legendre)
dΩ_2D = Quadrature.StandardQuadrature(
    canonical_qrule_2D, Geometry.get_num_elements(geometry_2D)
)
````

We define the weak form inputs as a `WeakFormInputs` object. The weak form inputs contain
the trial and test spaces, and the forcing function. The trial and test spaces are the
same in this case, which is the default.

````@example Biharmonic
wfi_2D = Assemblers.WeakFormInputs(Λ⁰_2D, f⁰_2D)
````

We can now assemble the linear system and solve it to obtain the solution. We define the
Dirichlet boundary conditions using the appropriate helper function.

````@example Biharmonic
bc_2D = Forms.set_dirichlet_boundary_conditions(Λ⁰_2D, 0.0)
````

We assemble the linear system and solve it to obtain the solution. Note that we do not
need to redefine the weak form itself; it was already written in a dimensio-independent
way.

````@example Biharmonic
lhs_expressions_2D, rhs_expressions_2D = zero_form_biharmonic(wfi_2D, dΩ_2D)
weak_form_2D = Assemblers.WeakForm(lhs_expressions_2D, rhs_expressions_2D, wfi_2D)
A_2D, b_2D = Assemblers.assemble(weak_form_2D, bc_2D)
sol_2D = vec(A_2D \ b_2D)
ϕ⁰_2D = Forms.build_form_field(Λ⁰_2D, sol_2D)
````

We can now plot the solution using the `plot` function. This will write the output to a
VTK file that can be visualized using a VTK viewer, such as Paraview.

````@example Biharmonic
data_folder = joinpath(dirname(dirname(pathof(Mantis))), "examples", "data")
output_data_folder = joinpath(data_folder, "output", "HodgeLaplacian")
output_filename_2D = "Biharmonic-0form-Homogeneous-$(length(p))D.vtu"
output_file = joinpath(output_data_folder, output_filename_2D)
Plot.plot(
    ϕ⁰_2D;
    vtk_filename=output_file,
    n_subcells=1,
    degree=maximum(p),
    ascii=false,
    compress=false,
)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

