# # Biharmonic

# The biharmonic problem is a typical example of a higher-order PDE problem. It can appear
# in, for example, elasticity and fluid flow problems. In this example, we will briefly
# review what the biharmonic problem looks like, and we will implement it using `Mantis.jl`.

# ## Formulation

# ### The 1D case.

# In 1D the biharmonic problem is defined as:
# ```math
# \begin{alignat*}{2}
#     &\frac{\partial^4 \phi(x)}{\partial x^4} = - f(x)  \quad &&\text{for}\ x \in [0, L] \;, \\
#     &\phi(0) = \phi(L) = 0 \;, \\
#     &\frac{\partial^2 \phi}{\partial x^2}(0) = \frac{\partial^2 \phi}{\partial x^2}(L) = 0 \;,
# \end{alignat*}
# ```
# where we are looking for a function ``\phi(x)`` whose fourth derivative equals negative
# ``f(x)``, the forcing function. We have chosen the domain to be ``[0, L]`` with ``L``
# some length. Note that there are now two boundary conditions, since this is a higher
# order equation. In this example, we'll use homogeneous boundary conditions on the value
# and second derivatives.
# The weak formulation is then as follows.
# ```math
# \begin{gather*}
# \text{Given}\ f \in L^2 ([0, L]),\ \text{find}\ \psi \in H^2_{h,0}([0, L])\
# \text{such that} \\
# \int_0^L \frac{\partial^2 \psi}{\partial x^2} \frac{\partial^2 \phi}{\partial x^2} dx =
# \int_{\Omega} \psi f dx \quad  \forall \ \psi \in H^2_{h,0}([0, L]) \;.
# \end{gather*}
# ```
# For this problem, the homogeneous dirichlet boundary condition is an essential boundary
# condition, while the homogeneous boundary condition on the laplacian is a natural
# boundary condition which is already included in the above weak formulation.

# ### The differential form case in nD.

# Since `Mantis.jl` is designed to deal with differential forms, we prefer to work with the
# differential form formulation of the biharmonic problem. The 1D example above is the 1D
# version of the ``0``-form biharmonic problem with homogeneous boundary conditions on the
# value and laplacian. The ``0``-form biharmonic problem in ``n``-dimensions on domain
# ``\Omega \subset \mathbb{R}^n`` with boundary ``\partial \Omega`` is
# ```math
# \begin{alignat*}{2}
#     &\Delta^2 \phi^0 = - f^0  \quad &&\text{on}\ \Omega \;, \\
#     &tr(\phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;, \\
#     &tr(\Delta \phi^0) = 0  \quad &&\text{on}\ \partial\Omega \;.
# \end{alignat*}
# ```
# The weak formulation is then as follows.
# ```math
# \begin{gather*}
# \text{Given}\ f^0 \in L^2 \Lambda^0 (\Omega),\ \text{find}\ \phi^0 \in H^2
# \Lambda^0_h (\Omega)\ \text{such that} \\
# \int_{\Omega} \Delta \psi^0 \wedge \star \Delta \phi^0 = \int_{\Omega} \psi^0
# \wedge \star f^0 \quad  \forall \ \psi^0 \in H^2\Lambda^0_{h,0} (\Omega) \;.
# \end{gather*}
# ```

# ### What is actually computed?
# In many finite element codes, and this is also true for `Mantis.jl`, the integrals in the
# above weak formulations are not directly computed on the given domain. Instead, they are
# pulled-back (mapped) to a reference domain.
#
# For higher-order operators, such as the biharmonic operator, this usually causes
# derivatives of the (inverse) metric to appear. The Laplacian in 1D, for example, becomes
# (assuming we have a smooth enough mapping ``\Phi: \xi \to x``)
# ```math
# \begin{equation}
# \Delta \phi^0 = \frac{1}{\sqrt{det(g)}} \left( \frac{\partial }{\partial \xi} \left (
# \frac{1}{\sqrt{det(g)}} \right ) \frac{\partial^2 \phi}{\partial \xi^2} +
# \frac{1}{\sqrt{det(g)}} \frac{\partial^2 \phi}{\partial \xi^2} \right )\;,
# \end{equation}
# ```
# while in 2D, it becomes (assuming we have a smooth enough mapping ``\Phi: (\xi, \eta) \to
# (x, y)``
# ```math
# \begin{equation}
# \Delta \phi^0 = \frac{\partial^2 \phi^0}{\partial \xi^2} g^{1,1} +
# \frac{\partial \phi^0}{\partial \xi}\frac{\partial g^{1,1}}{\partial \xi} + \frac{\partial^2 \phi^0}{\partial \xi \partial \eta} g^{1,2} +
# \frac{\partial \phi^0}{\partial \eta}\frac{\partial g^{1,2}}{\partial \xi} + \frac{\partial^2 \phi^0}{\partial \eta \partial \xi} g^{2,1} +
# \frac{\partial \phi^0}{\partial \xi}\frac{\partial g^{2,1}}{\partial \eta} + \frac{\partial^2 \phi^0}{\partial \eta^2} g^{2,2} +
# \frac{\partial \phi^0}{\partial \eta}\frac{\partial g^{2,2}}{\partial \eta} + \frac{1}{\sqrt{det(g)}} \left (
# (\frac{\partial \phi^0}{\partial \xi} g^{1,1} + \frac{\partial \phi^0}{\partial \eta} g^{1,2}) \frac{\partial}{\partial \xi}(\sqrt{det(g)}) +
# (\frac{\partial \phi^0}{\partial \xi} g^{2,1} + \frac{\partial \phi^0}{\partial \eta} g^{2,2}) \frac{\partial}{\partial \eta}(\sqrt{det(g)}) \right) \;.
# \end{equation}
# ```
# These expressions are for the Laplacian applied to ``0``-forms, in which case
# ```math
# \Delta = \delta \mathrm{d} = \star \mathrm{d} \star \mathrm{d} \;,
# ```
# with ``\delta`` the codifferential.
#
# These expressions tend to become rather complex. Fortunately, the user does not have to
# implement such transformations. `Mantis.jl` can handle this automatically.

# ## Implementation

# ### The 1D case

# Let's first look at the 1D version. We will start by creating a geometry (just a line in
# 1D) from 0.0 to 1.0 with 2 elements. On this geometry, we create a B-spline space of
# maximally smooth quadratics (so, degree 2 and regularity 1). Note that since the
# biharmonic problem requires the computation of the Laplacian, the regularity ``k`` must
# be at least 1.

using Mantis
using GLMakie # Used for plotting later on. It also allows the use of L"".
using DisplayAs #hide

starting_point = (0.0,)
box_size = (1.0,)
num_elements = (2,)
p = (3,)
k = (2,)

geometry = Geometry.create_cartesian_box(starting_point, box_size, num_elements)
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)

# Mantis works with forms, so we need to define the form space. In this case, we are
# working with ``0``-forms, so we define the form space as follows.

Λ⁰ = Forms.FormSpace(0, B, L"\phi_h")

# We define the weak form inputs. The weak form inputs contain the trial and test spaces,
# the forcing function, and the quadrature rule. We define the forcing function as a
# function of the coordinates. In this case, we define the forcing function as
# ``f^0 = 16 \pi^4 \sin(2 \pi x)``.

function forcing_function(x::Matrix{Float64})
    return [@. 16.0 * pi^4 * sin(2.0 * pi * x[:, 1])]
end
f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, L"f^0")

# The quadrature rule is defined as a tensor product rule of the degree of the B-spline
# space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

# We define the weak form for the biharmonic problem. The weak form is defined as a function
# that takes the weak form inputs and the quadrature rule as arguments. Note that this
# function is dimension-agnostic. The laplacian is written as ``δd``.

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

# We define the weak form inputs as a `WeakFormInputs` object. The weak form inputs contain
# the trial and test spaces, and the forcing function. The trial and test spaces are the
# same in this case, which is the default.
wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

# We can now assemble the linear system and solve it to obtain the solution. We define the
# Dirichlet boundary conditions using the appropriate helper function.

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

# We assemble the linear system and solve it to obtain the solution. Note that we do not
# need to redefine the weak form itself; it was already written in a dimension-independent
# way.

lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A, b = Assemblers.assemble(weak_form, bc)
sol = vec(A \ b)
ϕ⁰ = Forms.build_form_field(Λ⁰, sol)

# We want to plot the computed solution next to the exact solution, so we also create a
# FormField for the exact solution.
function exact_solution(x::Matrix{Float64})
    return [@. sin(2.0 * pi * x[:, 1])]
end
ϕ_exact = Forms.AnalyticalFormField(0, exact_solution, geometry, L"\phi_{exact}")

fig = Mantis.Plot.plot_solution((ϕ⁰, ϕ_exact); title="Solution", ylabel=" ")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# The above solution does not look very good. We only used two elements with a low
# polynomial degree for our basis functions. If we use 16 elements instead, we should get a
# much better approximation. So let's try that:
num_elements = (16,)

# The code below is the same as before. We don't need to redefine any of the functions that
# we created, those are still valid. We just rerun the rest.
geometry = Geometry.create_cartesian_box(starting_point, box_size, num_elements)
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)

Λ⁰ = Forms.FormSpace(0, B, L"\phi_h")

f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, L"f^0")

canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A, b = Assemblers.assemble(weak_form, bc)
sol = vec(A \ b)
ϕ⁰ = Forms.build_form_field(Λ⁰, sol)

ϕ_exact = Forms.AnalyticalFormField(0, exact_solution, geometry, L"\phi_{exact}")

fig = Mantis.Plot.plot_solution((ϕ⁰, ϕ_exact); title="Solution", ylabel=" ")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# The above solution is indeed much closer (in the 'eyeball-norm') than before. We can make
# this more precise by computing the error in the ``L^2``-norm and plotting how this error
# decreases as we choose increasingly fine meshes. This is called a convergence study.

# #### Convergence studies for the 1D case.

# One way to obtain some confirmation that these results are what we expect, we can compute
# the error on finer and finer meshes. To start, we create a function that will create all
# objects based on the geometry that we provide, and that will compute the error using the
# given geometry.
canonical_qrule_analysis = Quadrature.tensor_product_rule(
    3 .* p .+ 1, Quadrature.gauss_legendre
)

function compute_error_biharmonic(geometry, function_space)
    Λ⁰ = Forms.FormSpace(0, function_space, L"\phi_h")

    f⁰ = Forms.AnalyticalFormField(0, forcing_function, geometry, L"f^0")

    dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))

    wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)

    bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)

    lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
    A, b = Assemblers.assemble(weak_form, bc)
    sol = vec(A \ b)
    ϕ⁰ = Forms.build_form_field(Λ⁰, sol)

    ϕ⁰_exact = Forms.AnalyticalFormField(0, exact_solution, geometry, L"\phi_{exact}")

    dΩ_analysis = Quadrature.StandardQuadrature(
        canonical_qrule_analysis, Geometry.get_num_elements(geometry)
    )

    return Analysis.L2_norm(ϕ⁰ - ϕ⁰_exact, dΩ_analysis)
end

# To study the effect of introducing a different spacing of the elements in the grid, we can
# introduce a mapping. Here, we pick the mapping from the
# [One-dimensional mapped geometry](@ref) example. Do note that this will lead to a nearly
# singular system.
const exponent = 3
mapping(ξ̂) = ξ̂^exponent
d_map(ξ̂) = exponent * ξ̂^(exponent - 1)
d2_map(ξ̂) = exponent * (exponent - 1) * ξ̂^(exponent - 2)
full_map = Geometry.Mapping(
    (1, 1), ξ̂ -> mapping.(ξ̂[:, 1]), ξ̂ -> d_map.(ξ̂[:, 1]), ξ̂ -> d2_map.(ξ̂[:, 1])
)

# Then, we create a loop to build geometries with increasingly many elements and to compute
# the biharmonic equation with them.
num_elements_study = [(4,), (8,), (16,), (32,), (64,)]
h = Vector{Float64}(undef, length(num_elements_study))
errors_cartesian = Vector{Float64}(undef, length(num_elements_study))
errors_mapped = Vector{Float64}(undef, length(num_elements_study))
for i in eachindex(num_elements_study)
    geo_cartesian = Geometry.create_cartesian_box(
        starting_point, box_size, num_elements_study[i]
    )
    geo_mapped = Geometry.MappedGeometry(geo_cartesian, full_map)

    space_cartesian = FunctionSpaces.create_bspline_space(
        starting_point, box_size, num_elements_study[i], p, k
    )
    space_curvilinear = FunctionSpaces.BSplineSpace(geo_cartesian, full_map, p[1], k[1])

    h[i] = box_size[1] / num_elements_study[i][1]
    errors_cartesian[i] = compute_error_biharmonic(geo_cartesian, space_cartesian)
    errors_mapped[i] = compute_error_biharmonic(geo_mapped, space_curvilinear)
end

# We can then plot the results using GLMakie

fig2 = Figure()
ax2 = Axis(
    fig2[1, 1]; xlabel=L"h", ylabel=L"||ϕ_h - ϕ_{exact}||_{L^2}", xscale=log10, yscale=log10
)

scatterlines!(
    ax2, h, errors_mapped; label="Mapped", color=:blue, marker=:rect, markersize=10
)
C = errors_mapped[3] / (h[3]^4)
lines!(ax2, h, C .* (h .^ 4); label=L"O(h^4)", linestyle=:dot, color=:black)

scatterlines!(
    ax2, h, errors_cartesian; label="Cartesian", color=:red, marker=:circle, markersize=10
)
C2 = errors_cartesian[3] / (h[3]^4)
lines!(ax2, h, C2 .* (h .^ 4); label=L"O(h^4)", linestyle=:dash, color=:black)

fig2[1, 2] = Legend(fig2, ax2)
fig2 = DisplayAs.Text(DisplayAs.PNG(fig2)) #hide

# For this setup, we expect the error to decrease with a rate of ``p+1``, so 4. This is
# indeed confirmed by looking at the plot.

# ### The 2D case

# The 2D case works in essentially the same way. All we have to do is ensure that our inputs
# are now the 2D versions.

starting_point_2D = (0.0, 0.0)
box_size_2D = (1.0, 1.0)
num_elements_2D = (8, 8)
p_2D = (3, 3)
k_2D = (2, 2)

geometry_2D = Geometry.create_cartesian_box(starting_point_2D, box_size_2D, num_elements_2D)
B_2D = FunctionSpaces.create_bspline_space(
    starting_point_2D, box_size_2D, num_elements_2D, p_2D, k_2D
)

# Mantis works with forms, so we need to define the form space. In this case, we are
# working with ``0``-forms, so we define the form space as follows.

Λ⁰_2D = Forms.FormSpace(0, B_2D, L"\phi_h")

# We define the weak form inputs. The weak form inputs contain the trial and test spaces,
# the forcing function, and the quadrature rule. We define the forcing function as a
# function of the coordinates. In this case, we define the forcing function as
# ``f^0 = 64 \pi^4 \sin(2 \pi x) \sin(2 \pi y)``.

function forcing_function_2D(x::Matrix{Float64})
    return [@. 64.0 * pi^4 * sin(2.0 * pi * x[:, 1]) * sin(2.0 * pi * x[:, 2])]
end
f⁰_2D = Forms.AnalyticalFormField(0, forcing_function_2D, geometry_2D, L"f^0")

# The quadrature rule is defined as a tensor product rule of the degree of the B-spline
# space plus one. In this case, we define the quadrature rule as a Gauss-Legendre rule.
canonical_qrule_2D = Quadrature.tensor_product_rule(p_2D .+ 1, Quadrature.gauss_legendre)
dΩ_2D = Quadrature.StandardQuadrature(
    canonical_qrule_2D, Geometry.get_num_elements(geometry_2D)
)

# We define the weak form inputs as a `WeakFormInputs` object. The weak form inputs contain
# the trial and test spaces, and the forcing function. The trial and test spaces are the
# same in this case, which is the default.
wfi_2D = Assemblers.WeakFormInputs(Λ⁰_2D, f⁰_2D)

# We can now assemble the linear system and solve it to obtain the solution. We define the
# Dirichlet boundary conditions using the appropriate helper function.

bc_2D = Forms.set_dirichlet_boundary_conditions(Λ⁰_2D, 0.0)

# We assemble the linear system and solve it to obtain the solution. Note that we do not
# need to redefine the weak form itself; it was already written in a dimensio-independent
# way.

lhs_expressions_2D, rhs_expressions_2D = zero_form_biharmonic(wfi_2D, dΩ_2D)
weak_form_2D = Assemblers.WeakForm(lhs_expressions_2D, rhs_expressions_2D, wfi_2D)
A_2D, b_2D = Assemblers.assemble(weak_form_2D, bc_2D)
sol_2D = vec(A_2D \ b_2D)
ϕ⁰_2D = Forms.build_form_field(Λ⁰_2D, sol_2D)

# Knowing the exact solutions, we can compute the ``L^2``-error.
function exact_solution_2D(x::Matrix{Float64})
    return [@. sin(2.0 * pi * x[:, 1]) * sin(2.0 * pi * x[:, 2])]
end
ϕ⁰_exact_2D = Forms.AnalyticalFormField(0, exact_solution_2D, geometry_2D, L"\phi_{exact}")

canonical_qrule_2D_analysis = Quadrature.tensor_product_rule(
    3 .* p_2D .+ 1, Quadrature.gauss_legendre
)
dΩ_2D_analysis = Quadrature.StandardQuadrature(
    canonical_qrule_2D_analysis, Geometry.get_num_elements(geometry_2D)
)

println("L2 error: ", Analysis.L2_norm(ϕ⁰_2D - ϕ⁰_exact_2D, dΩ_2D_analysis))

# We can also write the output to a VTK file that can be visualized using a VTK viewer, such
# as Paraview. Note that we export both the computed and exact solutions.
output_filename_2D = "Biharmonic-0form-Homogeneous-Cartesian-$(length(p_2D))D"
Mantis.Plot.export_form_fields_to_vtk(
    (ϕ⁰_2D, ϕ⁰_exact_2D),
    output_filename_2D;
    output_directory_tree=["examples", "data", "output", "Biharmonic"],
)

# #### The 2D case on a more complicated geometry.

# If we want to use a different geometry instead, we create a space on the new geometry and
# reuse all other code.

mapping_2D_curv = Geometry.create_curvilinear_mapping(starting_point_2D, box_size_2D)
geometry_2D_curv = Geometry.MappedGeometry(geometry_2D, mapping_2D_curv)
B_2D_curv = FunctionSpaces.TensorProductSpace(
    FunctionSpaces.get_constituent_spaces(B_2D), Geometry.CartesianGeometry, mapping_2D_curv
)

Λ⁰_2D_curv = Forms.FormSpace(0, B_2D_curv, L"\phi_h")

f⁰_2D_curv = Forms.AnalyticalFormField(0, forcing_function_2D, geometry_2D_curv, L"f^0")

dΩ_2D_curv = Quadrature.StandardQuadrature(
    canonical_qrule_2D, Geometry.get_num_elements(geometry_2D_curv)
)

wfi_2D_curv = Assemblers.WeakFormInputs(Λ⁰_2D_curv, f⁰_2D_curv)

bc_2D_curv = Forms.set_dirichlet_boundary_conditions(Λ⁰_2D_curv, 0.0)

lhs_expressions_2D_curv, rhs_expressions_2D_curv = zero_form_biharmonic(
    wfi_2D_curv, dΩ_2D_curv
)
weak_form_2D_curv = Assemblers.WeakForm(
    lhs_expressions_2D_curv, rhs_expressions_2D_curv, wfi_2D_curv
)
A_2D_curv, b_2D_curv = Assemblers.assemble(weak_form_2D_curv, bc_2D_curv)
sol_2D_curv = vec(A_2D_curv \ b_2D_curv)
ϕ⁰_2D_curv = Forms.build_form_field(Λ⁰_2D_curv, sol_2D_curv)

ϕ⁰_exact_2D_curv = Forms.AnalyticalFormField(
    0, exact_solution_2D, geometry_2D_curv, L"\phi_{exact, curv}"
)

dΩ_2D_analysis_curv = Quadrature.StandardQuadrature(
    canonical_qrule_2D_analysis, Geometry.get_num_elements(geometry_2D_curv)
)

println("L2 error: ", Analysis.L2_norm(ϕ⁰_2D_curv - ϕ⁰_exact_2D_curv, dΩ_2D_analysis_curv))

output_filename_2D_curv = "Biharmonic-0form-Homogeneous-Curvilinear-$(length(p_2D))D"
Mantis.Plot.export_form_fields_to_vtk(
    (ϕ⁰_2D_curv, ϕ⁰_exact_2D_curv),
    output_filename_2D_curv;
    output_directory_tree=["examples", "data", "output", "Biharmonic"],
)

# #### Convergence studies for the 2D case.

# One way to obtain some confirmation that these results are what we expect, we can compute
# the error on finer and finer meshes. To start, we create a function that will create all
# objects based on the geometry that we provide, and that will compute the error using the
# given geometry.

function compute_error_biharmonic_2D(geometry, function_space)
    Λ⁰_2D = Forms.FormSpace(0, function_space, "ϕ")

    f⁰_2D = Forms.AnalyticalFormField(0, forcing_function_2D, geometry, "f⁰")

    dΩ_2D = Quadrature.StandardQuadrature(
        canonical_qrule_2D, Geometry.get_num_elements(geometry)
    )

    wfi_2D = Assemblers.WeakFormInputs(Λ⁰_2D, f⁰_2D)

    bc_2D = Forms.set_dirichlet_boundary_conditions(Λ⁰_2D, 0.0)

    lhs_expressions_2D, rhs_expressions_2D = zero_form_biharmonic(wfi_2D, dΩ_2D)
    weak_form_2D = Assemblers.WeakForm(lhs_expressions_2D, rhs_expressions_2D, wfi_2D)
    A_2D, b_2D = Assemblers.assemble(weak_form_2D, bc_2D)
    sol_2D = vec(A_2D \ b_2D)
    ϕ⁰_2D = Forms.build_form_field(Λ⁰_2D, sol_2D)

    ϕ⁰_exact_2D = Forms.AnalyticalFormField(0, exact_solution_2D, geometry, "ϕ_exact")

    dΩ_2D_analysis = Quadrature.StandardQuadrature(
        canonical_qrule_2D_analysis, Geometry.get_num_elements(geometry)
    )

    return Analysis.L2_norm(ϕ⁰_2D - ϕ⁰_exact_2D, dΩ_2D_analysis)
end

# Then, we create a loop to build geometries with increasingly many elements and to compute
# the biharmonic equation with them.
num_elements_study_2D = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
h_2D = Vector{Float64}(undef, length(num_elements_study_2D))
errors_cartesian_2D = Vector{Float64}(undef, length(num_elements_study_2D))
errors_curvilinear_2D = Vector{Float64}(undef, length(num_elements_study_2D))
for i in eachindex(num_elements_study_2D)
    geo_cartesian = Geometry.create_cartesian_box(
        starting_point_2D, box_size_2D, num_elements_study_2D[i]
    )
    curved_mapping = Geometry.create_curvilinear_mapping(starting_point_2D, box_size_2D)
    geo_curvilinear = Geometry.MappedGeometry(geo_cartesian, curved_mapping)

    space_cartesian = FunctionSpaces.create_bspline_space(
        starting_point_2D, box_size_2D, num_elements_study_2D[i], p_2D, k_2D
    )
    space_curvilinear = FunctionSpaces.TensorProductSpace(
        FunctionSpaces.get_constituent_spaces(space_cartesian),
        Geometry.CartesianGeometry,
        curved_mapping,
    )

    h_2D[i] = box_size_2D[1] / num_elements_study_2D[i][1]
    errors_cartesian_2D[i] = compute_error_biharmonic_2D(geo_cartesian, space_cartesian)
    errors_curvilinear_2D[i] = compute_error_biharmonic_2D(
        geo_curvilinear, space_curvilinear
    )
end

# We can then plot the results using GLMakie again.

fig3 = Figure()
ax3 = Axis(
    fig3[1, 1]; xlabel=L"h", ylabel=L"||ϕ_h - ϕ_{exact}||_{L^2}", xscale=log10, yscale=log10
)

scatterlines!(
    ax3,
    h_2D,
    errors_curvilinear_2D;
    label="Curvilinear",
    color=:blue,
    marker=:rect,
    markersize=10,
)
C = errors_curvilinear_2D[3] / (h[3]^4)
lines!(ax3, h_2D, C .* (h_2D .^ 4); label=L"O(h^4)", linestyle=:dot, color=:black)

scatterlines!(
    ax3,
    h_2D,
    errors_cartesian_2D;
    label="Cartesian",
    color=:red,
    marker=:circle,
    markersize=10,
)
C2 = errors_cartesian_2D[3] / (h[3]^4)
lines!(ax3, h_2D, C2 .* (h_2D .^ 4); label=L"O(h^4)", linestyle=:dash, color=:black)

fig3[1, 2] = Legend(fig3, ax3)
fig3 = DisplayAs.Text(DisplayAs.PNG(fig3)) #hide

# For this setup, we expect the error to decrease with a rate of ``p+1``, so 4. This is
# indeed confirmed by looking at the plot.
