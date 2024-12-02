import Mantis

using BenchmarkTools

function create_bspline_space(x_left, x_right, n_elements, p, k)
    breakpoints = collect(LinRange(x_left, x_right, n_elements+1))
    patch = Mantis.Mesh.Patch1D(breakpoints)
    
    kvec = fill(k, (n_elements+1,))
    kvec[1] = -1 # Open knot vector
    kvec[end] = -1

    return Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
end

function fe_run(weak_form_inputs, weak_form, bc_dirichlet)
    A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, bc_dirichlet)

    # Solve & add bcs.
    sol = A \ b

    return sol
end

verbose = true

# ########################################################################
# ## Test cases for the 2D Poisson problem.                             ##
# ########################################################################
if verbose
    println("Creating 2D Geometry and spaces ... \n")
end

# Dimension
n_2d = 2
# Number of elements.
m_x = 10
m_y = 10
# polynomial degree and inter-element continuity.
p_2d = (4, 4)
k_2d = (3, 3)
# Domain.
const Lleft = 0.0
const Lright = 1.0
const Lbottom = 0.0
const Ltop = 1.0


# Create function spaces (b-splines here).
trial_space_x = create_bspline_space(Lleft, Lright, m_x, p_2d[1], k_2d[1])
trial_space_y = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2], k_2d[2])
trial_space_x_pm1 = create_bspline_space(Lleft, Lright, m_x, p_2d[1]-1, k_2d[1]-1)
trial_space_y_pm1 = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2]-1, k_2d[2]-1)

test_space_x = create_bspline_space(Lleft, Lright, m_x, p_2d[1], k_2d[1])
test_space_y = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2], k_2d[2])
test_space_x_pm1 = create_bspline_space(Lleft, Lright, m_x, p_2d[1]-1, k_2d[1]-1)
test_space_y_pm1 = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2]-1, k_2d[2]-1)

trial_space_2d_volume = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x_pm1, trial_space_y_pm1)
test_space_2d_volume = Mantis.FunctionSpaces.TensorProductSpace(test_space_x_pm1, test_space_y_pm1)

trial_space_2d_1_form_x = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x_pm1, trial_space_y)
trial_space_2d_1_form_y = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y_pm1)
test_space_2d_1_form_x = Mantis.FunctionSpaces.TensorProductSpace(test_space_x_pm1, test_space_y)
test_space_2d_1_form_y = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y_pm1)

trial_space_2d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y)
test_space_2d = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y)

# Set Dirichlet boundary conditions to zero.
bc_dirichlet_2d = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in trial_space_2d.dof_partition[1][j])
bc_dirichlet_2d_empty = Dict{Int, Float64}()

# Create the geometries.
brk_2d_x = collect(LinRange(Lleft, Lright, m_x+1))
brk_2d_y = collect(LinRange(Lbottom, Ltop, m_y+1))
geom_cartesian = Mantis.Geometry.CartesianGeometry((brk_2d_x, brk_2d_y))

const crazy_c = 0.2
function mapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function dmapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
end
dimension = (n_2d, n_2d)
curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
geom_crazy = Mantis.Geometry.MappedGeometry(geom_cartesian, curved_mapping)


# Setup the quadrature rule.
q_rule_2d = Mantis.Quadrature.tensor_product_rule((p_2d[1] + 1, p_2d[2] + 1), Mantis.Quadrature.gauss_legendre)

# Create form spaces (both test and trial)
# Cartesian mesh
zero_form_space_trial_2d_cart = Mantis.Forms.FormSpace(0, geom_cartesian, (trial_space_2d,), "φ")
zero_form_space_test_2d_cart = Mantis.Forms.FormSpace(0, geom_cartesian, (test_space_2d,), "ϕ")

two_form_space_trial_2d_cart = Mantis.Forms.FormSpace(2, geom_cartesian, (trial_space_2d_volume,), "φ")
two_form_space_test_2d_cart = Mantis.Forms.FormSpace(2, geom_cartesian, (test_space_2d_volume,), "ϕ")

one_form_space_trial_2d_cart = Mantis.Forms.FormSpace(1, geom_cartesian, (trial_space_2d_1_form_x, trial_space_2d_1_form_y), "u")
one_form_space_test_2d_cart = Mantis.Forms.FormSpace(1, geom_cartesian, (test_space_2d_1_form_x, test_space_2d_1_form_y), "q")

# Crazy mesh
zero_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(0, geom_crazy, (trial_space_2d,), "φ")
zero_form_space_test_2d_crazy = Mantis.Forms.FormSpace(0, geom_crazy, (test_space_2d,), "ϕ")

two_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(2, geom_crazy, (trial_space_2d_volume,), "φ")
two_form_space_test_2d_crazy = Mantis.Forms.FormSpace(2, geom_crazy, (test_space_2d_volume,), "ϕ")

one_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(1, geom_crazy, (trial_space_2d_1_form_x, trial_space_2d_1_form_y), "u")
one_form_space_test_2d_crazy = Mantis.Forms.FormSpace(1, geom_crazy, (test_space_2d_1_form_x, test_space_2d_1_form_y), "q")

# Create the forcing forms
function forcing_function_const_2d(x::Matrix{Float64})
    return [ones(size(x, 1))]
end

f⁰_cart_const_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_const_2d, geom_cartesian, "f")
f²_cart_const_2d = Mantis.Forms.AnalyticalFormField(2, forcing_function_const_2d, geom_cartesian, "f")

f⁰_crazy_const_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_const_2d, geom_crazy, "f")
f²_crazy_const_2d = Mantis.Forms.AnalyticalFormField(2, forcing_function_const_2d, geom_crazy, "f")


function forcing_function_sine_2d(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end

f⁰_cart_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, geom_cartesian, "f")
f²_cart_sine_2d = Mantis.Forms.AnalyticalFormField(2, forcing_function_sine_2d, geom_cartesian, "f")

f⁰_crazy_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, geom_crazy, "f")
f²_crazy_sine_2d = Mantis.Forms.AnalyticalFormField(2, forcing_function_sine_2d, geom_crazy, "f")


# Create the exact_solutions as appropriate form.
function exact_sol_sine_2d(x::Matrix{Float64})
    return [@. sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end

function exact_sol_sine_2d_grad(x::Matrix{Float64})
    return [-2.0.*pi.*sinpi.(2.0 .* x[:,1]).*cospi.(2.0 .* x[:,2]), 2.0.*pi.*cospi.(2.0 .* x[:,1]).*sinpi.(2.0 .* x[:,2])]
end

sol⁰_cart_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, geom_cartesian, "sol")
sol¹_cart_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(1, exact_sol_sine_2d_grad, geom_cartesian, "sol")
sol²_cart_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(2, exact_sol_sine_2d, geom_cartesian, "sol")

sol⁰_crazy_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, geom_crazy, "sol")
sol¹_crazy_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(1, exact_sol_sine_2d_grad, geom_crazy, "sol")
sol²_crazy_sine_2d_exact_sol = Mantis.Forms.AnalyticalFormField(2, exact_sol_sine_2d, geom_crazy, "sol")

########################################################################
## Test cases for the 3D Poisson problem.                             ##
########################################################################

if verbose
    println("Creating 3D Geometry and spaces ...")
end

# Dimension
n_3d = 3
# Number of elements.
m_3d_x = 5
m_3d_y = 5
m_3d_z = 5
# polynomial degree and inter-element continuity.
p_3d = (3, 4, 1)
k_3d = (2, 2, 0)
# Domain. The length of the domain is chosen so that the normal 
# derivatives of the exact solution are zero at the boundary. This is 
# the only Neumann b.c. that we can specify at the moment.
const Lx1 = 0.0
const Lx2 = 1.0
const Ly1 = 0.0
const Ly2 = 1.0
const Lz1 = 0.0
const Lz2 = 1.0

# Tensor product b-spline case on a Cartesian geometry.
# Create Patch.
brk_3d_x = collect(LinRange(Lx1, Lx2, m_3d_x+1))
brk_3d_y = collect(LinRange(Ly1, Ly2, m_3d_y+1))
brk_3d_z = collect(LinRange(Lz1, Lz2, m_3d_z+1))
patch_3d_x = Mantis.Mesh.Patch1D(brk_3d_x)
patch_3d_y = Mantis.Mesh.Patch1D(brk_3d_y)
patch_3d_z = Mantis.Mesh.Patch1D(brk_3d_z)
# Continuity vector for OPEN knot vector.
kvec_3d_x = fill(k_3d[1], (m_3d_x+1,))
kvec_3d_x[1] = -1
kvec_3d_x[end] = -1
kvec_3d_y = fill(k_3d[2], (m_3d_y+1,))
kvec_3d_y[1] = -1
kvec_3d_y[end] = -1
kvec_3d_z = fill(k_3d[3], (m_3d_z+1,))
kvec_3d_z[1] = -1
kvec_3d_z[end] = -1
# Create function spaces (b-splines here).
trial_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
test_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
trial_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
test_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
trial_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)
test_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)

trial_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_x, trial_space_3d_y)
test_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_x, test_space_3d_y)

trial_space_3d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_xy, trial_space_3d_z)
test_space_3d = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_xy, test_space_3d_z)

# Set Dirichlet boundary conditions to zero.
bc_dirichlet_3d = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] for i in trial_space_3d.dof_partition[1][j])

# Create the geometry.
geom_3d_cartesian = Mantis.Geometry.CartesianGeometry((brk_3d_x, brk_3d_y, brk_3d_z))

# Setup the quadrature rule.
q_rule_3d = Mantis.Quadrature.tensor_product_rule(p_3d .+ 1, Mantis.Quadrature.gauss_legendre)

# Create form spaces (both test and trial)
# Cartesian mesh
zero_form_space_trial_3d_cart = Mantis.Forms.FormSpace(0, geom_3d_cartesian, (trial_space_3d,), "φ")
zero_form_space_test_3d_cart = Mantis.Forms.FormSpace(0, geom_3d_cartesian, (test_space_3d,), "ϕ")


function exact_sol_sine_3d(x::Float64, y::Float64, z::Float64)
    return sinpi(2.0 * x) * sinpi(2.0 * y) * sinpi(2.0 * z)
end

# Create the forcing form.
function forcing_function_sine_3d(x::Matrix{Float64})
    return [@. 12.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2]) * sinpi(2.0 * x[:,3])]
end
f⁰_cart_sine_3d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_3d, geom_3d_cartesian, "f")

# Create the exact_solutions as appropriate form.
function exact_sol_sine_3d(x::Matrix{Float64})
    return [@. sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2]) * sinpi(2.0 * x[:,3])]
end
sol⁰_cart_sine_3d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_3d, geom_3d_cartesian, "sol")

cases = ["const2d-Dirichlet", "const2d-Dirichlet-crazy", "sine2d-Dirichlet", "sine2d-Dirichlet-crazy", "sine2d-Dirichlet-mixed", "sine2d-Dirichlet-mixed-crazy", "sine3d-Dirichlet"]
benchmarks = []
for case in cases
    if case == "const2d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_const_2d, zero_form_space_trial_2d_cart, zero_form_space_test_2d_cart, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d))

    elseif case == "const2d-Dirichlet-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_crazy_const_2d, zero_form_space_trial_2d_crazy, zero_form_space_test_2d_crazy, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d))

    elseif case == "sine2d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_sine_2d, zero_form_space_trial_2d_cart, zero_form_space_test_2d_cart, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d))       

    elseif case == "sine2d-Dirichlet-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_crazy_sine_2d, zero_form_space_trial_2d_crazy, zero_form_space_test_2d_crazy, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d))

    elseif case == "sine2d-Dirichlet-mixed"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(f²_cart_sine_2d, one_form_space_trial_2d_cart, two_form_space_trial_2d_cart, one_form_space_test_2d_cart, two_form_space_test_2d_cart, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty))

    elseif case == "sine2d-Dirichlet-mixed-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(f²_crazy_sine_2d, one_form_space_trial_2d_crazy, two_form_space_trial_2d_crazy, one_form_space_test_2d_crazy, two_form_space_test_2d_crazy, q_rule_2d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty))

    elseif case == "sine3d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_sine_3d, zero_form_space_trial_3d_cart, zero_form_space_test_3d_cart, q_rule_3d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_3d)
        push!(benchmarks, @benchmark fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_3d))

    end
end

if verbose
    for id ∈ eachindex(cases)
        println("Benchmark for case: ", cases[id])
        display(benchmarks[id])
        println("\n")
    end
end