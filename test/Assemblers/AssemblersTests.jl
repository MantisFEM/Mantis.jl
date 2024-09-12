

import Mantis

using Test
using LinearAlgebra



function create_bspline_space(x_left, x_right, n_elements, p, k)
    breakpoints = collect(LinRange(x_left, x_right, n_elements+1))
    patch = Mantis.Mesh.Patch1D(breakpoints)
    
    kvec = fill(k, (n_elements+1,))
    kvec[1] = -1 # Open knot vector
    kvec[end] = -1

    return Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
end




# This is how MANTIS can be called to solve a problem.
function fe_run(weak_form_inputs, weak_form, bc_dirichlet, case, test, verbose)
    if verbose
        println("Running case "*case*" ...")
    end

    if verbose
        println("Assembling ...")
    end
    A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, bc_dirichlet)

    # if n > 1 && isempty(bc_dirichlet)
    #     # Add the average = 0 condition for Neumann b.c. (derivatives are 
    #     # assumed to be zero!). Note that this only sums the coefficients.
    #     A = vcat(A, ones((1,size(A)[2])))
    #     A = hcat(A, vcat(ones(size(A)[1]-1), 0.0))
    #     b = vcat(b, 0.0)
    # end

    if test
        if verbose
            println("Running tests ...")
        end
        #@test isapprox(A, A', rtol=1e-12)  # Full system matrices need not be symmetric due to the boundary conditions.
        @test isempty(nullspace(Matrix(A)))  # Only works on dense matrices!
        @test LinearAlgebra.cond(Matrix(A)) < 1e10
    end

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    sol = A \ b

    return sol
end

function write_form_sol_to_file(form_sols, var_names, geom, p, k, case, n, verbose)
    
    # This is for the plotting.
    for (form_sol, var_name) in zip(form_sols, var_names)
        if verbose
            println("Writing form '$var_name' to file ...")
        end
        
        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-Forms-$n-D-p$p-k$k-elements$msave-case-"*case*"-var_$var_name.vtu"
        #output_filename_error = "Poisson-Forms-$n-D-p$p-k$k-m$msave-case-"*case*"-error.vtu"

        output_file = joinpath(output_data_folder, output_filename)
        #output_file_error = joinpath(output_data_folder, output_filename_error)

        if n == 1
            out_deg = maximum([1, p])
        else
            out_deg = maximum([1, maximum(p)])
        end
        Mantis.Plot.plot(form_sol; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end
end







using InteractiveUtils
# Here we setup and specify the inputs to the FE problem.

# Compute base directories for data input and output
Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Poisson") # Create this folder first if you haven't done so yet.

# Choose whether to write the output to a file, run the tests, and/or 
# print progress statements. Make sure they are set as indicated when 
# committing and that the grid is not much larger than 10x10
write_to_output_file = false  # false
run_tests = true              # true
verbose = false               # false





# ########################################################################
# ## Test cases for the 1D Poisson problem.                             ##
# ########################################################################

if verbose
    println("Creating 1D Geometry and spaces ...")
end

# Dimension
n_1d = 1
# Number of elements.
m_1d = 5
# polynomial degree and inter-element continuity.
p_1d = 3
k_1d = 2
# Domain.
const Lleft_1d = 0.0  # exact solutions are for xl = 0.0
const Lright_1d = 2.0  # exact solution for xr > 0.0


# Create function spaces (b-splines here).
trial_space_1d = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d, k_1d)
test_space_1d = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d, k_1d)

trial_space_1d_pm1 = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d-1, k_1d-1)
test_space_1d_pm1 = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d-1, k_1d-1)

# Set Dirichlet boundary conditions.
const bc_left_sine_1d = 0.0
const bc_right_sine_1d = 1.0
bc_sine_1d = Dict{Int, Float64}(1 => bc_left_sine_1d, Mantis.FunctionSpaces.get_num_basis(trial_space_1d) => bc_right_sine_1d)
bc_cos_1d = Dict{Int, Float64}(1 => 0.0, Mantis.FunctionSpaces.get_num_basis(trial_space_1d) => 0.0)
bc_const_1d = Dict{Int, Float64}(1 => 0.0, Mantis.FunctionSpaces.get_num_basis(trial_space_1d) => 0.0)
bc_const_1d_empty = Dict{Int, Float64}()

# Create the geometry.
brk_1d = collect(LinRange(Lleft_1d, Lright_1d, m_1d+1))
geom_1d = Mantis.Geometry.CartesianGeometry((brk_1d,))

# Setup the quadrature rule.
q_rule_1d = Mantis.Quadrature.tensor_product_rule((p_1d + 1,), Mantis.Quadrature.gauss_legendre)

# Create form spaces (both test and trial)
zero_form_space_trial_1d = Mantis.Forms.FormSpace(0, geom_1d, (trial_space_1d,), "φ")
zero_form_space_test_1d = Mantis.Forms.FormSpace(0, geom_1d, (test_space_1d,), "ϕ")
one_form_space_trial_1d = Mantis.Forms.FormSpace(1, geom_1d, (trial_space_1d_pm1,), "φ")
one_form_space_test_1d = Mantis.Forms.FormSpace(1, geom_1d, (test_space_1d_pm1,), "ϕ")

# Forcing forms
# Constant forcing
function forcing_const_1d(x::Matrix{Float64})
    return [ones(size(x,1))]
end
f⁰_const_1d = Mantis.Forms.AnalyticalFormField(0, forcing_const_1d, geom_1d, "f")

# Sine forcing
function forcing_sine_1d(x::Matrix{Float64})
    return [@. pi^2 * sinpi.(x[:,1])]
end
f⁰_sine_1d = Mantis.Forms.AnalyticalFormField(0, forcing_sine_1d, geom_1d, "f")

# Cosine forcing for the mixed formulation.
function forcing_cos_1d(x::Matrix{Float64})
    return [@. pi^2 * cospi.(x[:,1]-0.5)]
end
f¹_cos_1d = Mantis.Forms.AnalyticalFormField(1, forcing_cos_1d, geom_1d, "f")


# Exact solutions as forms.
# Constant forcing
function exact_sol_const_1d(x::Matrix{Float64})
    return [@. -0.5 * x[:,1]^2 + 0.5*Lright_1d*x[:,1]]
end
sol⁰_const_1d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_const_1d, geom_1d, "sol")

# Sine forcing
function exact_sol_sine_1d(x::Matrix{Float64})
    return [@. sinpi(x[:,1]) + ((bc_right_sine_1d - bc_left_sine_1d - sinpi(Lright_1d)) / Lright_1d)*x[:,1] + bc_left_sine_1d]
end
sol⁰_sine_1d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_1d, geom_1d, "sol")

# Cosine forcing
function exact_sol_cos_1d_zero_form(x::Matrix{Float64})
    return [@. -pi * sinpi((x[:,1] - 0.5)) - cospi((Lright_1d - 0.5)) / Lright_1d]
end
function exact_sol_cos_1d_one_form(x::Matrix{Float64})
    return [@. cospi((x[:,1] - 0.5)) - x[:,1] * cospi((Lright_1d - 0.5)) / Lright_1d]
end
sol⁰_cos_1d_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_cos_1d_zero_form, geom_1d, "sol")
sol¹_cos_1d_exact_sol = Mantis.Forms.AnalyticalFormField(1, exact_sol_cos_1d_one_form, geom_1d, "sol")




# ########################################################################
# ## Test cases for the 2D Poisson problem.                             ##
# ########################################################################

if verbose
    println("Creating 2D Geometry and spaces ...")
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
bc_dirichlet_2d = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in trial_space_2d.dof_partition[j])
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
bc_dirichlet_3d = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] for i in trial_space_3d.dof_partition[j])

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











########################################################################
## Running all testcases.                                             ##
########################################################################
if verbose
    println()
end

cases = ["const1d-Dirichlet", "sine1d-Dirichlet", "cos1d-Dirichlet-mixed", "const2d-Dirichlet", "const2d-Dirichlet-crazy", "sine2d-Dirichlet", "sine2d-Dirichlet-crazy", "sine2d-Dirichlet-mixed", "sine2d-Dirichlet-mixed-crazy", "sine3d-Dirichlet"]
for case in cases

    if case == "const1d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_const_1d, zero_form_space_trial_1d, zero_form_space_test_1d, q_rule_1d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_const_1d, case, run_tests, verbose)

        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_1d, "α")
        α⁰.coefficients .= sol
        if run_tests
            @test isapprox(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_const_1d_exact_sol, q_rule_1d, "L2"), 0.0, atol=1e-14)
            @test isapprox(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_const_1d_exact_sol, q_rule_1d, "Linf"), 0.0, atol=1e-14)
        end
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_const_1d_exact_sol, q_rule_1d, "L2"))
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, sol⁰_const_1d_exact_sol], ["zero_form", "exact_zero_form"], geom_1d, p_1d, k_1d, case, n_1d, verbose)
        end
    
    elseif case == "sine1d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_1d, zero_form_space_trial_1d, zero_form_space_test_1d, q_rule_1d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_sine_1d, case, run_tests, verbose)

        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_1d, "α")
        α⁰.coefficients .= sol
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_sine_1d_exact_sol, q_rule_1d, "L2"))
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, sol⁰_sine_1d_exact_sol], ["zero_form", "exact_zero_form"], geom_1d, p_1d, k_1d, case, n_1d, verbose)
        end
    
    elseif case == "cos1d-Dirichlet-mixed"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(f¹_cos_1d, zero_form_space_trial_1d, one_form_space_trial_1d, zero_form_space_test_1d, one_form_space_test_1d, q_rule_1d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_mixed, bc_const_1d_empty, case, run_tests, verbose)

        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_1d, "α")
        ξ¹ = Mantis.Forms.FormField(one_form_space_trial_1d, "ξ")
        α⁰.coefficients .= sol[1:Mantis.Forms.get_num_basis(zero_form_space_trial_1d)]
        ξ¹.coefficients .= sol[Mantis.Forms.get_num_basis(zero_form_space_trial_1d)+1:end]
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_cos_1d_exact_sol, q_rule_1d, "L2"))
            print("Total L2 error 1-form: ")
            println(Mantis.Assemblers.compute_error_total(ξ¹, sol¹_cos_1d_exact_sol, q_rule_1d, "L2"))
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, ξ¹, sol⁰_cos_1d_exact_sol, sol¹_cos_1d_exact_sol], ["zero_form", "one_form", "exact_zero_form", "exact_one_form"], geom_1d, p_1d, k_1d, case, n_1d, verbose)
        end
    
    elseif case == "const2d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_const_2d, zero_form_space_trial_2d_cart, zero_form_space_test_2d_cart, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, case, run_tests, verbose)
        if write_to_output_file
            α⁰ = Mantis.Forms.FormField(zero_form_space_trial_2d_cart, "α")
            α⁰.coefficients .= sol

            write_form_sol_to_file([α⁰], ["zero_form"], geom_cartesian, p_2d, k_2d, case, n_2d, verbose)
        end
        
    elseif case == "const2d-Dirichlet-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_crazy_const_2d, zero_form_space_trial_2d_crazy, zero_form_space_test_2d_crazy, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, case*"_crazy_c$crazy_c", run_tests, verbose)
        if write_to_output_file
            α⁰ = Mantis.Forms.FormField(zero_form_space_trial_2d_crazy, "α")
            α⁰.coefficients .= sol

            write_form_sol_to_file([α⁰], ["zero_form"], geom_crazy, p_2d, k_2d, case*"_crazy_c$crazy_c", n_2d, verbose)
        end
        
    elseif case == "sine2d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_sine_2d, zero_form_space_trial_2d_cart, zero_form_space_test_2d_cart, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, case, run_tests, verbose)

        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_2d_cart, "α")
        α⁰.coefficients .= sol
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_cart_sine_2d_exact_sol, q_rule_2d, "L2"))
        end
        if write_to_output_file
           write_form_sol_to_file([α⁰, sol⁰_cart_sine_2d_exact_sol, α⁰ - sol⁰_cart_sine_2d_exact_sol], ["zero_form", "exact_zero_form", "error_zero_form"], geom_cartesian, p_2d, k_2d, case, n_2d, verbose)
        end
        
    elseif case == "sine2d-Dirichlet-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_crazy_sine_2d, zero_form_space_trial_2d_crazy, zero_form_space_test_2d_crazy, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, case*"_crazy_c$crazy_c", run_tests, verbose)
        
        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_2d_crazy, "α")
        α⁰.coefficients .= sol
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_crazy_sine_2d_exact_sol, q_rule_2d, "L2"))
        end
        if write_to_output_file
           write_form_sol_to_file([α⁰, sol⁰_crazy_sine_2d_exact_sol, α⁰ - sol⁰_crazy_sine_2d_exact_sol], ["zero_form", "exact_zero_form", "error_zero_form"], geom_crazy, p_2d, k_2d, case*"_crazy_c$crazy_c", n_2d, verbose)
        end
        
    elseif case == "sine2d-Dirichlet-mixed"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(f²_cart_sine_2d, one_form_space_trial_2d_cart, two_form_space_trial_2d_cart, one_form_space_test_2d_cart, two_form_space_test_2d_cart, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty, case, run_tests, verbose)

        # Create solution(s) as forms.
        ξ¹ = Mantis.Forms.FormField(one_form_space_trial_2d_cart, "ξ")
        β² = Mantis.Forms.FormField(two_form_space_trial_2d_cart, "β")
        ξ¹.coefficients .= sol[1:Mantis.Forms.get_num_basis(one_form_space_trial_2d_cart)]
        β².coefficients .= sol[Mantis.Forms.get_num_basis(one_form_space_trial_2d_cart)+1:end]
        if verbose
            print("Total L2 error 1-form: ")
            println(Mantis.Assemblers.compute_error_total(ξ¹, sol¹_cart_sine_2d_exact_sol, q_rule_2d, "L2"))
            print("Total L2 error 2-form: ")
            println(Mantis.Assemblers.compute_error_total(β², sol²_cart_sine_2d_exact_sol, q_rule_2d, "L2"))
        end
        if write_to_output_file
            write_form_sol_to_file([β², ξ¹, sol²_cart_sine_2d_exact_sol, sol¹_cart_sine_2d_exact_sol, β² - sol²_cart_sine_2d_exact_sol, ξ¹ - sol¹_cart_sine_2d_exact_sol], ["two_form", "one_form", "exact_two_form", "exact_one_form", "error_two_form", "error_one_form"], geom_cartesian, p_2d, k_2d, case, n_2d, verbose)
        end
    
    elseif case == "sine2d-Dirichlet-mixed-crazy"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(f²_crazy_sine_2d, one_form_space_trial_2d_crazy, two_form_space_trial_2d_crazy, one_form_space_test_2d_crazy, two_form_space_test_2d_crazy, q_rule_2d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty, case*"_crazy_c$crazy_c", run_tests, verbose)
        
        # Create solution(s) as forms.
        ξ¹ = Mantis.Forms.FormField(one_form_space_trial_2d_crazy, "ξ")
        β² = Mantis.Forms.FormField(two_form_space_trial_2d_crazy, "β")
        ξ¹.coefficients .= sol[1:Mantis.Forms.get_num_basis(one_form_space_test_2d_crazy)]
        β².coefficients .= sol[Mantis.Forms.get_num_basis(one_form_space_test_2d_crazy)+1:end]
        if verbose
            print("Total L2 error 1-form: ")
            println(Mantis.Assemblers.compute_error_total(ξ¹, sol¹_crazy_sine_2d_exact_sol, q_rule_2d, "L2"))
            print("Total L2 error 2-form: ")
            println(Mantis.Assemblers.compute_error_total(β², sol²_crazy_sine_2d_exact_sol, q_rule_2d, "L2"))
        end
        if write_to_output_file
            write_form_sol_to_file([β², ξ¹, sol²_crazy_sine_2d_exact_sol, sol¹_crazy_sine_2d_exact_sol, β² - sol²_crazy_sine_2d_exact_sol, ξ¹ - sol¹_crazy_sine_2d_exact_sol], ["two_form", "one_form", "exact_two_form", "exact_one_form", "error_two_form", "error_one_form"], geom_crazy, p_2d, k_2d, case*"_crazy_c$crazy_c", n_2d, verbose)
        end

    elseif case == "sine3d-Dirichlet"
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_cart_sine_3d, zero_form_space_trial_3d_cart, zero_form_space_test_3d_cart, q_rule_3d)
        sol = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_3d, case, run_tests, verbose)

        # Create solution(s) as forms.
        α⁰ = Mantis.Forms.FormField(zero_form_space_trial_3d_cart, "α")
        α⁰.coefficients .= sol
        if verbose
            print("Total L2 error 0-form: ")
            println(Mantis.Assemblers.compute_error_total(α⁰, sol⁰_cart_sine_3d_exact_sol, q_rule_3d, "L2"))
        end
        if write_to_output_file
           write_form_sol_to_file([α⁰, sol⁰_cart_sine_3d_exact_sol, α⁰ - sol⁰_cart_sine_3d_exact_sol], ["zero_form", "exact_zero_form", "error_zero_form"], geom_3d_cartesian, p_3d, k_3d, case, n_3d, verbose)
        end

    else
        if verbose
            println("Warning: case '"*case*"' unknown. Skipping.") 
        end
    end
    if verbose
        println()  # Extra blank line to separate the different runs.
    end
end


