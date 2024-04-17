import Mantis, LinearAlgebra, Plots

abstract type AbstractFunctionSpace end

####### GEOMETRY DEFINITION

abstract type AbstractGeometry end

####### FEM-BASED GEOMETRY AND EVALUATORS

struct FEMGeometry <: AbstractGeometry
    geometry_coeffs::Vector{Float64}
    fem_space::AbstractFunctionSpace
end

function evaluate_map(geometry::FEMGeometry, element_id::Int, xi::Vector{Float64}, ::Vector{Float64})
    # evaluate fem space
    fem_basis, fem_basis_indices, _ = evaluate_val(geometry.fem_space, element_id, xi)
    # combine with coefficients and return
    return fem_basis * geometry.geometry_coeffs[fem_basis_indices]
end

function evaluate_dmap(geometry::FEMGeometry, element_id::Int, xi::Vector{Float64}, ::Vector{Float64})
    # evaluate fem space
    dfem_basis, fem_basis_indices, _ = evaluate_der(geometry.fem_space, element_id, xi)
    # combine with coefficients and return
    return dfem_basis * geometry.geometry_coeffs[fem_basis_indices]
end

####### ANALYTICAL GEOMETRY AND EVALUATORS

struct AnalGeometry <: AbstractGeometry
    map::Function
    dmap::Function
end

function evaluate_map(geometry::AnalGeometry, ::Int, ::Vector{Float64}, x::Vector{Float64})
    return geometry.map(x)
end

function evaluate_dmap(geometry::AnalGeometry, ::Int, ::Vector{Float64}, x::Vector{Float64})
    return geometry.dmap(x)
end

####### NEW FEM SPACE AND EVALUATORS

struct NewFEMSpace <: AbstractFunctionSpace
    space::AbstractFunctionSpace
    geometry::AbstractGeometry
end

function evaluate_val(fem_space::NewFEMSpace, element_id::Int, ξ::Vector{Float64})
    # first evaluate the space
    fem_basis, basis_indices, x = evaluate_val(fem_space.space, element_id, ξ)
    # then evaluate the geometry at the right points
    y = evaluate_map(fem_space.geometry, element_id, ξ, x)
    return fem_basis, basis_indices, y
end

function evaluate_der(fem_space::NewFEMSpace, element_id::Int, ξ::Vector{Float64})
    # first evaluate the space
    dfem_basis, basis_indices, x = evaluate_der(fem_space.space, element_id, ξ)
    # then evaluate the geometry and derivatives at the right points
    y = evaluate_map(fem_space.geometry, element_id, ξ, x)
    dydx = evaluate_dmap(fem_space.geometry, element_id, ξ, x)
    # transform the values of the computed basis appropriately
    for i = 1:length(ξ)
        dfem_basis[i,:] ./= dydx[i]
    end

    return dfem_basis, basis_indices, y
end

function get_num_elements(fem_space::NewFEMSpace)
    return get_num_elements(fem_space.space)
end

function get_dim(fem_space::NewFEMSpace)
    return get_dim(fem_space.space)
end

####### A SPECIFIC INSTANCE OF A NEW FEM SPACE: ELEMENT SPACE

struct NewElementSpace <: AbstractFunctionSpace
    polynomials::Mantis.FunctionSpaces.Bernstein
    geometry::AbstractGeometry

    function NewElementSpace(p)
        polynomials = Mantis.FunctionSpaces.Bernstein(p)
        map(xi) = xi
        dmap(xi) = 1.0*ones(size(xi))
        new(polynomials, AnalGeometry(map, dmap))
    end
end

function evaluate_val(fem_space::NewElementSpace, ::Int, ξ::Vector{Float64})
    return Mantis.FunctionSpaces.evaluate(fem_space.polynomials, ξ, 0)[:,:,1], 1:fem_space.polynomials.p+1, ξ
end

function evaluate_der(fem_space::NewElementSpace, ::Int, ξ::Vector{Float64})
    return Mantis.FunctionSpaces.evaluate(fem_space.polynomials, ξ, 1)[:,:,2], 1:fem_space.polynomials.p+1, ξ
end

function get_num_elements(::NewElementSpace)
    return 1
end

function get_dim(fem_space::NewElementSpace)
    return fem_space.polynomials.p+1
end

####### COMPOSITE FEM SPACE

struct NewUnstructuredSpace <: AbstractFunctionSpace
    fem_spaces::NTuple{m, AbstractFunctionSpace} where {m}
    extraction_op::Mantis.FunctionSpaces.ExtractionOperator
    us_config::Dict
    
    function NewUnstructuredSpace(fem_spaces::NTuple{m,AbstractFunctionSpace}, extraction_op::Mantis.FunctionSpaces.ExtractionOperator, us_config::Dict) where {m}
        new(fem_spaces, extraction_op, us_config)
    end

    function NewUnstructuredSpace(fem_spaces::NTuple{m,AbstractFunctionSpace}, extraction_op::Mantis.FunctionSpaces.ExtractionOperator)where {m}
        # build 1D topology
        patch_neighbours = [-1 (1:m-1)...
                            (2:m)... -1]
        # number of elements per patch
        patch_nels = [0; cumsum([get_num_elements(fem_spaces[i]) for i = 1:m])]

        # assemble patch config in a dictionary
        us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        new(fem_spaces, extraction_op, us_config)
    end
end

function get_dim(us_space::NewUnstructuredSpace)
    return us_space.extraction_op.space_dim
end

function get_num_elements(us_space::NewUnstructuredSpace)
    return Mantis.FunctionSpaces.get_num_elements(us_space.extraction_op)
end

function get_extraction(us_space::NewUnstructuredSpace, element_id::Int)
    return Mantis.FunctionSpaces.get_extraction(us_space.extraction_op, element_id)
end

function get_space_id(us_space::NewUnstructuredSpace, element_id::Int)
    return findlast(us_space.us_config["patch_nels"] .< element_id)
end

function get_local_basis(us_space::NewUnstructuredSpace, element_id::Int, xi::Vector{Float64})
    space_id = get_space_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]
    return evaluate_val(us_space.fem_spaces[space_id], space_element_id, xi)
end

function get_local_dbasis(us_space::NewUnstructuredSpace, element_id::Int, xi::Vector{Float64})
    space_id = get_space_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]
    return evaluate_der(us_space.fem_spaces[space_id], space_element_id, xi)
end

function evaluate_val(us_space::NewUnstructuredSpace, element_id::Int, xi::Vector{Float64})
    extraction_coefficients, basis_indices = get_extraction(us_space, element_id)
    us_basis, _, x = get_local_basis(us_space, element_id, xi)
    us_basis .= @views us_basis * extraction_coefficients

    return us_basis, basis_indices, x
end

function evaluate_der(us_space::NewUnstructuredSpace, element_id::Int, xi::Vector{Float64})
    extraction_coefficients, basis_indices = get_extraction(us_space, element_id)
    us_dbasis, _, x = get_local_dbasis(us_space, element_id, xi)
    us_dbasis .= @views us_dbasis * extraction_coefficients

    return us_dbasis, basis_indices, x
end

####### PLOTTING
function plot_basis_vals(fem_space::AbstractFunctionSpace, xi::Vector{Float64})
    Vals = Vector{Array{Float64}}(undef, get_dim(fem_space))
    for i = 1:get_dim(fem_space)
        Vals[i] = [[] []]
    end
    for el = 1:get_num_elements(fem_space)
        vals, inds, x = evaluate_val(fem_space, el, xi)
        for i = 1:length(inds)
            Vals[inds[i]] = [Vals[inds[i]]; [x vals[:,i]]]
        end
    end
    p = Plots.plot()
    for i = 1:get_dim(fem_space)
        Plots.plot!(p, Vals[i][:,1], Vals[i][:,2])
    end
    Plots.display(p)
end

####### EXAMPLES OF APPLICATIONS

# polynomial degree
deg = 3
# plotting points
nplt = 21
xi_plot = collect(LinRange(0.0,1.0,nplt))

#### LEVEL 0
# starting space with trivial geometry
B0 = NewElementSpace(deg)

#### LEVEL 1
# Mapped discontinuous spaces with non-trivial analytical geometry
map11(xi) = 0.25 * xi.^2 + 0.25 * xi; dmap11(xi) = 0.5 * xi .+ 0.25
map12(xi) = 0.5 * xi .+ 0.5; dmap12(xi) = 0.5 * ones(size(xi))
B11 = NewFEMSpace(B0, AnalGeometry(map11,dmap11))
B12 = NewFEMSpace(B0, AnalGeometry(map12,dmap12))

plot_basis_vals(B11, xi_plot)
plot_basis_vals(B12, xi_plot)

# # (B0, geom1, geom2) -> (B0, geom1), (B0, geom2) -> multi-patch space

# evaluations at end points
B11_val, inds11, x11 = evaluate_val(B11, 1, [1.0])
B12_val, inds12, x12 = evaluate_val(B12, 1, [0.0])
B11_der, _, x11 = evaluate_der(B11, 1, [1.0])
B12_der, _, x12 = evaluate_der(B12, 1, [0.0])
n11 = get_dim(B11)
n12 = get_dim(B12)
# extraction operator computation
g_extraction_coefficients = zeros(n11+n12, n11+n12-2)
g_extraction_coefficients[1:n11-1, 1:n11-1] = Matrix(LinearAlgebra.I, n11-1,n11-1)
g_extraction_coefficients[end-n12+2:end, end-n12+2:end] = Matrix(LinearAlgebra.I, n12-1,n12-1)
constraint_vec = [-B11_der[end-1], -B11_der[end]+B12_der[1], B12_der[2]]
alpha = -constraint_vec[1]./constraint_vec[2]
beta = -constraint_vec[3]./constraint_vec[2]
g_extraction_coefficients[n11:n11+1,n11-1] .= alpha
g_extraction_coefficients[n11:n11+1,n11] .= beta
extraction_coeffs = Vector{Array{Float64}}(undef,2)
extraction_coeffs[1] = g_extraction_coefficients[1:deg+1,1:deg+1]
extraction_coeffs[2] = g_extraction_coefficients[deg+2:end,deg:end]
basis_indices = Vector{Vector{Int}}(undef,2)
basis_indices[1] = 1:n11
basis_indices[2] = n11-1:n11+n12-2
num_elements = 2
space_dim = n11+n12-2

# combined space
B1 = NewUnstructuredSpace((B11, B12),Mantis.FunctionSpaces.ExtractionOperator(extraction_coeffs,basis_indices,num_elements,space_dim))

plot_basis_vals(B1, xi_plot)

#### LEVEL 2
# Mapped discontinuous spaces with non-trivial FEM geometry
geom21 = FEMGeometry(LinRange(0.0, 2.0, get_dim(B1)), B1)
geom22 = FEMGeometry(LinRange(0.0, 2.0, get_dim(B1)).^2 .+ 2, B1)
B21 = NewFEMSpace(B1, geom21)
B22 = NewFEMSpace(B1, geom22)

plot_basis_vals(B21, xi_plot)
plot_basis_vals(B22, xi_plot)

readline()

# # evaluations at end points
# B21_val, inds21, x21 = evaluate_val(B21, 1, [1.0])
# B22_val, inds22, x22 = evaluate_val(B22, 1, [0.0])
# B21_der, _, x21 = evaluate_der(B21, 1, [1.0])
# B22_der, _, x22 = evaluate_der(B22, 1, [0.0])
# n21 = get_dim(B21)
# n22 = get_dim(B22)
# # extraction operator computation
# g_extraction_coefficients = zeros(n21+n22, n21+n22-2)
# g_extraction_coefficients[1:n21-1, 1:n21-1] = Matrix(LinearAlgebra.I, n21-1,n21-1)
# g_extraction_coefficients[end-n22+2:end, end-n22+2:end] = Matrix(LinearAlgebra.I, n22-1,n22-1)
# constraint_vec = [-B21_der[end-1], -B21_der[end]+B22_der[1], B22_der[2]]
# alpha = -constraint_vec[1]./constraint_vec[2]
# beta = -constraint_vec[3]./constraint_vec[2]
# g_extraction_coefficients[n21:n21+1,n21-1] .= alpha
# g_extraction_coefficients[n21:n21+1,n21] .= beta
# g_extraction_coefficients_1 = g_extraction_coefficients[1:n21,:]
# g_extraction_coefficients_2 = g_extraction_coefficients[n21+1:end,:]
# extraction_coeffs = Vector{Array{Float64}}(undef,4)
# extraction_coeffs[1] = g_extraction_coefficients[1:deg+1,1:deg+1]
# extraction_coeffs[2] = g_extraction_coefficients[deg+2:end,deg:2*deg]

# basis_indices = Vector{Vector{Int}}(undef,4)
# # basis_indices[1] = 1:n21
# # basis_indices[2] = n21-1:n21+n22-2

# num_elements = 4
# space_dim = n21+n22-2

# # # combined space
# # B2 = NewUnstructuredSpace((B21, B22),Mantis.FunctionSpaces.ExtractionOperator(extraction_coeffs,basis_indices,num_elements,space_dim))