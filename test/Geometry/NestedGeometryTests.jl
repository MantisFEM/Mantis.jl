module NestedGeometryTests

import Mantis

import ReadVTK
using Printf
using Test
using LinearAlgebra

# Refer to the following file for method and variable definitions
include("GeometryTestsHelpers.jl")

# Create tensor-product base spaces
ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

# cartesian geometry of base space
cart_geometry = Mantis.Geometry.CartesianGeometry((
    breakpoints1, breakpoints2
))

# Define the mapping ϕ of the geometry and its derivative.
# ϕ(x,y) = [(x + 0.2)*cos(y), (x + 0.2)*sin(y)\
function mapping(x::AbstractVector)
    return [(x[1] + 0.2) * cos(x[2]), (x[1] + 0.2) * sin(x[2])]
end
function dmapping(x::AbstractVector)
    return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
end
function mapping(x::AbstractArray)
    return [(x[:, 1] .+ 0.2) .* cos.(x[:, 2]), (x[:, 1] .+ 0.2) .* sin.(x[:, 2])]
end
function dmapping(x::AbstractArray)
    return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
end

# build mapped geometry
dimension = (2, 2)
curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
mapped_geometry = Mantis.Geometry.MappedGeometry(cart_geometry, curved_mapping)

# create hierarchical B-spline space
deg1 = 2
deg2 = 2

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.build_two_scale_operator(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.build_two_scale_operator(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace((CB1, CB2))
FTP = Mantis.FunctionSpaces.TensorProductSpace((FB1, FB2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(CTP, FTP, (TS1,TS2))

coarse_elements_to_refine = [3,4,5,8,9,10]
refined_elements = vcat(Mantis.FunctionSpaces.get_element_children.(Ref(CTS), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([collect(1:CTP_num_els),refined_elements])

hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], refined_domains)

### TEST 1: Full mappings

# create full nested geometry map
hmapping = Mantis.Geometry.create_hierarchical_mesh_nestedness_map(hier_space, Int[])
# compose with different geometries and output
nested_geometry_cart = Mantis.Geometry.NestedGeometry(cart_geometry, hmapping)
nested_geometry_map = Mantis.Geometry.NestedGeometry(mapped_geometry, hmapping)

file_name = "nested_geometry_cart.vtu"
output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nested_geometry_cart;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=2,
    ascii=false,
    compress=false
)

# Test geometry
# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(
    reference_directory_tree, file_name
)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))

file_name = "nested_geometry_map.vtu"
output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nested_geometry_map;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=2,
    ascii=false,
    compress=false
)

# Test geometry
# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(
    reference_directory_tree, file_name
)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))

### TEST 2: Trimmed mappings

# create trimmed nested geometry map
exclude_elements = [1, 2, 3, 4, 5, 6]
hmapping_trim = Mantis.Geometry.create_hierarchical_mesh_nestedness_map(hier_space, exclude_elements)
# compose with different geometries and output
nested_geometry_cart_trim = Mantis.Geometry.NestedGeometry(cart_geometry, hmapping_trim)
nested_geometry_map_trim = Mantis.Geometry.NestedGeometry(mapped_geometry, hmapping_trim)

file_name = "nested_geometry_cart_trim.vtu"
output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nested_geometry_cart_trim;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=2,
    ascii=false,
    compress=false
)

# Test geometry
# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(
    reference_directory_tree, file_name
)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))

file_name = "nested_geometry_map_trim.vtu"
output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nested_geometry_map_trim;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=2,
    ascii=false,
    compress=false
)

# Test geometry
# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(
    reference_directory_tree, file_name
)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))

end
