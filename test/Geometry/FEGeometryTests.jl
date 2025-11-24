module FEGeometryTests

import Mantis

import ReadVTK
import LinearAlgebra

using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Test FEMGeometry (Polar) --------------------------------------------------
# deg = 2
# Wt = pi / 2
# b_θ = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
# b_r = Mantis.FunctionSpaces.Bernstein(deg)

# (P_sol, E_sol), (P_geom, E_geom, geom_coeffs_polar), _ =
#     Mantis.FunctionSpaces.create_scalar_polar_splines_and_geometry(
#         (4, 1), (b_θ, b_r), (1, -1), 1.0
# )
# geom = Mantis.Geometry.FEMGeometry(P_geom, geom_coeffs_polar)
# field_coeffs = Matrix{Float64}(
#     LinearAlgebra.I,
#     Mantis.FunctionSpaces.get_num_basis(P_sol),
#     Mantis.FunctionSpaces.get_num_basis(P_sol),
# )

# # Generate the plot
# file_name = "fem_geometry_polar_test.vtu"
# output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)

# Test FEGeometry (Annulus) --------------------------------------------------
deg = 2
Wt = pi / 2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Geometry.CartesianGeometry(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0
    geom_coeffs_0 .* r1
]
geom = Mantis.Geometry.FEGeometry(TP, geom_coeffs)
# Generate the plot
file_name = "fem_geometry_annulus_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    geom;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the data from the output file.
output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry - LagrangexBernstein (Square w/ hole) ----------------------
deg = 1
b = Mantis.FunctionSpaces.LobattoLegendre(deg)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Geometry.CartesianGeometry(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 0, 0, 0, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [0])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0
    geom_coeffs_0 .* r1
]
geom = Mantis.Geometry.FEGeometry(TP, geom_coeffs)
# Generate the plot
file_name = "fem_geometry_lagrange_square_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    geom;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=1,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the data from the output file.
output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (Spiral) ---------------------------------------------------
deg = 2
Wt = pi / 2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Geometry.CartesianGeometry(breakpoints)
GB = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])

# control points for geometry
geom_coeffs = [
    +0.0 -1.0 0.0
    +1.0 -1.0 0.25
    +1.0 +1.0 0.5
    -1.0 +1.0 0.75
    -1.0 -1.0 1.0
    +0.0 -1.0 1.25
]
spiral_geom = Mantis.Geometry.FEGeometry(GB, geom_coeffs)

# Generate the plot
file_name = "fem_geometry_spiral_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    spiral_geom;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the data from the output file.
output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (wavy surface) ---------------------------------------------
deg = 2
Wt = pi / 2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Geometry.CartesianGeometry(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
    geom_coeffs_0.*r1 +[+1.0, -1.0, +1.0, -1.0]
]
wavy_surface_geom = Mantis.Geometry.FEGeometry(TP, geom_coeffs)

# Generate the plot
file_name = "fem_geometry_wavy_surface_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    wavy_surface_geom;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the data from the output file.
output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (NURBS quarter annulus) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
B = Mantis.FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((B, b1))
# control points for geometry
geom_coeffs_0 = [
    0.0 1.0
    1.0 1.0
    1.0 0.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0.*r0 zeros(3)
    geom_coeffs_0.*r1 zeros(3)
]
quarter_annulus_geom = Mantis.Geometry.FEGeometry(TP, geom_coeffs)

# Generate the plot
file_name = "fem_geometry_nurbs_quarter_annulus_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    quarter_annulus_geom;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
#reference_points, reference_cells = get_point_cell_data(
#    reference_directory_tree, file_name
#)
# Read the data from the output file.
#output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
#@test all(isapprox.(reference_points, output_points; atol=atol))
#@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (NURBS annulus) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
B = (br, br, br, br)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0.*r0 zeros(4)
    geom_coeffs_0.*r1 zeros(4)
]
nurbs_annulus = Mantis.Geometry.FEGeometry(TP, geom_coeffs)

# Generate the plot
file_name = "fem_geometry_nurbs_annulus_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nurbs_annulus;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
#reference_points, reference_cells = get_point_cell_data(
#    reference_directory_tree, file_name
#)
# Read the data from the output file.
#output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
#@test all(isapprox.(reference_points, output_points; atol=atol))
#@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (NURBS wavy surface) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
B = (br, br, br, br)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
    geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]
]
nurbs_wavy_surface = Mantis.Geometry.FEGeometry(TP, geom_coeffs)

# Generate the plot
file_name = "fem_geometry_nurbs_wavy_surface_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nurbs_wavy_surface;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Test geometry
# Read the data from the reference file.
#reference_points, reference_cells = get_point_cell_data(
#    reference_directory_tree, file_name
#)
# Read the data from the output file.
#output_points, output_cells = get_point_cell_data(output_file_path)

# Check if data is point-wise identical.
#@test all(isapprox.(reference_points, output_points; atol=atol))
#@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test FEGeometry (NURBS vs GTB basis) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
Bsp = Mantis.FunctionSpaces.GTBSplineSpace((b, b, b, b), [1, 1, 1, 1])
Nurbs = Mantis.FunctionSpaces.GTBSplineSpace((br, br, br, br), [1, 1, 1, 1])

Wt = pi / 2
gt = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
B = Mantis.FunctionSpaces.BSplineSpace(
    Mantis.Geometry.CartesianGeometry([0.0, 1.0, 2.0, 3.0, 4.0]), gt, [-1, 1, 1, 1, -1]
)
GTB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])

b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])

TP_bsp = Mantis.FunctionSpaces.TensorProductSpace((Bsp, b1))
TP_nurbs = Mantis.FunctionSpaces.TensorProductSpace((Nurbs, b1))
TP_gtb = Mantis.FunctionSpaces.TensorProductSpace((GTB, b1))

# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0.*r0 zeros(4)
    geom_coeffs_0.*r1 zeros(4)
]
nurbs_annulus = Mantis.Geometry.FEGeometry(TP_nurbs, geom_coeffs)
gtb_annulus = Mantis.Geometry.FEGeometry(TP_gtb, geom_coeffs)

# Generate the plot - NURBS + BSP
file_name = "fem_geometry_nurbs_bsp_basis_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    nurbs_annulus;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Generate the plot - GTB
file_name = "fem_geometry_nurbs_gtb_basis_test.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    gtb_annulus;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

end
