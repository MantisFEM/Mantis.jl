module PlotTests

import Mantis

import Mmap
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Geometry")
output_data_folder = joinpath(data_folder, "output", "Geometry")

# Test CartesianGeometry ------------------------------------------------------
for nx = 1:3
    for ny = 1:3
        breakpoints = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)))
        geom = Mantis.Geometry.CartesianGeometry(breakpoints)
        
        # Generate the plot
        output_filename = @sprintf "cartesian_test_nx_%d_ny_%d.vtu" nx ny
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

        # Test geometry 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------

# Test MappedCartesianGeometry ------------------------------------------------
for nx = 1:3
    for ny = 1:3
        breakpoints = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)))
        geom = Mantis.Geometry.CartesianGeometry(breakpoints)
        function mapping(x::Vector{Float64})
            return [(x[1] + 0.2)*cos(x[2]), (x[1] + 0.2)*sin(x[2])]
        end
        function dmapping(x::Vector{Float64})
            return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
        end
        dimension = (2, 2)
        curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
        mapped_geometry = Mantis.Geometry.MappedGeometry(geom, curved_mapping)


        # Generate the plot
        output_filename = @sprintf "mapped_cartesian_test_nx_%d_ny_%d.vtu" nx ny
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(mapped_geometry; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 3)

        # Test geometry 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------

# Test FEMGeometry (Annulus) --------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace((GB,b1), Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_annulus_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4)

# Test geometry 
input_file = joinpath(input_data_folder, output_filename)
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
# -----------------------------------------------------------------------------

# Test FEMGeometry - LagrangexBernstein (Square w/ hole) ----------------------
deg = 1
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.LobattoLegendre(deg))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [0,0,0,0])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace((GB,b1), Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_lagrange_square_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

# Test geometry 
input_file = joinpath(input_data_folder, output_filename)
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
# -----------------------------------------------------------------------------

# Test FEMGeometry (Spiral) ---------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, -1])
# control points for geometry
geom_coeffs =   [0.0 -1.0 0.0
1.0  -1.0 0.25
1.0   1.0 0.5
-1.0   1.0 0.75
-1.0  -1.0 1.0
0.0  -1.0 1.25]
spiral_geom = Mantis.Geometry.FEMGeometry(GB, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_spiral_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(spiral_geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4)

# Test geometry 
input_file = joinpath(input_data_folder, output_filename)
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
# -----------------------------------------------------------------------------

# Test FEMGeometry (wavy surface) ---------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace((GB,b1), Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
               geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]]
wavy_surface_geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_wavy_surface_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(wavy_surface_geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4)

# Test geometry 
input_file = joinpath(input_data_folder, output_filename)
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
# -----------------------------------------------------------------------------

# Test Tensor Product Geometry ------------------------------------------------



# -----------------------------------------------------------------------------

end