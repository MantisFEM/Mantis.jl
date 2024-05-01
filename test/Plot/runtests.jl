module PlotTests

import Mantis

import Mmap
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Plot")
output_data_folder = joinpath(data_folder, "output", "Plot")

# Test Plotting of 2D Geometry ------------------------------------------------
# Generate the geometry
nx = 3
ny = 2
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

# Generate the plots
degrees_range = 1:3:10
n_subcells_range = 1:3:10

for n_subcells in n_subcells_range
    for degree in degrees_range 
        output_filename = @sprintf "mapped_cartesian_test_nx_%02d_ny_%02d__n_sub_%02d_degree_%02d.vtu" nx ny n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)

        # Plot
        Mantis.Plot.plot(mapped_geometry; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        # @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------

# Test 1D Geometry ------------------------------------------------------------
# Generate the Geometry
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
geom = Mantis.Geometry.FEMGeometry(GB, geom_coeffs)

# Generate the plots
degrees_range = 1:3:10
n_subcells_range = 1:3:10

for n_subcells in n_subcells_range
    for degree in degrees_range 
        output_filename = @sprintf "spiral_fem_geometry__n_sub_%02d_degree_%02d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        
        # Plot
        Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        # @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------
end