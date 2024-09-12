module PlotTests

import Mantis

import ReadVTK
# import Mmap
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Plot")
output_data_folder = joinpath(data_folder, "output", "Plot")

# Test Plotting of 3D Geometry (toroidal annulus) -------------------------------------------
deg = 2
Wt = pi/2
bθ = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
Bθ = ntuple( i -> bθ, 4)
GBθ = Mantis.FunctionSpaces.GTBSplineSpace(Bθ, [1, 1, 1, 1])
Br = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP_θr = Mantis.FunctionSpaces.TensorProductSpace(GBθ, Br)
TP_θrϕ = Mantis.FunctionSpaces.TensorProductSpace(TP_θr, GBθ)
# control points for geometry
geom_coeffs_θ =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs_θr = [geom_coeffs_θ.*r0
                  geom_coeffs_θ.*r1]
geom_coeffs_θr0 = [geom_coeffs_θr.+[3*r1 0] zeros(8)]

# rotate the 3D points around the y-axis
geom_coeffs_θrϕ = Vector{Matrix{Float64}}(undef,4)
geom_coeffs_θrϕ[1] = geom_coeffs_θr0
for i ∈ 1:3
    ϕ = i*π/2
    R = [cos(ϕ) 0 sin(ϕ); 0 1 0; -sin(ϕ) 0 cos(ϕ)]
    geom_coeffs_θrϕ[i+1] = geom_coeffs_θr0 * R'
end
geom_coeffs_θrϕ = vcat(geom_coeffs_θrϕ...)
geom = Mantis.Geometry.FEMGeometry(TP_θrϕ, geom_coeffs_θrϕ)
# Generate the plot
output_filename = "fem_geometry_toroidal_annulus_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test Plotting of 3D Geometry (hollow cylinder) -------------------------------------------
deg = 2
Wt = pi/2
bθ = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
Bθ = ntuple( i -> bθ, 4)
GBθ = Mantis.FunctionSpaces.GTBSplineSpace(Bθ, [1, 1, 1, 1])
Br = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP_θr = Mantis.FunctionSpaces.TensorProductSpace(GBθ, Br)
Bz = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP_θrz = Mantis.FunctionSpaces.TensorProductSpace(TP_θr, Bz)
# control points for geometry
geom_coeffs_θ =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
z0 = 0
z1 = 1
geom_coeffs_θr = [geom_coeffs_θ.*r0
                    geom_coeffs_θ.*r1]
geom_coeffs_θrz = [geom_coeffs_θr z0.*ones(8)
                    geom_coeffs_θr z1.*ones(8)]
geom = Mantis.Geometry.FEMGeometry(TP_θrz, geom_coeffs_θrz)
# Generate the plot
output_filename = "fem_geometry_hollow_cylinder_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test Plotting of 3D Geometry (Cartesian cuboid) -------------------------------------------
nx = 4
ny = 3
nz = 2
breakpoints = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)), collect(LinRange(0.0,4.0,nz+1)))
geom = Mantis.Geometry.CartesianGeometry(breakpoints)
# Generate the plot
output_filename = "cartesian_geometry_cuboid_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 2, degree = 4, ascii = false, compress = false)

# -----------------------------------------------------------------------------

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
        Mantis.Plot.plot(mapped_geometry; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree, ascii = false, compress = false)

        # Test plotting 
        # Read the cell data from the reference file
        reference_file = joinpath(input_data_folder, output_filename)
        vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
        reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
        reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

        # Read the cell data from the output file
        vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
        output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
        output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])
        
        # # Check if cell data is identical
        @test reference_points ≈ output_points atol = 1e-13
        @test reference_cells == output_cells
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
        Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree, ascii = false, compress = false)

        # Test plotting  
        # Read the cell data from the reference file
        reference_file = joinpath(input_data_folder, output_filename)
        vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
        reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
        reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

        # Read the cell data from the output file
        vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
        output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
        output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])
        
        # Check if cell data is identical
        @test reference_points ≈ output_points atol = 1e-14
        @test reference_cells == output_cells
    end
end
# -----------------------------------------------------------------------------
end