module PlotTests

import Mantis

import Mmap
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Plot")
output_data_folder = joinpath(data_folder, "output")

# Test CartesianGeometry -----------------------------------------------------------
for nx = 1:4
    for ny = 1:4
        breakpoints = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)))
        geom = Mantis.Geometry.CartesianGeometry(breakpoints)
        # Generate the plot
        output_filename = @sprintf "cartesian_test_nx_%d_ny_%d.vtu" nx ny
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)
    end
end

# Test FEMGeometry -----------------------------------------------------------
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
output_filename = "fem_geometry_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4)

# Test FEMGeometry - Lagrange -----------------------------------------------------------
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
output_filename = "fem_geometry_lagrange_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

# Test AnalGeometry -----------------------------------------------------------
n_subcells_to_test = 1:4
degrees_to_test = 1:4
for n_subcells in n_subcells_to_test 
    for degree in degrees_to_test
        # Rectangle based geometries input parameters
        n_elements = (2, 2)
        xy_start = [0.0, 0.0]
        xy_end = [1.0, 1.0]

        # Rectangle geometry
        rectangle = Mantis.Geometry.Rectangle{2, 2}(n_elements, xy_start, xy_end)

        # Generate the plot
        output_filename = @sprintf "rectangle_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(rectangle; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))

        # Mapped custom rectangle geometry
        function mapping(x::Vector{Float64})
            return [(x[1] + 0.2)*cos(x[2]), (x[1] + 0.2)*sin(x[2])]
        end

        function dmapping(x::Vector{Float64})
            return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
        end

        mapped_rectangle = Mantis.Geometry.MappedRectangle{2, 2}(n_elements, xy_start, xy_end, mapping, dmapping)

        # Generate the plot 
        output_filename = @sprintf "mapped_rectangle_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(mapped_rectangle; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
        # -----------------------------------------------------------------------------


        # Test Mapped geometry --------------------------------------------------------
        # This results in the same geometry as the custom mapped_rectangle, the difference is that is uses composition
        # via the MappedGeometry struct: we provide the straight geometry rectangle and then a mapping and it generates
        # the resulting geometry, which is a composition of the rectangle geometry and the mapping 
        dimension = (2, 2)

        curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)

        mapped_geometry = Mantis.Geometry.MappedGeometry(rectangle, curved_mapping)

        # Generate the plot 
        output_filename = @sprintf "mapped_geometry_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(mapped_geometry; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------

struct SpiralGeometry{n, m} <: Mantis.Geometry.AbstractAnalGeometry{n, m}
    n_elements::Int
end

function Mantis.Geometry.evaluate(geometry::SpiralGeometry{1, 3}, element_idx::Int64, ξ::Float64)
    Δt = 1.0/geometry.n_elements
    t = Δt * (element_idx - 1) + ξ * Δt
    return [cos(4.0*π*t), sin(4.0*π*t), 2.0*t]
end

line = SpiralGeometry{1, 3}(10)
Mantis.Plot.plot(line; vtk_filename = joinpath(output_data_folder, "helix_p_10"), n_subcells = 1, degree = 10)

end