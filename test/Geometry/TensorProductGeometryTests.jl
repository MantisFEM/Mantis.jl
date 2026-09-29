module TensorProductGeometryTests

using Mantis

# Refer to the following file for method and variable definitions
include("GeometryTestsHelpers.jl")

import ReadVTK
using Test

# Test Square Tensor Product Geometry -----------------------------------------
# Generate a tensor product geometry by combining two lines

# Line geometries
line_1_geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (10,))
line_2_geometry = Geometry.create_cartesian_box((2.0,), (1.0,), (10,))

# Tensor product geometry
tensor_prod_geometry = Geometry.TensorProductGeometry((line_1_geometry, line_2_geometry))

# Set file name and path
file_name = "tensor_product_geometry.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
# Generate the vtk file
Plot.plot(
    tensor_prod_geometry;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test Cylinder Tensor Product Geometry ---------------------------------------
deg = 2
nθ_elements = 4
Wt = 2.0 * pi / nθ_elements
b = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = collect(LinRange(0.0, nθ_elements, nθ_elements + 1))
patch = Geometry.CartesianGeometry(breakpoints)
B = FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = FunctionSpaces.GTBSplineSpace((B,), [1])

# control points for geometry
# radius of cylinder is 1.0
geom_coeffs_circle = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
cylinder_circle_geometry = FunctionSpaces.FEGeometry(GB, geom_coeffs_circle)
dx_cylinder_line = 0.1
nz_elements = 10
cylinder_line_geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (nz_elements,))

# Tensor product geometry
cylinder_tensor_prod_geometry = Geometry.TensorProductGeometry((
    cylinder_circle_geometry, cylinder_line_geometry
))

# Set file name and path
file_name = "tensor_product_cylinder_geometry.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
# Generate the vtk file
Plot.plot(
    cylinder_tensor_prod_geometry;
    vtk_filename=output_file_path[1:(end - 4)], #remove the file extension
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)
# Read the point and cell data from the reference file
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the point and cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if point and cell data is identical
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))

# Test Jacobian with single point evaluation
# We check the Jacobian
#    J^{k}_{ij} = \partial{\Phi^{i}(\boldsymbol{x}_{k})}{\partial x^{0}_{j}}
# at four points k at different z levels

for element_row_idx in 1:nz_elements
    # Compute Jacobian at x_{1} = [1.0, 0.0, z]
    # This corresponds to the point with local coordinates [0.0, 0.0] on the first element of
    # row element_row_idx
    ξ = Points.TensorProductPoints(([0.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 1, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [0.0, 1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the first element
    # of row element_row_idx
    ξ = Points.TensorProductPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        -0.5*π 0.0
        0.0 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 1, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [-1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the second element
    # of row element_row_idx
    ξ = Points.TensorProductPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        -0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 2, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [0.0, -1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the third element
    # of row element_row_idx
    ξ = Points.TensorProductPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.5*π 0.0
        0.0 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 3, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian again at x_{1} = [1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the fourth element
    # of row element_row_idx
    ξ = Points.TensorProductPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 4, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))
end

# -----------------------------------------------------------------------------

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(geometry, answers)
    @test all(Geometry.get_factor_num_elements(geometry) .== answers[1])

    @test Geometry.get_num_patches(geometry) == answers[2]
    @test Geometry.get_num_elements(geometry) == answers[3]
    @test Geometry.get_manifold_dim(geometry) == answers[4]
    @test Geometry.get_image_dim(geometry) == answers[5]
    @test Geometry.get_num_elements_per_patch(geometry) == answers[6]
    @test Geometry.get_num_elements(geometry, 1) == answers[7]
    @test Geometry.get_element_lengths(geometry, 1) == answers[8]
    @test all(isapprox.(Geometry.get_element_measure(geometry, 1), answers[9], rtol=1e-14))

    patch_id, local_element_id = Geometry.get_patch_and_local_element_id(
        geometry, answers[11]
    )
    @test (patch_id, local_element_id) == answers[10]
    @test Geometry.get_global_element_id(geometry, patch_id, local_element_id) ==
        answers[11]

    @test all(
        all.([
            isapprox.(
                Geometry.get_element_vertices(geometry, 1)[i], answers[12][i], rtol=1e-14
            ) for i in eachindex(answers[12])
        ]),
    )

    return nothing
end

# Reduction test, single-patch, single element, single geometry, 1D.
cg1 = Geometry.CartesianGeometry(([-1, 1],))
tpgeometry1 = Geometry.TensorProductGeometry((cg1,))
answers_1 = ((1,), 1, 1, 1, 1, (1,), 1, (2.0,), 2.0, (1, 1), 1, ((-1, 1),))
basic_tests(tpgeometry1, answers_1)

# All cartesian. 3D: 2D (2 patches) tensored with 1D (3 patches).
cg1d = Geometry.CartesianGeometry(
    ((LinRange(0.0, 1.0, 2),), (LinRange(1.0, 2.0, 3),), (LinRange(2.0, 3.0, 4),)),
    # A chain of three line patches: [0, 1], [1, 2] and [2, 3].
    Topology.MeshTopology(((1, 2), (2, 3), (3, 4)), Topology.LINE),
)
cg2d = Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)), # First patch
        (LinRange(1.0, 2.0, 6), LinRange(0.0, 1.0, 7)), # Second patch
    ),
    # The patches meet along x = 1, so they share vertices 2 and 3.
    Topology.MeshTopology(((1, 2, 3, 4), (2, 5, 6, 3)), Topology.QUAD),
)
tpgeometry2 = Geometry.TensorProductGeometry((cg2d, cg1d))
answers_2 = (
    (42, 6),
    6,
    252,
    3,
    3,
    (12, 30, 24, 60, 36, 90),
    12,
    (1.0 / 3, 0.25, 1.0),
    1.0 / 12.0,
    (5, 36),
    162,
    ((0.0, 1.0 / 3.0), (0.0, 0.25), (0.0, 1.0)),
)
basic_tests(tpgeometry2, answers_2)

# Test Topology of Tensor Product Geometries ----------------------------------
# The product of single-patch geometries is a single patch, of any factor dimensions.
square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (3, 2))
line = Geometry.create_cartesian_box((0.0,), (1.0,), (4,))
tp_square_line = Geometry.TensorProductGeometry((square, line))
@test Geometry.get_manifold_dim(tp_square_line) == 3
@test Geometry.get_topology(tp_square_line) === Topology.SINGLE_PATCH_TOPOLOGY_3D
@test Geometry.get_factor_manifold_indices(tp_square_line) == ((1, 2), (3,))
# Topologies, and hence tensor-product geometries, are at most three-dimensional.
@test_throws ArgumentError Geometry.TensorProductGeometry((square, square))

# On a single patch, the elements on each topological object are those of the Cartesian
# geometry with the same elements.
cartesian_box = Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3, 2, 4))
for geometric_dim in 0:3, local_object_id in 1:size(Topology.HEX, geometric_dim + 1)
    @test Geometry.get_elements(tp_square_line, 1, local_object_id, geometric_dim) ==
        Geometry.get_elements(cartesian_box, 1, local_object_id, geometric_dim)
end

# A multi-patch factor gives a multi-patch product, whose topology is the tensor product of
# the factor topologies.
topology_2 = Geometry.get_topology(tpgeometry2)
@test topology_2 isa Topology.TensorProductTopology
@test Topology.get_factor_topologies(topology_2) ==
    (Geometry.get_topology(cg2d), Geometry.get_topology(cg1d))
@test Topology.get_num_patches(topology_2) == Geometry.get_num_patches(tpgeometry2)
# Two quadrilaterals sharing an edge, extruded through three segments.
@test size(topology_2) == (6 * 4, 7 * 4 + 6 * 3, 2 * 4 + 7 * 3, 2 * 3)

# Converting to a Cartesian geometry keeps the topology.
cartesian_2 = convert(Geometry.CartesianGeometry, tpgeometry2)
@test Geometry.get_topology(cartesian_2) === topology_2
@test Geometry.get_num_elements_per_patch(cartesian_2) ==
    Geometry.get_num_elements_per_patch(tpgeometry2)

"""
    interface_elements(geometry, patch_id, local_id, object_dim)

Return the elements on the local object `local_id` of dimension `object_dim` of `patch_id`,
and those on the same object seen from its first neighbouring patch, re-ordered with the
rotation and orientation that the topology reports.
"""
function interface_elements(geometry, patch_id, local_id, object_dim)
    topology = Geometry.get_topology(geometry)
    neighbours = Topology.compute_neighbours(topology, patch_id, local_id, object_dim)
    neighbour_id, neighbour_local_id, rotation, orientation = neighbours[:, 1]
    own = Geometry.get_elements(geometry, patch_id, local_id, object_dim)
    other = Geometry.get_elements(
        geometry, neighbour_id, neighbour_local_id, object_dim, rotation, orientation
    )
    return own, other
end

# The local numbering of the tensor-product topology is the one the geometry uses: the
# elements on either side of an interface are the same elements of the mesh. In 2D, the
# rotation and orientation also match them one by one.
two_patch_line = Geometry.CartesianGeometry(
    ((collect(0.0:0.5:1.0),), (collect(1.0:0.25:2.0),)),
    Topology.MeshTopology(((1, 2), (2, 3)), Topology.LINE),
)
centre_2d = Points.TensorProductPoints([0.5], [0.5])
for factors in ((two_patch_line, line), (line, two_patch_line))
    geometry = convert(Geometry.CartesianGeometry, Geometry.TensorProductGeometry(factors))
    topology = Geometry.get_topology(geometry)
    @test topology isa Topology.TensorProductTopology
    _, interfaces = Topology.get_boundaries_and_interfaces(topology)
    for (dim, interface_id) in interfaces
        dim == 1 || continue
        patch_id = topology[2, 3][interface_id][1]
        local_id = Topology.get_local_id(topology, patch_id, interface_id, 1)
        own, other = interface_elements(geometry, patch_id, local_id, 1)
        # Along the interface (the direction in which its position is zero), the centroids
        # of paired elements coincide.
        position = Topology.id_to_position(Topology.QUAD, 1, local_id)
        along = findfirst(iszero, position)
        centroid(element_id) = Geometry.evaluate(geometry, element_id, centre_2d)[1, along]
        @test length(own) == length(other)
        @test centroid.(own) ≈ centroid.(other)
    end
end

# In 3D, the element sets on either side of each interface face coincide.
centre_3d = Points.TensorProductPoints([0.5], [0.5], [0.5])
geometry_3d = convert(
    Geometry.CartesianGeometry,
    Geometry.TensorProductGeometry((two_patch_line, two_patch_line, two_patch_line)),
)
topology_3d = Geometry.get_topology(geometry_3d)
_, interfaces_3d = Topology.get_boundaries_and_interfaces(topology_3d)
for (dim, interface_id) in interfaces_3d
    dim == 2 || continue
    patch_id = topology_3d[3, 4][interface_id][1]
    local_id = Topology.get_local_id(topology_3d, patch_id, interface_id, 2)
    own, other = interface_elements(geometry_3d, patch_id, local_id, 2)
    along = findall(iszero, Topology.id_to_position(Topology.HEX, 2, local_id))
    centroid(element_id) = Geometry.evaluate(geometry_3d, element_id, centre_3d)[1, along]
    @test sort(centroid.(own)) ≈ sort(centroid.(other))
end
# -----------------------------------------------------------------------------

end
