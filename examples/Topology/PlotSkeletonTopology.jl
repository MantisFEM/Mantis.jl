using Mantis
using GLMakie

# Generate a parent topology
parent_topology = Topology.MeshTopology([
    [1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 3, 6, 11, 12, 7]
])

# Construct a geometry for it
geo3d2 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 4), LinRange(0.0, 1.0, 4)),
        (LinRange(1.0, 3.0, 4), LinRange(1.0, 3.0, 4), LinRange(0.0, 1.0, 4)),
    ),
    parent_topology,
)
# fig = Mantis.Plot.plot_topology(geo3d2)

# display(fig)

# Generate a skeleton topology
skeleton_topology = Mantis.Topology.SkeletonTopology(parent_topology)

patch_parents = Mantis.Topology.get_patch_parents(skeleton_topology, 1)

skeleton_geometry = Mantis.Geometry.SkeletonGeometry(geo3d2)

num_elements_per_patch = Mantis.Geometry.get_num_elements_per_patch(skeleton_geometry)

parent_elements_ids, patch_parents = Mantis.Geometry.get_parent_elements(skeleton_geometry, 1, 1)

constituent_points = tuple(LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3))
points_skeleton = Mantis.Points.CartesianPoints(constituent_points)

points_parent = Mantis.Geometry.skeleton_element_to_parent_element_coords(
    points_skeleton, patch_parents[2, 1], patch_parents[3, 1], patch_parents[4, 1])

for point in points_parent
    display(point)
end

# skeleton_geo_eval = Mantis.Geometry.evaluate(skeleton_geometry, 10, points_skeleton)

function plot_points_sequential(geometry, points::Matrix{Float64}, lag::Float64=0.1)
    size(points, 2) == 3 || throw(ArgumentError("points must be an n×3 matrix"))
    
    fig = Mantis.Plot.plot_topology(geometry)
    resize!(fig, 1600, 1200)
    ax = fig.content[1]
    ax.elevation[] = π/6
    ax.azimuth[] = 2*π/10

    xs = Observable(Float64[])
    ys = Observable(Float64[])
    zs = Observable(Float64[])

    scatter!(ax, xs, ys, zs)

    display(fig)

    for i in 1:size(points, 1)
        push!(xs[], points[i, 1])
        push!(ys[], points[i, 2])
        push!(zs[], points[i, 3])
        notify(xs)
        notify(ys)
        notify(zs)
        sleep(lag)
    end
end

# skeleton_topology[3, 1]

# for element_id in 1:Mantis.Geometry.get_num_elements(skeleton_geometry)
#     skeleton_geo_eval = Mantis.Geometry.evaluate(skeleton_geometry, element_id, points_skeleton)
#     plot_points_sequential(geo3d2, skeleton_geo_eval, 0.2)
# end

# for element_id in 19:27
#     skeleton_geo_eval = Mantis.Geometry.evaluate(skeleton_geometry, element_id, points_skeleton)
#     plot_points_sequential(geo3d2, skeleton_geo_eval, 0.2)
# end

parent_elements_ids, patch_parents = Mantis.Geometry.get_parent_elements(skeleton_geometry, 3, 1)
points_parent = Mantis.Geometry.skeleton_element_to_parent_element_coords(
points_skeleton, patch_parents[2, 1], patch_parents[3, 1], patch_parents[4, 1])

# # Oriol Periodic B-Splines

# starting_points = (0.0, 0.0)
# box_sizes = (2.0, 1.0)
# num_elements = (4, 2)
# deg = (3, 2)
# section_spaces = (
#     Mantis.FunctionSpaces.Bernstein(deg[1]), Mantis.FunctionSpaces.Bernstein(deg[2])
# )
# regularities = (deg[1]-1, deg[2]-1)

# # create tensor-product space
# Bx, By = Mantis.FunctionSpaces.create_dim_wise_bspline_spaces(
#     starting_points, box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
# )
# Bx_periodic = Mantis.FunctionSpaces.GTBSplineSpace((Bx,), [deg[1]-1])
# By_periodic = Mantis.FunctionSpaces.GTBSplineSpace((By,), [deg[2]-1])

# using Makie

# fig = Figure()
# ax = Axis(fig[1, 1])
# dx = box_sizes[1]/num_elements[1]
# xi_plot = LinRange(0.0, 1.0, 50)

# for element_id in 1:num_elements[1]
#     xi = Mantis.Points.CartesianPoints((LinRange(0.0, 1.0, 50),))
#     Bx_periodic_evaluated, basis_indices = Mantis.FunctionSpaces.evaluate(Bx_periodic, element_id, xi)

#     x_plot = xi_plot.*dx .+ (element_id - 1)*dx
#     for basis_to_plot_id in 1:length(basis_indices)
#         lines!(ax, x_plot, Bx_periodic_evaluated[1][1][1][:, basis_to_plot_id]; color=Makie.wong_colors()[basis_indices[basis_to_plot_id]])
#     end
# end
# display(fig)
