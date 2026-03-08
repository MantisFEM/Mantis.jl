using Mantis

# # Generate a parent topology
# parent_topology = Topology.MeshTopology([[1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 3, 6, 11, 12, 7]])

# # Construct a geometry for it
# geo3d2 = Mantis.Geometry.CartesianGeometry(
#     (
#         (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6), LinRange(0.0, 1.0, 8)),
#         (LinRange(1.0, 3.0, 3), LinRange(1.0, 3.0, 4), LinRange(0.0, 1.0, 8)),
#     ),
#     parent_topology,
# )
# fig = Mantis.Plot.plot_topology(geo3d2)

# display(fig)


# # Generate a skeleton topology
# skeleton_topology = Topology.SkeletonTopology(parent_topology)

# patch_parents = Topology.get_patch_parents(skeleton_topology, 1)

# skeleton_geometry = Mantis.Geometry.SkeletonGeometry(geo3d2)


# Oriol Periodic B-Splines

starting_points = (0.0, 0.0)
box_sizes = (2.0, 1.0)
num_elements = (4, 2)
deg = (3, 2)
section_spaces = (
    Mantis.FunctionSpaces.Bernstein(deg[1]), Mantis.FunctionSpaces.Bernstein(deg[2])
)
regularities = (deg[1]-1, deg[2]-1)

# create tensor-product space
Bx, By = Mantis.FunctionSpaces.create_dim_wise_bspline_spaces(
    starting_points, box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
)
Bx_periodic = Mantis.FunctionSpaces.GTBSplineSpace((Bx,), [deg[1]-1])
By_periodic = Mantis.FunctionSpaces.GTBSplineSpace((By,), [deg[2]-1])



using Makie

fig = Figure()
ax = Axis(fig[1, 1])
dx = box_sizes[1]/num_elements[1]
xi_plot = LinRange(0.0, 1.0, 50)

for element_id in 1:num_elements[1]
    xi = Mantis.Points.CartesianPoints((LinRange(0.0, 1.0, 50),))
    Bx_periodic_evaluated, basis_indices = Mantis.FunctionSpaces.evaluate(Bx_periodic, element_id, xi)

    x_plot = xi_plot.*dx .+ (element_id - 1)*dx
    for basis_to_plot_id in 1:length(basis_indices)
        lines!(ax, x_plot, Bx_periodic_evaluated[1][1][1][:, basis_to_plot_id]; color=Makie.wong_colors()[basis_indices[basis_to_plot_id]])
    end
end
display(fig)