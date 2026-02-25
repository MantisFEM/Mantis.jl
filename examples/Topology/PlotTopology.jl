using Mantis

geo = Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)))
# fig = Mantis.Plot.plot_topology(geo)

geo2 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)),
        (LinRange(1.0, 3.0, 3), LinRange(1.0, 3.0, 4)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3]]),
)
# fig = Mantis.Plot.plot_topology(geo2)

geo3 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)),
        (LinRange(1.0, 3.0, 4), LinRange(1.0, 3.0, 4)),
        (LinRange(1.0, 3.0, 9), LinRange(3.0, 5.0, 6)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3], [3, 6, 7, 8]]),
)
fig = Mantis.Plot.plot_topology(geo3)

display(fig)
