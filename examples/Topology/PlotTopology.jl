using Mantis

geo = Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)))
geo1d = Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4),))
geo1d2 = Mantis.Geometry.CartesianGeometry(
    ((LinRange(0.0, 1.0, 4),), (LinRange(1.0, 2.0, 6),)),
    Topology.MeshTopology([[1, 2], [2, 3]]),
)
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
# fig = Mantis.Plot.plot_topology(geo3)

geo4 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)),
        (LinRange(1.0, 3.0, 4), LinRange(1.0, 3.0, 4)),
        (LinRange(1.0, 3.0, 9), LinRange(3.0, 5.0, 6)),
        (LinRange(3.0, 4.0, 4), LinRange(1.0, 3.0, 6)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3], [3, 6, 7, 8], [5, 9, 10, 6]]),
)
# fig = Mantis.Plot.plot_topology(geo4)

geo5 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6)),
        (LinRange(1.0, 3.0, 4), LinRange(1.0, 3.0, 4)),
        (LinRange(1.0, 3.0, 9), LinRange(3.0, 5.0, 6)),
        (LinRange(3.0, 4.0, 4), LinRange(1.0, 3.0, 6)),
        (LinRange(3.0, 4.0, 4), LinRange(3.0, 5.0, 6)),
    ),
    Topology.MeshTopology([
        [1, 2, 3, 4], [2, 5, 6, 3], [3, 6, 7, 8], [5, 9, 10, 6], [6, 10, 11, 7]
    ]),
)
# fig = Mantis.Plot.plot_topology(geo5)

geo3d2 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6), LinRange(0.0, 1.0, 8)),
        (LinRange(1.0, 3.0, 3), LinRange(1.0, 3.0, 4), LinRange(0.0, 1.0, 8)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 3, 6, 11, 12, 7]]),
)
fig = Mantis.Plot.plot_topology(geo3d2)

display(fig)
