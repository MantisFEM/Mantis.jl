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

function mapping_patch_1_geo2mapped(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [-x[2], x[1]]
end
function dmapping_patch_1_geo2mapped(x::AbstractVector{Float64})
    return [
        [0.0 1.0]
        [-1.0 0.0]
    ]
end
mapping_obj_1_geo2mapped = Geometry.Mapping(
    (2, 2), mapping_patch_1_geo2mapped, dmapping_patch_1_geo2mapped
)
function mapping_patch_2_geo2mapped(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [x[1], x[2]]
end
function dmapping_patch_2_geo2mapped(x::AbstractVector{Float64})
    return [
        [1.0 0.0]
        [0.0 1.0]
    ]
end
mapping_obj_2_geo2mapped = Geometry.Mapping(
    (2, 2), mapping_patch_2_geo2mapped, dmapping_patch_2_geo2mapped
)

geo2mapped = Mantis.Geometry.MappedGeometry(
    Mantis.Geometry.CartesianGeometry(
        (
            (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)),
            (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4)),
        ),
        Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3]]),#[5, 6, 7, 8]
    ),
    # (
    #     Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3))),
    #     Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4))),
    # ),
    (mapping_obj_1_geo2mapped, mapping_obj_2_geo2mapped),
    Topology.MeshTopology([[1, 2, 3, 4], [1, 5, 6, 2]]),
)
fig = Mantis.Plot.plot_topology(geo2mapped)

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
# fig = Mantis.Plot.plot_topology(geo3d2)

display(fig)
