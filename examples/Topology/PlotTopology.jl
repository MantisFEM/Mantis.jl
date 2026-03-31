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
geo22 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)),
        (LinRange(1.0, 2.0, 3), LinRange(0.0, 1.0, 3)),
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

geo2base = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)),
        (LinRange(1.0, 2.0, 3), LinRange(0.0, 1.0, 4)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3]]),#[5, 6, 7, 8]
)
tp1_geo2base = FunctionSpaces.create_bspline_space(
    (0.0,0.0), (1.0, 1.0), (3,4), (2,2), (1,1)
);
tp2_geo2base = FunctionSpaces.create_bspline_space(
    (1.0,0.0), (1.0, 1.0), (2,3), (2,2), (1,1)
);
# c0_geo2base = FunctionSpaces.C0Space((tp1_geo2base, tp2_geo2base), geo2base); # mismatched dofs.

geo2mapped = Mantis.Geometry.MappedGeometry(
    # Mantis.Geometry.CartesianGeometry(
    #     (
    #         (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)),
    #         (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4)),
    #     ),
    #     Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3]]),#[5, 6, 7, 8]
    # ),
    (
        Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5))),
        Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4))),
    ),
    (mapping_obj_1_geo2mapped, mapping_obj_2_geo2mapped),
    Topology.MeshTopology([[1, 2, 3, 4], [1, 5, 6, 2]]),
)

# fig = Mantis.Plot.plot_topology(geo2mapped)
tp1_geo2mapped = FunctionSpaces.create_bspline_space(
    (0.0,0.0), (1.0, 1.0), (3,4), (2,2), (1,1)
);
tp2_geo2mapped = FunctionSpaces.create_bspline_space(
    (0.0,0.0), (1.0, 1.0), (2,3), (2,2), (1,1)
);
c0_geo2mapped = FunctionSpaces.C0Space((tp1_geo2mapped, tp2_geo2mapped), geo2mapped);

function forcing_function(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sin(2.0 * pi * x[:, 1]) * sin(2.0 * pi * x[:, 2])]
end
f⁰ = Forms.AnalyticalFormField(0, forcing_function, geo2mapped, "f⁰");
formspace = Forms.FormSpace(0, c0_geo2mapped, "C0");
canonical_qrule = Quadrature.tensor_product_rule((5,5), Quadrature.gauss_legendre);
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geo2mapped));
sol = Assemblers.solve_zero_form_hodge_laplacian(formspace, f⁰, dΩ);

geo3_2cube = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)),
        (LinRange(1.0, 2.0, 3), LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 3, 6, 11, 12, 7]]),
)
tp1_geo3_2cube = FunctionSpaces.create_bspline_space(
    (0.0,0.0,0.0), (1.0, 1.0,1.0), (2,3,4), (2,2,2), (1,1,1)
);
tp2_geo3_2cube = FunctionSpaces.create_bspline_space(
    (1.0,0.0,0.0), (1.0, 1.0,1.0), (2,3,4), (2,2,2), (1,1,1)
);
c0_geo3_2cube = FunctionSpaces.C0Space((tp1_geo3_2cube, tp2_geo3_2cube), geo3_2cube);
function forcing_function_3D(x::Matrix{Float64})
    return [@. 12.0 * pi^2 * sin(2.0 * pi * x[:, 1]) * sin(2.0 * pi * x[:, 2]) * sin(2.0 * pi * x[:, 3])]
end
f⁰_3D = Forms.AnalyticalFormField(0, forcing_function_3D, geo3_2cube, "f⁰");
formspace_3D = Forms.FormSpace(0, c0_geo3_2cube, "C0");
canonical_qrule_3D = Quadrature.tensor_product_rule((5,5,5), Quadrature.gauss_legendre);
dΩ_3D = Quadrature.StandardQuadrature(canonical_qrule_3D, Geometry.get_num_elements(geo3_2cube));
sol_3D = Assemblers.solve_zero_form_hodge_laplacian(formspace_3D, f⁰_3D, dΩ_3D);

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

geo3d1 = Mantis.Geometry.CartesianGeometry(
    ((LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6), LinRange(0.0, 1.0, 8)),),
    Topology.MeshTopology([[1, 2, 3, 4, 5, 6, 7, 8]]),
)
geo3d2 = Mantis.Geometry.CartesianGeometry(
    (
        (LinRange(0.0, 1.0, 4), LinRange(1.0, 3.0, 6), LinRange(0.0, 1.0, 8)),
        (LinRange(1.0, 3.0, 3), LinRange(1.0, 3.0, 4), LinRange(0.0, 1.0, 8)),
    ),
    Topology.MeshTopology([[1, 2, 3, 4, 5, 6, 7, 8], [2, 9, 10, 3, 6, 11, 12, 7]]),
)
# fig = Mantis.Plot.plot_topology(geo3d2)

function mapping_mobius(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    u = x[1]
    v = x[2]
    return [
        (1.0 + (v / 2) * cos(u / 2)) * cos(u),
        (1.0 + (v / 2) * cos(u / 2)) * sin(u),
        (v / 2) * sin(u / 2),
    ]
end
function dmapping_mobius(x::AbstractVector{Float64})
    u = x[1]
    v = x[2]
    return [
        [-v / 4 * sin(u / 2) * cos(u) - v / 2 * cos(u / 2) * sin(u) 1 / 2 *
                                                                    cos(u / 2) *
                                                                    cos(u)]
        [-v / 4 * sin(u / 2) * cos(u) - v / 2 * cos(u / 2) * cos(u) 1 / 2 *
                                                                    cos(u / 2) *
                                                                    sin(u)]
        [v / 4 * cos(u / 2) 1 / 2 * sin(u / 2)]
    ]
end
mapping_obj_mobius = Geometry.Mapping((2, 3), mapping_mobius, dmapping_mobius)

mobius_1patch = Mantis.Geometry.MappedGeometry(
    (
        Mantis.Geometry.CartesianGeometry((
            LinRange(0.0, 2.0 * pi, 21), LinRange(-1.0, 1.0, 21)
        )),
    ),
    mapping_obj_mobius,
    Topology.MeshTopology([[1, 2, 4, 3]]),
)
mobius_2patch = Mantis.Geometry.MappedGeometry(
    (
        Mantis.Geometry.CartesianGeometry((LinRange(0.0, pi, 9), LinRange(-1.0, 1.0, 9))),
        Mantis.Geometry.CartesianGeometry((
            LinRange(pi, 2.0 * pi, 9), LinRange(-1.0, 1.0, 9)
        )),
    ),
    mapping_obj_mobius,
    Topology.MeshTopology([[1, 2, 3, 4], [2, 5, 6, 3]]),#[2, 4, 1, 3]]),
)
# fig = Mantis.Plot.plot_topology(mobius_2patch)

function mapping_cylinder(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    return [1.0 * cos(x[1]), 1.0 * sin(x[1]), x[2]]
end
function dmapping_cylinder(x::AbstractVector{Float64})
    return [
        [-sin(x[1]) cos(x[1])]
        [0.0 0.0]
        [0.0 1.0]
    ]
end
mapping_obj_cylinder = Geometry.Mapping((2, 3), mapping_cylinder, dmapping_cylinder)

cylinder_1patch = Mantis.Geometry.MappedGeometry(
    (
        Mantis.Geometry.CartesianGeometry((
            LinRange(0.0, 2.0 * pi, 11), LinRange(0.0, 1.0, 6)
        )),
    ),
    mapping_obj_cylinder,
    Topology.MeshTopology([[1, 2, 3, 4]]),
)
cylinder_2patch = Mantis.Geometry.MappedGeometry(
    (
        Mantis.Geometry.CartesianGeometry((LinRange(0.0, pi, 9), LinRange(-1.0, 1.0, 9))),
        Mantis.Geometry.CartesianGeometry((
            LinRange(pi, 2.0 * pi, 9), LinRange(-1.0, 1.0, 9)
        )),
    ),
    mapping_obj_cylinder,
    Topology.MeshTopology([[1, 2, 3, 4], [2, 1, 4, 3]]),#[2, 4, 1, 3]]),
)

function mapping_rect(x::AbstractVector{Float64})
    # 0.0 <= x[1] <= 1.0 and 0.0 <= x[2] <= 1.0
    u = x[1]
    v = x[2]
    return [u, v, u + v]
end
function dmapping_rect(x::AbstractVector{Float64})
    u = x[1]
    v = x[2]
    return [
        [1.0 0.0]
        [0.0 1.0]
        [1.0 1.0]
    ]
end
mapping_obj_rect = Geometry.Mapping((2, 3), mapping_rect, dmapping_rect)

rect_1patch = Mantis.Geometry.MappedGeometry(
    (Mantis.Geometry.CartesianGeometry((LinRange(0.0, 1.0, 21), LinRange(0.0, 1.0, 21))),),
    mapping_obj_rect,
    Topology.MeshTopology([[1, 2, 3, 4]]),
)

# fig = Mantis.Plot.plot_topology(cylinder_1patch)

# display(fig)
