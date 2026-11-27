using Mantis

using Test

EXPORT = false

function integrate_volume(
    geometry::Geometry.AbstractGeometry{manifold_dim}, quadrature
) where {manifold_dim}
    one = Forms.FormPullback(
        Forms.ConstantFormSpace(manifold_dim, geometry, "1"), Forms.Canonical
    )
    area = mapreduce(+, 1:Geometry.get_num_elements(geometry); init=0.0) do element_id
        element_rule = Quadrature.get_element_quadrature_rule(quadrature, element_id)
        nodes = Quadrature.get_nodes(element_rule)
        weights = Quadrature.get_weights(element_rule)
        one_eval = Forms.evaluate(one, element_id, nodes)[1][1]
        element_area = mapreduce(
            pair -> pair[1] * pair[2], +, zip(one_eval, weights); init=0.0
        )

        return element_area
    end

    return area
end

function integrate_boundary(
    geometry::Geometry.AbstractGeometry{manifold_dim}, quadrature
) where {manifold_dim}
    one = Forms.FormPullback(
        Forms.ConstantFormSpace(manifold_dim, geometry, "1"), Forms.Canonical
    )
    area = mapreduce(+, 1:Geometry.get_num_elements(geometry); init=0.0) do element_id
        if !Quadrature.is_cut(quadrature, element_id)
            return 0.0
        end

        element_rule = Quadrature.get_element_quadrature_rule(quadrature, element_id)
        nodes = Quadrature.get_nodes(element_rule)
        weights = Quadrature.get_weights(element_rule)
        one_eval = Forms.evaluate(one, element_id, nodes)[1][1]
        element_area = mapreduce(
            pair -> pair[1] * pair[2], +, zip(one_eval, weights); init=0.0
        )

        return element_area
    end

    return area
end

const num_elements_2d = (20, 20)
const num_elements_3d = (20, 20, 20)
const order = 7
const background_geometry_2d = Geometry.create_cartesian_box(
    (-1.0, -1.0), (2.0, 2.0), num_elements_2d
)
const background_geometry_3d = Geometry.create_cartesian_box(
    (-1.0, -1.0, -1.0), (2.0, 2.0, 2.0), num_elements_3d
)

# B²
const radius = 0.8
const center_2d = (0.0, 0.0)
level_set(x) = (x[1] - center_2d[1])^2 + (x[2] - center_2d[2])^2 - radius ^ 2
active_elements, cut_elements = Geometry.level_set_identification(
    background_geometry_2d, level_set
)
immersed_geometry = Geometry.create_immersed_geometry(
    background_geometry_2d, active_elements
)
Q = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1)
    )[1],
    order,
)
@test isapprox(integrate_volume(immersed_geometry, Q), π * radius^2; atol=1e-7)
S = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1)
    )[1],
    order;
    surface=true,
)
@test isapprox(integrate_boundary(immersed_geometry, S), 2π * radius; atol=1e-7)
if EXPORT
    immersed_points = Plot.ImmersedPoints(
        immersed_geometry, cut_elements, level_set; n_subcells=1, degree=3
    )
    Plot.export_geometry_to_vtk(immersed_geometry, immersed_points, "circle_geo")
end

# B³
const center_3d = (0.0, 0.0, 0.0, 0.0)
level_set(x) =
    (x[1] - center_3d[1])^2 + (x[2] - center_3d[2])^2 + (x[3] - center_3d[3])^2 - radius ^ 2
active_elements, cut_elements = Geometry.level_set_identification(
    background_geometry_3d, level_set
)
immersed_geometry = Geometry.create_immersed_geometry(
    background_geometry_3d, active_elements
)
Q = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1, order + 1)
    )[1],
    order,
)
@test isapprox(integrate_volume(immersed_geometry, Q), 4 / 3 * π * radius^3; atol=1e-7)
S = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1, order + 1)
    )[1],
    order;
    surface=true,
)
@test isapprox(integrate_boundary(immersed_geometry, S), 4π * radius^2; atol=1e-6)
if EXPORT
    immersed_points = Plot.ImmersedPoints(
        immersed_geometry, cut_elements, level_set; n_subcells=2, degree=1
    )
    Plot.export_geometry_to_vtk(immersed_geometry, immersed_points, "cassini_geo")
end

# Cassini Oval
const cassini_background = Geometry.create_cartesian_box(
    (-2.0, -2.0, -2.0), (4.0, 4.0, 4.0), num_elements_3d
)
p = [(-1.0, 0.0, -sqrt(3) / 2.0), (1.0, 0.0, -sqrt(3) / 2.0), (0.0, 0.0, sqrt(3) / 2.0)]
b = 1.6
level_set(x) = mapreduce(point -> sum((x .- point) .* (x .- point)), *, p) - b^2
active_elements, cut_elements = Geometry.level_set_identification(
    cassini_background, level_set
)
immersed_geometry = Geometry.create_immersed_geometry(cassini_background, active_elements)
Q = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1, order + 1)
    )[1],
    order,
)
@test isapprox(integrate_volume(immersed_geometry, Q), 1.4603490821; atol=1e-6)
S = Quadrature.ImmersedQuadrature(
    immersed_geometry,
    cut_elements,
    level_set,
    Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, (order + 1, order + 1, order + 1)
    )[1],
    order;
    surface=true,
)
@test isapprox(integrate_boundary(immersed_geometry, S), 9.2135653606; atol=1e-4)
if EXPORT
    immersed_points = Plot.ImmersedPoints(
        immersed_geometry, cut_elements, level_set; n_subcells=2, degree=1
    )
    Plot.export_geometry_to_vtk(immersed_geometry, immersed_points, "cassini_geo")
end

# Non-matching dimensions between geometry and quadrature
@test_throws MethodError Quadrature.ImmersedQuadrature(
    background_geometry_2d,
    Bool[],
    x -> x,
    Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, (1, 1, 1))[1],
    order,
)
# Non-matching number of elements between geometry and cut_elements
@test_throws ArgumentError Quadrature.ImmersedQuadrature(
    background_geometry_3d,
    Bool[],
    x -> x,
    Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, (1, 1, 1))[1],
    order,
)
