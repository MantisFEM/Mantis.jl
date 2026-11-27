import Suppressor
import ImplicitIntegration

struct ImmersedQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}
    canonical_rule::Q
    cut_elements::Vector{Bool}
    cut_rules::Vector{
        CanonicalQuadratureRule{
            manifold_dim,
            Points.PointSet{manifold_dim, Float64, NTuple{manifold_dim, Vector{Float64}}},
        },
    }
    num_elements::Int

    function ImmersedQuadrature(
        geometry::G,
        cut_elements::Vector{Bool},
        level_set::Function,
        canonical_rule::Q,
        order::Int;
        surface=false,
    ) where {
        manifold_dim,
        G <: Geometry.AbstractGeometry{manifold_dim},
        Q <: AbstractElementQuadratureRule{manifold_dim},
    }
        num_elements = Geometry.get_num_elements(geometry)
        if num_elements != length(cut_elements)
            return throw(
                ArgumentError(
                    LazyString(
                        "The number of elements in the geometry must match the length of the
                        cut-element identification vector. Got ",
                        num_elements,
                        " and ",
                        length(cut_elements),
                        ".",
                    ),
                ),
            )
        end

        # FIX: This will currently lead to the method `get_element_quadrature_rule` being
        # type unstable, since the quadrature type of `element_id` will depend on whether
        # the element is cut or not, and is indexed at runtime.
        # The best fix is likely for `Integral` to evaluate by calling `get_element_nodes`,
        # `get_element_weights` directly, regardless of the quadrature rule.
        cut_rules = Vector{
            CanonicalQuadratureRule{
                manifold_dim,
                Points.PointSet{
                    manifold_dim, Float64, NTuple{manifold_dim, Vector{Float64}}
                },
            },
        }(
            undef, num_elements
        )
        for element_id in 1:num_elements
            if cut_elements[element_id]
                vertices = Geometry.get_element_vertices(geometry, element_id)
                # ImplicitIntegration.quadgen takes in two points, `ll` and `ur`, defining a
                # the bounding box of the domain for which to provide the quadrature rule.
                # In this case the bounding box is the element.
                ll = ntuple(k -> vertices[k][1], manifold_dim)
                ur = ntuple(k -> vertices[k][2], manifold_dim)
                quad_data = Suppressor.@suppress ImplicitIntegration.quadgen(
                    level_set, ll, ur; order=order, surface=surface
                )[1]
                # The quadrature data `ImplicitIntegration.quadgen` is given the element's
                # coordinates, so we need to pullback the nodes and weights to the canonical
                # domain.
                nodes, weights = pullback_quad_data(
                    quad_data.coords, quad_data.weights, ll, ur
                )
                cut_rules[element_id] = CanonicalQuadratureRule(
                    Points.PointSet(nodes), weights
                )
            end
        end

        return new{manifold_dim, Q}(canonical_rule, cut_elements, cut_rules, num_elements)
    end
end

function pullback_quad_data(
    quad_nodes::AbstractVector,
    weights::Vector{Float64},
    lc::NTuple{manifold_dim, Float64},
    hc::NTuple{manifold_dim, Float64},
) where {manifold_dim}
    nodes = [
        ntuple(k -> (quad_nodes[point][k] - lc[k]) / (hc[k] - lc[k]), manifold_dim) for
        point in eachindex(quad_nodes)
    ]
    for i in eachindex(weights), k in 1:manifold_dim
        weights[i] /= (hc[k] - lc[k])
    end

    return nodes, weights
end

get_cut_elements(rule::ImmersedQuadrature) = rule.cut_elements
get_cut_quadrature_rules(rule::ImmersedQuadrature) = rule.cut_rules
get_num_evaluation_elements(rule::ImmersedQuadrature) = rule.num_elements
get_num_base_elements(rule::ImmersedQuadrature) = rule.num_elements
get_element_idxs(rule::ImmersedQuadrature, element_id::Int) = [element_id]
get_canonical_quadrature_rule(rule::ImmersedQuadrature) = rule.canonical_rule

function is_cut(rule::ImmersedQuadrature, element_id::Int)
    return get_cut_elements(rule)[element_id]
end

function get_element_quadrature_rule(rule::ImmersedQuadrature, element_id::Int)
    # STABILITY: This results in a `Union` type due to the tensor-product/point-set
    # disparity.
    if is_cut(rule, element_id)
        return get_cut_quadrature_rules(rule)[element_id]
    end

    return get_canonical_quadrature_rule(rule)
end
