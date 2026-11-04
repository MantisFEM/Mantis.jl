"""
    L2_norm(
        u::Forms.AbstractForm{manifold_dim},
        dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    ) where {manifold_dim}

Compute the ``L^2``-norm of the given form `u` using the given quadrature dΩ. See also
[`L2_norm_per_element`](@ref).
"""
function L2_norm(
    u::Forms.AbstractForm{manifold_dim},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
) where {manifold_dim}
    norm = 0.0
    inner_prod = ∫(u ∧ ★(u), dΩ)
    for element_id in 1:Forms.get_num_elements(u)
        norm += Forms.evaluate(inner_prod, element_id)[1][1]
    end

    return sqrt(norm)
end

"""
    L2_norm_per_element(
        u::Forms.AbstractForm{manifold_dim},
        dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    ) where {manifold_dim}

Compute the ``L^2``-norm per element of the given form `u` using the given quadrature dΩ.
See also [`L2_norm`](@ref).
"""
function L2_norm_per_element(
    u::Forms.AbstractForm{manifold_dim},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
) where {manifold_dim}
    num_elements = Quadrature.get_num_base_elements(dΩ)
    inner_prod = ∫(u ∧ ★(u), dΩ)
    result = Vector{Float64}(undef, num_elements)
    for element_id in eachindex(result)
        result[element_id] = sqrt(Forms.evaluate(inner_prod, element_id)[1][1])
    end

    return result
end

"""
    Linf_norm(
        u::Forms.AbstractForm{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
    ) where {manifold_dim}
    Linf_norm(
        u::Forms.AbstractForm{manifold_dim},
        dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    ) where {manifold_dim}

Compute the ``L^{\\infty}``-norm of the given form `u` by evaluating `u` at
`xi`, or the nodes of `dΩ`. See also [`Linf_norm_per_element`](@ref).
"""
function Linf_norm(
    u::Forms.AbstractForm{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    norm = 0.0
    for element_id in 1:Forms.get_num_elements(u)
        norm = max(norm, maximum(abs, Forms.evaluate(u, element_id, xi)[1][1]))
    end

    return norm
end

function Linf_norm(
    u::Forms.AbstractForm{manifold_dim},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
) where {manifold_dim}
    norm = 0.0
    for element_id in 1:Forms.get_num_elements(u)
        xi = Quadrature.get_nodes(Quadrature.get_element_quadrature_rule(dΩ, element_id))
        norm = max(norm, maximum(abs, Forms.evaluate(u, element_id, xi)[1][1]))
    end

    return norm
end

"""
    Linf_norm_per_element(
        u::Forms.AbstractForm{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
    ) where {manifold_dim}
    Linf_norm_per_element(
        u::Forms.AbstractForm{manifold_dim},
        dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    ) where {manifold_dim}

Compute the ``L^{\\infty}``-norm per element of the given form `u` by evaluating `u` at
`xi`, or the nodes of `dΩ`. See also [`Linf_norm`](@ref).
"""
function Linf_norm_per_element(
    u::Forms.AbstractForm{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    num_elements = Forms.get_num_elements(u)
    result = zeros(num_elements)
    for element_id in eachindex(result)
        result[element_id] = maximum(abs, Forms.evaluate(u, element_id, xi)[1][1])
    end

    return result
end

function Linf_norm_per_element(
    u::Forms.AbstractForm{manifold_dim},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
) where {manifold_dim}
    num_elements = Forms.get_num_elements(u)
    result = zeros(num_elements)
    for element_id in eachindex(result)
        xi = Quadrature.get_nodes(Quadrature.get_element_quadrature_rule(dΩ, element_id))
        result[element_id] = maximum(abs, Forms.evaluate(u, element_id, xi)[1][1])
    end

    return result
end
