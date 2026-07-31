"""
    get_num_evaluation_elements(global_quad_rule::AbstractGlobalQuadratureRule)

Get the number of quadrature elements in the global quadrature rule. These are the smallest
element units on which quadrature is performed.

# Arguments

  - `global_quad_rule::AbstractGlobalQuadratureRule`: The global quadrature rule.

# Returns

  - `Int`: The number of quadrature elements in the global quadrature rule.
"""
function get_num_evaluation_elements(global_quad_rule::AbstractGlobalQuadratureRule)
    return global_quad_rule.num_quad_elements
end

"""
    get_num_base_elements(global_quad_rule::AbstractGlobalQuadratureRule)

Get the number of base elements in the global quadrature rule. Each base element can have
several sub-elements on which quadrature is performed.

# Arguments

  - `global_quad_rule::AbstractGlobalQuadratureRule`: The global quadrature rule.

# Returns

  - `Int`: The number of base elements in the global quadrature rule.
"""
function get_num_base_elements(global_quad_rule::AbstractGlobalQuadratureRule)
    return global_quad_rule.num_base_elements
end

"""
    get_element_idxs(global_quad_rule::AbstractGlobalQuadratureRule, element_idx::Int)

Get the indices of the quadrature elements in a given base element.

# Arguments

  - `global_quad_rule::AbstractGlobalQuadratureRule`: The global quadrature rule.
  - `element_idx::Int`: The index of the base element.

# Returns

  - `Vector{Int}`: The indices of the quadrature elements in the specified base element.
"""
function get_element_idxs(::AbstractGlobalQuadratureRule, ::Int)
    return throw(
        ArgumentError("Method not implemented for this type of global quadrature rule.")
    )
end

"""
    get_element_quadrature_rule(global_quad_rule::AbstractGlobalQuadratureRule, element_idx::Int)

Get the quadrature rule for a specific quadrature element index .

# Arguments

  - `global_quad_rule::AbstractGlobalQuadratureRule`: The global quadrature rule.
  - `element_idx::Int`: The index of the quadrature element.

# Returns

  - `AbstractElementQuadratureRule`: The quadrature rule for the specified quadrature element.
"""
function get_element_quadrature_rule(::AbstractGlobalQuadratureRule, ::Int)
    return throw(
        ArgumentError("Method not implemented for this type of global quadrature rule.")
    )
end

# specific implementations
include("./StandardQuadrature.jl")
