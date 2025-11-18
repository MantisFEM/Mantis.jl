############################################################################################
#                                        Structures                                        #
############################################################################################

"""
  UnaryOperatorTransformation{manifold_dim, O, T} <:
  AbstractRealValuedOperator{manifold_dim}

Structure holding the necessary information to evaluate a unary, algebraic transformation
of a `AbstractRealValuedOperator`.

# Fields
- `operator::O`: The operator to which the transformation is applied.
- `transformation::T`: The transformation to apply to the operator.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the operator is defined.
- `O <: AbstractRealValuedOperator{manifold_dim}`: Type of the original real-valued
  operator.
- `T <: Function`: Function defining the algebraic transformation.

# Inner constructors
- `UnaryOperatorTransformation(operator::O, transformation::T)`: General constructor.
- `Base.:*(factor::Number, operator::AbstractRealValuedOperator)`: Alias for the
  multiplication of a real-valued operator with a constant factor.
- `Base.:-(operator::AbstractRealValuedOperator)`: Alias for the additive inverse of a
  real-valued operator.
"""
struct UnaryOperatorTransformation{manifold_dim, O, T} <:
       AbstractRealValuedOperator{manifold_dim}
    operator::O
    transformation::T

    function UnaryOperatorTransformation(
        operator::O, transformation::T
    ) where {manifold_dim, O <: AbstractRealValuedOperator{manifold_dim}, T <: Function}
        return new{manifold_dim, O, T}(operator, transformation)
    end

    function Base.:*(factor::Number, operator::AbstractRealValuedOperator)
        return UnaryOperatorTransformation(operator, x -> factor * x)
    end

    Base.:-(operator::AbstractRealValuedOperator) = -1.0 * operator
end

"""
  BinaryOperatorTransformation{manifold_dim, O1, O2, T} <:
  AbstractRealValuedOperator{manifold_dim}

Structure holding the necessary information to evaluate a binary, algebraic transformation
acting on two real-valued operators.

!!! warning
    The basis underlying each operator must compatible, this is checked. If not compatible
    an ArgumentError is thrown.

# Fields
- `operator_1::O1`: The first real-valued operator.
- `operator_2::O2`: The second real-valued operator.
- `transformation::T`: The transformation to apply to the operators.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the operators are defined.
- `O1 <: AbstractRealValuedOperator{manifold_dim}`: The type of the first operator.
- `O2 <: AbstractRealValuedOperator{manifold_dim}`: The type of the second operator.
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `BinaryOperatorTransformation(operator_1::O1, operator_2::O2, transformation::T )`:
  General constructor.
- `Base.:+(operator_1::O1, operator_1::O2)`: Alias for the sum of two operators.
- `Base.:-(operator_1::O1, operator_2::O2)`: Alias for the difference of two operators.
"""
struct BinaryOperatorTransformation{manifold_dim, O1, O2, T} <:
       AbstractRealValuedOperator{manifold_dim}
    operator_1::O1
    operator_2::O2
    transformation::T

    function BinaryOperatorTransformation(
        operator_1::O1, operator_2::O2, transformation::T
    ) where {
        manifold_dim,
        O1 <: AbstractRealValuedOperator{manifold_dim},
        O2 <: AbstractRealValuedOperator{manifold_dim},
        T <: Function,
    }
        # Check if both forms contain the same forms in their tree
        tree_form_1 = get_form_space_tree(operator_1)
        tree_form_2 = get_form_space_tree(operator_2)

        if !(tree_form_1 === tree_form_2)
            throw(
                ArgumentError(
                    "Both forms in the binary transformation must contain the same forms in their tree.",
                ),
            )
        end

        return new{manifold_dim, O1, O2, T}(operator_1, operator_2, transformation)
    end

    function Base.:+(
        operator_1::AbstractRealValuedOperator, operator_2::AbstractRealValuedOperator
    )
        return BinaryOperatorTransformation(operator_1, operator_2, (x, y) -> x + y)
    end

    function Base.:-(
        operator_1::AbstractRealValuedOperator, operator_2::AbstractRealValuedOperator
    )
        return BinaryOperatorTransformation(operator_1, operator_2, (x, y) -> x - y)
    end
end

"""
  UnaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T} <:
  AbstractForm{manifold_dim, form_rank, expression_rank, G}

Structure holding the necessary information to evaluate a unary, algebraic transformation
of a differential form expression.

# Fields
- `form::F`: The differential form expression to which the transformation is applied.
- `transformation::T`: The transformation function to apply to the form.
- `label::String`: The label to associate with the resulting transformed form.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the form is defined.
- `form_rank::Int`: The rank of the differential form.
- `expression_rank::Int`: The rank of the expression.
- `G <: AbstractGeometry{manifold_dim}`: The type of the geometry where the form is defined.
- `F <: AbstractForm{manifold_dim, form_rank, expression_rank, G}`: The
    type of the original form expression .
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `UnaryFormTransformation(form::F, transformation::T, label::String)`:
    General constructor.
- `Base.:-(form::AbstractForm)`: Alias for the additive inverse of a
differential form expression .
- `Base.:*(factor::Number, form::AbstractForm)`: Alias for the
    multiplication of a differential form expression with a constant factor.
"""
struct UnaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form::F
    transformation::T
    label::String

    function UnaryFormTransformation(
        form::F, transformation::T, label::String
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
        T <: Function,
    }
        label = "(" * label * get_label(form) * ")"

        return new{manifold_dim, form_rank, expression_rank, G, F, T}(
            form, transformation, label
        )
    end

    function Base.:-(form::AbstractForm)
        return UnaryFormTransformation(form, x -> -x, "-")
    end

    function Base.:*(factor::Number, form::AbstractForm)
        return UnaryFormTransformation(form, x -> factor * x, "$(factor)*")
    end
end

"""
  BinaryFormTransformation{manifold_dim, form_rank, F1, F2, G, T} <:
  AbstractForm{manifold_dim, form_rank, G}

Structure holding the necessary information to evaluate a binary, algebraic transformation
acting on two differential form expressions.

# Fields
- `form_1::F1`: The first differential form expression.
- `form_2::F2`: The second differential form expression.
- `transformation::T`: The transformation to apply to the differential forms.
- `label::String`: The label to associate to the resulting differential form.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the form expressions are defined.
- `form_rank::Int`: The rank of both differential form expressions.
- `expression_rank::Int`: The expression rank of both differential form expressions.
- `F1 <: AbstractForm{manifold_dim, form_rank, expression_rank, G}`: The type of the first form expression.
- `F2 <: AbstractForm{manifold_dim, form_rank, expression_rank, G}`: The type of the second form expression.
- `G <: AbstractGeometry{manifold_dim}`: The type of the geometry where both form expressions are
  defined.
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `BinaryFormTransformation(form_1::F1, form_2::F2, transformation::T, label::String)`: General
  constructor.
- `Base.:+(form_1::F1, form_2::F2)`: Alias for the sum of two differential form expressions.
- `Base.:-(form_1::F1, form_2::F2)`: Alias for the difference of two differential form expressions.
"""
struct BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, G, T} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form_1::F1
    form_2::F2
    transformation::T
    label::String

    function BinaryFormTransformation(
        form_1::F1, form_2::F2, transformation::T, label::String
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G,
        F1 <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
        F2 <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
        T <: Function,
    }
        # Check if both forms contain the same forms in their tree
        tree_form_1 = get_form_space_tree(form_1)
        tree_form_2 = get_form_space_tree(form_2)

        if !(tree_form_1 === tree_form_2)
            throw(
                ArgumentError(
                    "Both forms in the binary transformation must contain the same forms in their tree.",
                ),
            )
        end

        label = "(" * get_label(form_1) * label * get_label(form_2) * ")"

        return new{manifold_dim, form_rank, expression_rank, F1, F2, G, T}(
            form_1, form_2, transformation, label
        )
    end

    function Base.:+(form_1::AbstractForm, form_2::AbstractForm)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x + y, "+")
    end

    function Base.:-(form_1::AbstractForm, form_2::AbstractForm)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x - y, "-")
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_transformation(una_trans::UnaryOperatorTransformation) = una_trans.transformation
get_operator(una_trans::UnaryOperatorTransformation) = una_trans.operator

get_transformation(bin_trans::BinaryOperatorTransformation) = bin_trans.transformation
get_operators(bin_trans::BinaryOperatorTransformation) =
    bin_trans.operator_1, bin_trans.operator_2

get_transformation(una_form::UnaryFormTransformation) = una_form.transformation
get_form(una_form::UnaryFormTransformation) = una_form.form
get_geometry(una_trans::UnaryFormTransformation) = get_geometry(get_form(una_trans))
get_label(una_form::UnaryFormTransformation) = una_form.label

get_transformation(bin_trans::BinaryFormTransformation) = bin_trans.transformation
get_forms(bin_trans::BinaryFormTransformation) = bin_trans.form_1, bin_trans.form_2
get_geometry(bin_trans::BinaryFormTransformation) = get_geometry(get_forms(bin_trans)...)
get_label(bin_form::BinaryFormTransformation) = bin_form.label

function get_estimated_nnz_per_elem(una_trans::UnaryOperatorTransformation)
    return get_estimated_nnz_per_elem(get_operator(una_trans))
end

function get_num_evaluation_elements(una_trans::UnaryOperatorTransformation)
    return get_num_evaluation_elements(get_operator(una_trans))
end

function get_num_elements(una_trans::UnaryOperatorTransformation)
    return get_num_elements(get_operator(una_trans))
end

function get_num_elements(bin_trans::BinaryOperatorTransformation)
    return get_num_elements(get_operators(bin_trans)[1])
end

function get_estimated_nnz_per_elem(bin_trans::BinaryOperatorTransformation)
    return get_estimated_nnz_per_elem(get_operators(bin_trans)[1])
end

function get_num_evaluation_elements(bin_trans::BinaryOperatorTransformation)
    return get_num_evaluation_elements(get_operators(bin_trans)[1])
end

"""
    get_form_space_tree(uni_trans::UnaryOperatorTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the unary transformation, e.g., for
`c*((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have expression_rank > 1.
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `uni_trans::UnaryOperatorTransformation`: The unary transformation structure.

# Returns
- `Tuple(<:AbstractForm)`: The list of form spaces present in the tree of the unary transformation.
"""
function get_form_space_tree(una_trans::UnaryOperatorTransformation)
    return get_form_space_tree(una_trans.operator)
end

"""
    get_form_space_tree(bin_trans::BinaryOperatorTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the binary transformation, e.g., for
`(α ∧ β) + γ`, it returns the spaces of `α`, `β`, and `γ`, if all have exprssion_rank > 1. If `α` has expression_rank = 0,
it returns only the spaces of `β` and `γ`.

# Arguments
- `bin_trans::BinaryOperatorTransformation`: The binary transformation structure.

# Returns
- `Tuple(<:AbstractFormSpace)`: The list of FormSpace present in the tree of the binary transformation.
"""
function get_form_space_tree(bin_trans::BinaryOperatorTransformation)
    tree_form_1 = get_form_space_tree(bin_trans.operator_1)
    tree_form_2 = get_form_space_tree(bin_trans.operator_2)

    if !(tree_form_1 === tree_form_2)
        throw(
            ArgumentError(
                "Both forms in the binary transformation must contain the same forms in their tree.",
            ),
        )
    end

    # We can now safely return the tree of just one of the forms, since the trees are the same.
    return tree_form_1
end

"""
    get_form_space_tree(uni_trans::UnaryFormTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the unary transformation, e.g., for
`c*((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have expression_rank > 1.
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `uni_trans::UnaryFormTransformation`: The unary transformation structure.

# Returns
- `Tuple(<:AbstractForm)`: The list of form spaces present in the tree of the unary transformation.
"""
function get_form_space_tree(una_trans::UnaryFormTransformation)
    return get_form_space_tree(una_trans.form)
end

"""
    get_form_space_tree(bin_trans::BinaryFormTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the binary transformation, e.g., for
`(α ∧ β) + γ`, it returns the spaces of `α`, `β`, and `γ`, if all have exprssion_rank > 1. If `α` has expression_rank = 0,
it returns only the spaces of `β` and `γ`.

# Arguments
- `bin_trans::BinaryFormTransformation`: The binary transformation structure.

# Returns
- `Tuple(<:AbstractForm)`: The list of forms present in the tree of the binary transformation.
"""
function get_form_space_tree(bin_trans::BinaryFormTransformation)
    # Note that here we do not need to check if both trees are the same, since this is already
    # done in the inner constructor of the BinaryFormTransformation struct.
    tree_form_1 = get_form_space_tree(bin_trans.form_1)

    return tree_form_1
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(una_trans::UnaryOperatorTransformation, element_id::Int)
    operator = get_operator(una_trans)
    transformation = get_transformation(una_trans)
    eval, indices = evaluate(operator, element_id)
    eval .= transformation.(eval)

    return eval, indices
end

function evaluate(bin_trans::BinaryOperatorTransformation, element_id::Int)
    operators = get_operators(bin_trans)
    transformation = get_transformation(bin_trans)
    eval_1, indices = evaluate(operators[1], element_id)
    eval_2, _ = evaluate(operators[2], element_id)
    eval_1 .= transformation.(eval_1, eval_2)  # we store the output in eval_1 to avoid extra
    # allocations, the same reason for using .= and
    # transformation.()

    return eval_1, indices
end

function evaluate(
    uni_trans::UnaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    form = get_form(uni_trans)
    transformation = get_transformation(uni_trans)
    form_eval, indices = evaluate(form, element_id, xi)
    form_eval .= transformation.(form_eval)  # we store the output in form_eval to avoid extra
    # allocations, the same reason for using .= and
    # transformation.()
    return form_eval, indices
end

function evaluate(
    bin_trans::BinaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    forms = get_forms(bin_trans)
    transformation = get_transformation(bin_trans)

    form_1_eval, indices = evaluate(forms[1], element_id, xi)
    form_2_eval, indices = evaluate(forms[2], element_id, xi)
    form_1_eval .= transformation.(form_1_eval, form_2_eval)  # we store the output in form_eval_1 to avoid extra
    # allocations, the same reason for using .= and
    # transformation.()
    return form_1_eval, indices
end
