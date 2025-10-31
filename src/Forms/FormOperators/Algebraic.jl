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
    It is expected that the basis underlying each operator are compatible, and
    this will not be checked.

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
  AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

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
- `F <: AbstractFormField{manifold_dim, form_rank, expression_rank, G}`: The
    type of the original form expression .
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `UnaryFormTransformation(form::F, transformation::T, label::String)`:
    General constructor.
- `Base.:-(form::AbstractFormField)`: Alias for the additive inverse of a
differential form expression .
- `Base.:*(factor::Number, form::AbstractFormField)`: Alias for the
    multiplication of a differential form expression with a constant factor.
"""
struct UnaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
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
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
        T <: Function,
    }
        label = "(" * label * get_label(form) * ")"

        return new{manifold_dim, form_rank, expression_rank, G, F, T}(
            form, transformation, label
        )
    end

    function Base.:-(form::AbstractFormExpression)
        return UnaryFormTransformation(form, x -> -x, "-")
    end

    function Base.:*(factor::Number, form::AbstractFormExpression)
        return UnaryFormTransformation(form, x -> factor * x, "$(factor)*")
    end
end

"""
  BinaryFormTransformation{manifold_dim, form_rank, F1, F2, G, T} <:
  AbstractFormField{manifold_dim, form_rank, G}

Structure holding the necessary information to evaluate a binary, algebraic transformation
acting on two differential form fields.

!!! warning
    It is expected that the basis underlying each operator are compatible, and
    this will not be checked.

# Fields
- `form_1::F1`: The first differential form.
- `form_2::F2`: The second differential form.
- `transformation::T`: The transformation to apply to the differential forms.
- `label::String`: The label to associate to the resulting differential form.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the forms are defined.
- `form_rank::Int`: The rank of both differential forms.
- `F1 <: AbstractFormField{manifold_dim, form_rank, G}`: The type of the first form field.
- `F2 <: AbstractFormField{manifold_dim, form_rank, G}`: The type of the second form field.
- `G <: AbstractGeometry{manifold_dim}`: The type of the geometry where both forms are
  defined.
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `BinaryFormTransformation(form_1::F1, form_2::F2, transformation::T, label::String)`: General
  constructor.
- `Base.:+(form_1::F1, form_2::F2)`: Alias for the sum of two differential form fields.
- `Base.:-(form_1::F1, form_2::F2)`: Alias for the difference of two differential form
  fields.
"""
struct BinaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F1, F2, T} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
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
        F1 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
        F2 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
        T <: Function,
    }
        label = "(" * get_label(form_1) * label * get_label(form_2) * ")"

        return new{manifold_dim, form_rank, expression_rank, G, F1, F2, T}(
            form_1, form_2, transformation, label
        )
    end

    function Base.:+(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x + y, "+")
    end

    function Base.:-(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x - y, "-")
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_transformation(una_trans::UnaryOperatorTransformation) = una_trans.transformation
get_operator(una_trans::UnaryOperatorTransformation) = una_trans.operator
get_transformation(bin_trans::BinaryOperatorTransformation) = bin_trans.transformation
get_transformation(una_trans::UnaryFormTransformation) = una_trans.transformation
get_form(una_trans::UnaryFormTransformation) = una_trans.form
get_transformation(bin_trans::BinaryFormTransformation) = bin_trans.transformation
get_forms(bin_trans::BinaryFormTransformation) = bin_trans.form_1, bin_trans.form_2
get_geometry(bin_trans::BinaryFormTransformation) = get_geometry(get_forms(bin_trans)...)
get_geometry(una_trans::UnaryFormTransformation) = get_geometry(get_form(una_trans))

function get_operators(bin_trans::BinaryOperatorTransformation)
    return bin_trans.operator_1, bin_trans.operator_2
end

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

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(una_trans::UnaryOperatorTransformation, element_id::Int)
    operator = get_operator(una_trans)
    transformation = get_transformation(una_trans)
    eval, indices = evaluate(operator, element_id)

    return transformation(eval), indices
end

function evaluate(bin_trans::BinaryOperatorTransformation, element_id::Int)
    operators = get_operators(bin_trans)
    transformation = get_transformation(bin_trans)
    eval_1, indices = evaluate(operators[1], element_id)
    eval_2, _ = evaluate(operators[2], element_id)
    eval = transformation(eval_1, eval_2)

    return eval, indices
end

function evaluate(
    una_trans::UnaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    form = get_form(una_trans)
    transformation = get_transformation(una_trans)
    form_eval, indices = evaluate(form, element_id, xi)

    return transformation(form_eval), indices
end

function evaluate(
    bin_trans::BinaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    forms = get_forms(bin_trans)
    transformation = get_transformation(bin_trans)
    form_1_eval, indices = evaluate(forms[1], element_id, xi)
    form_2_eval, _ = evaluate(forms[2], element_id, xi)
    eval = transformation(form_1_eval, form_2_eval)

    return eval, indices
end
