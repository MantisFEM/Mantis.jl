############################################################################################
#                                        Structures                                        #
############################################################################################

"""
    UnaryOperatorTransformation{manifold_dim, O, T} <:
    AbstractRealValuedOperator{manifold_dim}

Unary, algebraic transformation of an `AbstractRealValuedOperator`.

# Constructors
- `UnaryOperatorTransformation(operator::O, transformation::T)`: General constructor.
- `Base.:*(factor::Number, operator::AbstractRealValuedOperator)`: Point-wise
    multiplication of an operator with a number.
- `Base.:-(operator::AbstractRealValuedOperator)`: Point-wise additive inverse of an
    operator.

# Fields
- `operator::O`: The operator to which the transformation is applied.
- `transformation::T`: The transformation to apply to the operator.

# Type parameters
- `manifold_dim`: See [`AbstractRealValuedOperator`](@ref) for the details.
- `O <: AbstractRealValuedOperator{manifold_dim}`: Type of the original real-valued
  operator.
- `T <: Function`: Function defining the algebraic transformation.
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

Binary, algebraic transformation acting on two real-valued operators.

!!! warning
    The basis underlying each operator must compatible, this is checked. If not compatible
    an ArgumentError is thrown.

# Constructors
- `BinaryOperatorTransformation(operator_1::O1, operator_2::O2, transformation::T )`:
    General constructor.
- `Base.:+(operator_1::O1, operator_1::O2)`: Point-wise sum of two operators.
- `Base.:-(operator_1::O1, operator_2::O2)`: Point-wise difference of two operators.

# Fields
- `operator_1::O1`: The first real-valued operator.
- `operator_2::O2`: The second real-valued operator.
- `transformation::T`: The transformation to apply to the operators.

# Type parameters
- `manifold_dim`: See [`AbstractRealValuedOperator`](@ref) for the details.
- `O1 <: AbstractRealValuedOperator{manifold_dim}`: The type of the first operator.
- `O2 <: AbstractRealValuedOperator{manifold_dim}`: The type of the second operator.
- `T <: Function`: The type of the algebraic transformation.
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
    UnaryFormTransformation{manifold_dim, form_rank, expression_rank, F, T, L} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Unary, algebraic transformation of a differential form expression.

# Constructors
- `UnaryFormTransformation(form::F, transformation::T, label::String)`:
    General constructor.
- `Base.:-(form::AbstractForm)`: Point-wise additive inverse of a form.
- `Base.:*(factor::Number, form::AbstractForm)`: Point-wise multiplication of a form with a
    number.
- `Base.:*(form::AbstractForm, factor::Number)`: Point-wise multiplication of a form with a
    number.

# Fields
- `form::F`: The differential form expression to which the transformation is applied.
- `transformation::T`: The transformation function to apply to the form.
- `label::L`: The label to associate with the resulting transformed form.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `F <: AbstractForm{manifold_dim, form_rank, expression_rank}`: The type of the original
    form expression .
- `T <: Function`: The type of the algebraic transformation.
- `L <: AbstractString`: The type of the label.
"""
struct UnaryFormTransformation{manifold_dim, form_rank, expression_rank, F, T, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    transformation::T
    label::L

    function UnaryFormTransformation(
        form::F, transformation::T, label::AbstractString
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
        T <: Function,
    }
        label = "(" * label * get_label(form) * ")"

        return new{manifold_dim, form_rank, expression_rank, F, T, typeof(label)}(
            form, transformation, label
        )
    end

    function Base.:-(form::AbstractForm)
        return UnaryFormTransformation(form, x -> -x, "-")
    end

    function Base.:*(factor::Number, form::AbstractForm)
        return UnaryFormTransformation(form, x -> factor * x, "$(factor)*")
    end

    function Base.:*(form::AbstractForm, factor::Number)
        return factor * form
    end
end

"""
    BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, T, L} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Binary, algebraic transformation acting on two differential form expressions.

!!! warning "Compatibility of forms"
    When using these binary operations, you have to ensure that the operation makes sense
    for the given input. This is **not** checked!

# Constructors
- `BinaryFormTransformation(form_1::F1, form_2::F2, transformation::T, label::AbstractString)`: General
    constructor.
- `Base.:+(form_1::AbstractForm, form_2::AbstractForm)`: Point-wise sum of two forms.
- `Base.:-(form_1::AbstractForm, form_2::AbstractForm)`: Point-wise difference of two froms.
- `Base.:*(form_1::AbstractForm, form_2::AbstractForm)`: Point-wise product of two forms.

# Examples
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form 1");  # 0-form space with B as basis.

julia> sum_example = Λ⁰ₕ + Λ⁰ₕ;

julia> isa(sum_example, Forms.BinaryFormTransformation{2, 0, 1})
true

```

# Fields
- `form_1::F1`: The first differential form expression.
- `form_2::F2`: The second differential form expression.
- `transformation::T`: The transformation to apply to the differential forms.
- `label::L`: The label to associate to the resulting differential form.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `F1 <: AbstractForm{manifold_dim, form_rank, expression_rank}`: The type of the first
    form expression.
- `F2 <: AbstractForm{manifold_dim, form_rank, expression_rank}`: The type of the second
    form expression.
- `T <: Function`: The type of the algebraic transformation.
- `L <: AbstractString`: The type of the label.
"""
struct BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, T, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form_1::F1
    form_2::F2
    transformation::T
    label::L

    function BinaryFormTransformation(
        form_1::F1, form_2::F2, transformation::T, label::AbstractString
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F1 <: AbstractForm{manifold_dim, form_rank, expression_rank},
        F2 <: AbstractForm{manifold_dim, form_rank, expression_rank},
        T <: Function,
    }
        check_geometry(form_1, form_2)
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

        new_label = convert(
            typeof(label), "(" * get_label(form_1) * label * get_label(form_2) * ")"
        )

        return new{manifold_dim, form_rank, expression_rank, F1, F2, T, typeof(new_label)}(
            form_1, form_2, transformation, new_label
        )
    end

    function Base.:+(form_1::AbstractForm, form_2::AbstractForm)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x + y, "+")
    end

    function Base.:-(form_1::AbstractForm, form_2::AbstractForm)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x - y, "-")
    end

    function Base.:*(form_1::AbstractForm, form_2::AbstractForm)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x * y, "*")
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_transformation(una_trans::UnaryOperatorTransformation) = una_trans.transformation
get_operator(una_trans::UnaryOperatorTransformation) = una_trans.operator

get_transformation(bin_trans::BinaryOperatorTransformation) = bin_trans.transformation

function get_operators(bin_trans::BinaryOperatorTransformation)
    return bin_trans.operator_1, bin_trans.operator_2
end

get_transformation(una_form::UnaryFormTransformation) = una_form.transformation
get_geometry(una_trans::UnaryFormTransformation) = get_geometry(get_form(una_trans))
get_label(una_form::UnaryFormTransformation) = una_form.label

get_transformation(bin_trans::BinaryFormTransformation) = bin_trans.transformation
get_label(bin_form::BinaryFormTransformation) = bin_form.label

"""
    get_forms(bin_trans::BinaryFormTransformation)

Return both forms to which the binary form transformation is applied.
"""
get_forms(bin_trans::BinaryFormTransformation) = bin_trans.form_1, bin_trans.form_2

function get_geometry(bin_trans::BinaryFormTransformation)
    return get_geometry(first(get_forms(bin_trans)))
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

function get_form_space_tree(una_trans::UnaryOperatorTransformation)
    return get_form_space_tree(una_trans.operator)
end

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

function get_form_space_tree(una_trans::UnaryFormTransformation)
    return get_form_space_tree(una_trans.form)
end

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
    for i in eachindex(eval)
        eval[i] = transformation(eval[i])
    end

    return eval, indices
end

function evaluate(bin_trans::BinaryOperatorTransformation, element_id::Int)
    operator_1, operator_2 = get_operators(bin_trans)
    transformation = get_transformation(bin_trans)
    eval_1, indices = evaluate(operator_1, element_id)
    eval_2, _ = evaluate(operator_2, element_id)
    for i in eachindex(eval_1, eval_2)
        eval_1[i] = transformation(eval_1[i], eval_2[i])
    end

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
    for i in eachindex(form_eval)
        form_eval[i] = transformation(form_eval[i])
    end

    return form_eval, indices
end

function evaluate(
    bin_trans::BinaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    form_1, form_2 = get_forms(bin_trans)
    transformation = get_transformation(bin_trans)

    form_1_eval, indices = evaluate(form_1, element_id, xi)
    form_2_eval, _ = evaluate(form_2, element_id, xi)
    for c in eachindex(form_1_eval, form_2_eval)
        for i in eachindex(form_1_eval[c], form_2_eval[c])
            form_1_eval[c][i] = transformation(form_1_eval[c][i], form_2_eval[c][i])
        end
    end

    return form_1_eval, indices
end
