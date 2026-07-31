############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    WeakForm{manifold_dim, LHS, RHS, I}

Structure representing the weak-formulation of a continuous Petrov-Galerking method. Both
the left and right-hand sides of the formulation are given as blocks of real-valued
operators; these are defined from a set of inputs holding the test, trial and forcing terms,
and a constructor method that defines where the blocks are placed.

# Fields

  - `lhs_expressions::LHS`: The left-hand side blocks of the weak-formulation.
  - `rhs_expressions::RHS`: The right-hand side blocks of the weak-formulation.
  - `inputs::I`: The inputs for the weak-formulation, which include the test and trial spaces,
    and forcing terms.

# Type parameters

  - `manifold_dim::Int`: The dimension of the manifold where the weak-formulation is defined.
  - `LHS`: The type of the left-hand side expressions. Each row-column entry should be a
    subtype of  `AbstractRealValuedOperator` or `0`.
  - `RHS`: The type of the right-hand side expressions. Each row-column entry should be a
    subtype of  `AbstractRealValuedOperator` or `0`.
  - `I`: The type of the inputs. It should be a subtype of `WeakFormInputs{manifold_dim}`.

# Inner constructors

  - `WeakForm(inputs::I, constructor::F)`: Creates a new `WeakForm` instance with the given
    inputs and constructor function. The constructor function is used to generate the
    left-hand side and right-hand side blocks of real-valued operators.
"""
struct WeakForm{manifold_dim, LHS, RHS, I}
    lhs_expressions::LHS
    rhs_expressions::RHS
    inputs::I
    function WeakForm(
        lhs_expressions::LHS, rhs_expressions::RHS, inputs::I
    ) where {
        manifold_dim,
        lhs_num_rows,
        lhs_num_cols,
        rhs_num_rows,
        rhs_num_cols,
        LHS <: NTuple{
            lhs_num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        RHS <: NTuple{
            rhs_num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        I <: WeakFormInputs{manifold_dim},
    }
        if lhs_num_rows != rhs_num_rows
            throw(
                ArgumentError(
                    """"The number of rows on the left-hand side must match the number \
                    of rows on the right-hand side. The left-hand side has $(lhs_num_rows) \
                    rows and the right-hand side has $(rhs_num_rows) rows."""
                ),
            )
        end

        num_test = get_num_test(inputs)
        num_trial = get_num_trial(inputs)
        if num_test != lhs_num_rows
            throw(
                ArgumentError(
                    """The number of rows on the left-hand side must match the number \
                    of test forms. The left-hand side has $(lhs_num_rows) rows and the \
                    inputs have $(num_test) test spaces."""
                ),
            )
        end

        if num_trial != lhs_num_cols
            throw(
                ArgumentError(
                    """The number of columns on the left-hand side must match the number \
                    of trial forms. The left-hand side has $(lhs_num_cols) columns and \
                    the inputs have $(num_trial) trial spaces."""
                ),
            )
        end

        # TODO: We should think of how to incorporate expressions with different manifold
        # dimensions (boundary integrals.)

        return new{manifold_dim, LHS, RHS, I}(lhs_expressions, rhs_expressions, inputs)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_lhs_expressions(wf::WeakForm) = wf.lhs_expressions
get_rhs_expressions(wf::WeakForm) = wf.rhs_expressions
get_inputs(wf::WeakForm) = wf.inputs
get_test_forms(wf::WeakForm) = get_test_forms(get_inputs(wf))
get_trial_forms(wf::WeakForm) = get_trial_forms(get_inputs(wf))
get_forcings(wf::WeakForm) = get_forcings(get_inputs(wf))
get_forcing(wf::WeakForm, id::Int=1) = get_forcing(get_inputs(wf), id)
get_test_sizes(wf::WeakForm) = Forms.get_num_basis.(get_test_forms(wf))
get_trial_sizes(wf::WeakForm) = Forms.get_num_basis.(get_trial_forms(wf))
get_test_size(wf::WeakForm) = sum(get_test_sizes(wf))
get_trial_size(wf::WeakForm) = sum(get_trial_sizes(wf))
get_lhs_size(wf::WeakForm) = get_test_size(wf), get_trial_size(wf)

"""
    get_test_offsets(
        wf::WeakForm{manifold_dim, LHS, RHS, I}
    ) where {
        manifold_dim,
        num_rows,
        lhs_num_cols,
        rhs_num_cols,
        LHS <:
        NTuple{
            num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        RHS <:
        NTuple{
            num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        I,
    }

Returns the offsets of the test function spaces used in the weak form.

# Arguments

  - `wf::WeakForm{manifold_dim, LHS, RHS, I}`: The weak form being used.

# Returns

  - `NTuple{num_rows, Int}`: The offsets of the test function spaces.
"""
function get_test_offsets(
    wf::WeakForm{manifold_dim, LHS, RHS, I}
) where {
    manifold_dim,
    num_rows,
    lhs_num_cols,
    rhs_num_cols,
    LHS <:
    NTuple{num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    RHS <:
    NTuple{num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    I,
}
    test_cumsum = [0; cumsum(get_test_sizes(wf)[1:(end - 1)])...]

    return ntuple(exp_id -> test_cumsum[exp_id], num_rows)
end

"""
    get_trial_offsets(
        wf::WeakForm{manifold_dim, LHS, RHS, I}
    ) where {
        manifold_dim,
        num_rows,
        lhs_num_cols,
        rhs_num_cols,
        LHS <:
        NTuple{
            num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        RHS <:
        NTuple{
            num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        I,
    }

Returns the offsets of the trial function spaces used in the weak form.

# Arguments

  - `wf::WeakForm{manifold_dim, LHS, RHS, I}`: The weak form being used.

# Returns

  - `NTuple{lhs_num_cols, lhs_num_cols}`: The offsets of the trial function spaces.
"""
function get_trial_offsets(
    wf::WeakForm{manifold_dim, LHS, RHS, I}
) where {
    manifold_dim,
    num_rows,
    lhs_num_cols,
    rhs_num_cols,
    LHS <:
    NTuple{num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    RHS <:
    NTuple{num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    I,
}
    trial_cumsum = [0; cumsum(get_trial_sizes(wf)[1:(end - 1)])...]

    return ntuple(exp_id -> trial_cumsum[exp_id], lhs_num_cols)
end

function get_rhs_size(wf::WeakForm)
    if isnothing(get_forcing(wf))
        return get_test_size(wf), get_trial_size(wf)
    end

    return get_test_size(wf), 1
end

"""
    get_estimated_nnz_per_elem(wf::WeakForm)

Returns the estimated number of non-zero entries per element for the left- and right-hand
sides of the weak-formulation.

# Arguments

  - `wf::WeakForm`: The weak-formulation for which the estimated number of non-zero entries is
    to be determined.

# Returns

  - `Tuple(Int, Int)`: The estimated number of non-zero entries per element for the left-hand
    side and right-hand side of the weak-formulation, respectively.
"""
function get_estimated_nnz_per_elem(wf::WeakForm)
    left_hand_nnz = 0
    for lhs_row in get_lhs_expressions(wf), expression in lhs_row
        if expression == 0
            continue
        end

        left_hand_nnz += Forms.get_estimated_nnz_per_elem(expression)
    end

    right_hand_nnz = 0
    for rhs_row in get_rhs_expressions(wf), expression in rhs_row
        if expression == 0
            continue
        end

        right_hand_nnz += Forms.get_estimated_nnz_per_elem(expression)
    end

    return (left_hand_nnz, right_hand_nnz)
end

"""
    get_num_elements(wf::WeakForm)

Returns the number of elements over which the discrete weak-formulation is defined.

# Arguments

  - `wf::WeakForm`: The weak-formulation for which the number of elements is to be determined.

# Returns

  - `::Int`: The number of elements.
"""
function get_num_elements(wf::WeakForm)
    for lhs_row in get_lhs_expressions(wf), expression in lhs_row
        if expression == 0
            continue
        end

        return Forms.get_num_elements(expression)
    end

    return throw(
        ArgumentError("No elements found in the left-hand side of the weak-formulation.")
    )
end

"""
    get_num_evaluation_elements(wf::WeakForm)

Returns the maximum number of elements over which the weak form blocks are evaluated. This
is the max over all lhs and rhs expression blocks.

# Arguments

  - `wf::WeakForm`: The weak-formulation for which the number of quadrature elements is to be
    determined.

# Returns

  - `::Int`: The number of quadrature elements.
"""
function get_num_evaluation_elements(wf::WeakForm)
    num_eval_elements = 0
    for lhs_row in get_lhs_expressions(wf), expression in lhs_row
        if expression == 0
            continue
        end

        num_eval_elements = max(
            num_eval_elements, Forms.get_num_evaluation_elements(expression)
        )
    end

    for rhs_row in get_rhs_expressions(wf), expression in rhs_row
        if expression == 0
            continue
        end

        num_eval_elements = max(
            num_eval_elements, Forms.get_num_evaluation_elements(expression)
        )
    end

    return num_eval_elements
end
