module Assemblers

import LinearAlgebra
import SparseArrays as spa

using ..Geometry
using ..Forms
using ..Quadrature
using ..Mesh
using ..FunctionSpaces
using ..Analysis

# These exports make these symbols/functions/etc. available outside the Forms module. Only
# if they are also exported in the Mantis.jl file will they be available when using Mantis.

# Abstract types
# The public keyword is only available in Julia 1.11 and up. Since we also support LTS
# (currently 1.10), we add the following line from the manual:
# https://docs.julialang.org/en/v1.12/manual/modules/#Export-lists
VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("public AbstractInputs"))

# Global Assemblers
export assemble

# Weakforms and WeakFormInputs
export WeakFormInputs,
    WeakForm,
    get_lhs_expressions,
    get_rhs_expressions,
    get_inputs,
    get_test_forms,
    get_trial_forms,
    get_forcings,
    get_forcing,
    get_test_sizes,
    get_trial_sizes,
    get_test_size,
    get_trial_size,
    get_lhs_size,
    get_rhs_size,
    get_test_offsets,
    get_trial_offsets,
    get_estimated_nnz_per_elem,
    get_num_elements,
    get_num_evaluation_elements

# Pre-defined weak formulations
export L2_projection,
    solve_L2_projection,
    zero_form_hodge_laplacian,
    solve_zero_form_hodge_laplacian,
    one_form_hodge_laplacian,
    solve_one_form_hodge_laplacian,
    n_form_hodge_laplacian,
    solve_volume_form_hodge_laplacian,
    maxwell_eigenvalue,
    analytical_maxwell_eigenfunction,
    get_analytical_maxwell_eig,
    solve_maxwell_eig

abstract type AbstractInputs end

include("WeakFormulations/WeakFormulations.jl")
include("GlobalAssemblers.jl")
include("Hierarchical/Hierarchical.jl")

end
