"""
    WeakFormInputs{manifold_dim, TeF, TrF, F} <: AbstractInputs

Container for test and trial spaces, and forcing terms to be used in a [`WeakForm`](@ref).

# Constructors
- `WeakFormInputs(test_forms::TeF, trial_forms::TrF, forcings::F)`: Creates a new
    `WeakFormInputs` instance with the given test forms, trial forms, and forcing terms.
- `WeakFormInputs(test_forms::TeF, trial_forms::TrF)`: Creates a new `WeakFormInputs`
    instance with the given test forms and trial forms. The forcing terms are set to
    nothing.
- `WeakFormInputs(test_forms::TeF, trial_forms::TrF, forcing::F)`: Creates a new
    `WeakFormInputs` instance from a single test and trial space and a forcing term.
- `WeakFormInputs(test_forms::TeF, trial_forms::TrF)`: Creates a new `WeakFormInputs`
    instance from a single test and trial space. The forcing terms are set to nothing.
- `WeakFormInputs(forms::TrF, forcing::F)`: Creates a new `WeakFormInputs` instance with the
    given trial spaces and forcing terms. The test spaces are set to the same as the trial
    spaces.
- `WeakFormInputs(forms::TrF)`: Creates a new `WeakFormInputs` instance with the given trial
    spaces. The test spaces are set to the same as the trial spaces and the forcing terms
    are set to nothing.
- `WeakFormInputs(forms::TrF, forcing::F)`: Creates a new `WeakFormInputs` instance with a
    single trial space and forcing term. The test space is set to the same as the trial
    space.
- `WeakFormInputs(forms::TrF)`: Creates a new `WeakFormInputs` instance with a single trial
    space. The test space is set to the same as the trial space and the forcing term is set
    to nothing.

# Example
```jldoctest
julia> using Mantis

julia> V0, V1, V2, V3 = Forms.create_tensor_product_bspline_de_rham_complex((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 4, 4), (3, 3, 3), (2, 2, 2));

julia> geometry = Forms.get_geometry(V2);

julia> forcing_function(x) = [ones(size(x))];

julia> f = Forms.AnalyticalFormField(3, forcing_function, geometry, "f");

julia> weak_form_inputs = Assemblers.WeakFormInputs((V2, V3), (f,));

julia> isa(weak_form_inputs, Assemblers.WeakFormInputs{3})
true

```

# Fields
- `test_forms::TeF`: The test forms for the weak-formulation.
- `trial_forms::TrF`: The trial forms for the weak-formulation.
- `forcings::F`: The forcing terms for the weak-formulation, possibly nothing.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the weak-formulation is defined.
- `TeF`: The type of the tuple of test forms. Each entry should be a subtype of
    [`Forms.AbstractFormSpace`](@ref).
- `TrF`: The type of the tuple of trial forms. Each entry should be a subtype of
    [`Forms.AbstractFormSpace`](@ref).
- `F`: The type of the tuple of forcing terms. Each entry should be a subtype of
    [`Forms.AbstractFormField`](@ref).
"""
struct WeakFormInputs{manifold_dim, TeF, TrF, F} <: AbstractInputs
    test_forms::TeF
    trial_forms::TrF
    forcings::F

    function WeakFormInputs(
        test_forms::TeF, trial_forms::TrF, forcings::F
    ) where {
        manifold_dim,
        num_TeF,
        num_TrF,
        num_F,
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return new{manifold_dim, TeF, TrF, F}(test_forms, trial_forms, forcings)
    end

    function WeakFormInputs(
        test_forms::TeF, trial_forms::TrF
    ) where {
        manifold_dim,
        num_TeF,
        num_TrF,
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
    }
        return new{manifold_dim, TeF, TrF, Tuple{Nothing}}(
            test_forms, trial_forms, (nothing,)
        )
    end

    function WeakFormInputs(
        test_forms::TeF, trial_forms::TrF, forcing::F
    ) where {
        manifold_dim,
        TeF <: Forms.AbstractFormSpace{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim},
    }
        return WeakFormInputs((test_forms,), (trial_forms,), (forcing,))
    end

    function WeakFormInputs(
        test_forms::TeF, trial_forms::TrF
    ) where {
        manifold_dim,
        TeF <: Forms.AbstractFormSpace{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
    }
        return WeakFormInputs((test_forms,), (trial_forms,))
    end

    function WeakFormInputs(
        forms::TrF, forcing::F
    ) where {
        manifold_dim,
        num_TrF,
        num_F,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return WeakFormInputs(forms, forms, forcing)
    end

    function WeakFormInputs(
        forms::TrF
    ) where {
        manifold_dim, num_TrF, TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}}
    }
        return WeakFormInputs(forms, forms)
    end

    function WeakFormInputs(
        forms::TrF, forcing::F
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim},
    }
        return WeakFormInputs((forms,), (forms,), (forcing,))
    end

    function WeakFormInputs(
        forms::TrF
    ) where {manifold_dim, TrF <: Forms.AbstractFormSpace{manifold_dim}}
        return WeakFormInputs(forms, forms)
    end
end

"""
    get_trial_forms(wfi::WeakFormInputs)

Return a tuple of [`Forms.AbstractFormSpace`](@ref) that are used as trial spaces.
"""
get_trial_forms(wfi::WeakFormInputs) = wfi.trial_forms

"""
    get_test_forms(wfi::WeakFormInputs)

Return a tuple of [`Forms.AbstractFormSpace`](@ref) that are used as test spaces.
"""
get_test_forms(wfi::WeakFormInputs) = wfi.test_forms

"""
    get_forcings(wfi::WeakFormInputs)

Return a tuple of [`Forms.AbstractFormField`](@ref) that are used as test spaces.
"""
get_forcings(wfi::WeakFormInputs) = wfi.forcings

"""
    get_trial_form(wfi::WeakFormInputs, i::Int=1)

Return the `i`-th trial [`Forms.AbstractFormSpace`](@ref).
"""
get_trial_form(wfi::WeakFormInputs, i::Int=1) = get_trial_forms(wfi)[i]

"""
    get_test_form(wfi::WeakFormInputs, i::Int=1)

Return the `i`-th test [`Forms.AbstractFormSpace`](@ref).
"""
get_test_form(wfi::WeakFormInputs, i::Int=1) = get_test_forms(wfi)[i]

"""
    get_forcing(wfi::WeakFormInputs, i::Int=1)

Return the `i`-th forcing [`Forms.AbstractFormField`](@ref).
"""
get_forcing(wfi::WeakFormInputs, i::Int=1) = get_forcings(wfi)[i]

"""
    get_num_trial(wfi::WeakFormInputs)

Return the number of trial spaces.
"""
get_num_trial(wfi::WeakFormInputs) = length(get_trial_forms(wfi))

"""
    get_num_test(wfi::WeakFormInputs)

Return the number of test spaces.
"""
get_num_test(wfi::WeakFormInputs) = length(get_test_forms(wfi))

"""
    get_num_forcings(wfi::WeakFormInputs)

Return the number of forcing functions.
"""
get_num_forcings(wfi::WeakFormInputs) = length(get_forcings(wfi))
