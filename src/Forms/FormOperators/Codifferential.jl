############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    CoDifferential{manifold_dim, form_rank, expression_rank, F, L} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Represents the codifferential of an `AbstractForm`.

The codifferential, often denoted ``d^{\\star}`` or ``\\delta``, is a differential operator
mapping ``k``-forms to ``k-1``-forms. On manifolds without boundaries, it is the
``L^2``-adjoint of the exterior derivative. That is,
``(\\alpha^{k-1}, \\delta\\beta^k) = (d\\alpha^{k-1}, \\beta^k)``, where
``(\\cdot, \\cdot)`` is an ``L^2`` inner-product.

The codifferential can (formally) be defined in terms of the [ExteriorDerivative](@ref) and
[Hodge](@ref). We then have (with ``n`` the manifold dimension and ``k`` the form rank)
```math
\\delta = (-1)^{n(k+1)+1} \\star d \\star\\;.
```
The codifferential is often used to defined the Hodge-Laplacian as
``\\delta d + d \\delta``. Applied to zero-forms, the Hodge-Laplacian reduces to
``\\delta d``. Note that this gives the opposite sign compared to the standard Laplacian in
Cartesian coordinates.

The `manifold_dim` of the `CoDifferential` is inherited from the input form, while the
`form_rank` is the form rank of the input form minus 1.

# Constructors
- `CoDifferential(form::F)`: General constructor.

# Aliases
- `δ`: See [`δ`](@ref) (unicode character command `\\delta`).
- `dstar`: See [`dstar`](@ref).

# Examples
```jldoctest
julia> using Mantis

julia> Bx = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 1), (1, 0));

julia> By = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (1, 2), (0, 1));

julia> B = FunctionSpaces.DirectSumSpace((Bx, By)); # A FunctionSpace with 2 components.

julia> Λ¹ₕ = Forms.FormSpace(1, B, "1-form");  # 1-form space with B as basis.

julia> δΛ¹ₕ = δ(Λ¹ₕ);

julia> isa(δΛ¹ₕ, Forms.CoDifferential{2, 0, 1})
true

```

# Fields
- `form::F`: The form to which the codifferential is applied.
- `label::L`: The codifferential label. Adds "δ" to the label of `form`.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [AbstractForm](@ref) for the details.
- `F <: Forms.AbstractForm{manifold_dim, form_rank+1, expression_rank}`: The type of `form`.
- `L <: AbstractString`: The type of the label. Since a "δ" is added to the label, this
    type may differ from the label type of the underlying form.
"""
struct CoDifferential{manifold_dim, form_rank, expression_rank, F, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    label::L

    function CoDifferential(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
    }
        if form_rank == 0
            return throw(
                ArgumentError(
                    LazyString(
                        "Tried to compute the codifferential of a 0-form in ",
                        manifold_dim,
                        "D. The codifferential of a 0-form is not defined.",
                    ),
                ),
            )
        elseif form_rank > 1
            return throw(MethodError(CoDifferential, form))
        elseif manifold_dim > 2
            return throw(MethodError(CoDifferential, form))
        end

        old_label = get_label(form)
        new_label = convert(
            typeof(old_label), LaTeXStrings.L"\delta" * "(" * old_label * ")"
        )

        return new{manifold_dim, form_rank - 1, expression_rank, F, typeof(new_label)}(
            form, new_label
        )
    end
end

"""
    dstar

Symbolic wrapper for the codifferential. See [`CoDifferential`](@ref) for the details.
"""
const dstar = CoDifferential

"""
    δ

Symbolic wrapper for the codifferential. The unicode character command is `\\delta`. See
[`CoDifferential`](@ref) for the details.
"""
const δ = CoDifferential

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::CoDifferential{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return _evaluate_codifferential(get_form(form), element_id, xi)
end

function _evaluate_codifferential(
    form::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return throw(MethodError(_evaluate_codifferential, (form, element_id, xi)))
end

############################################################################################
#                                        Form Field                                        #
############################################################################################

function _evaluate_codifferential(
    form::FormField{manifold_dim, form_rank, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
    evals, indices = _evaluate_codifferential(get_form(form), element_id, xi)

    T = eltype(evals[1])
    coefficients = get_coefficients(form)
    n_derivative_components = size(evals, 1) # == binomial(manifold_dim, form_rank + 1).

    d_form_eval = Vector{Vector{T}}(undef, n_derivative_components)
    for component_id in eachindex(d_form_eval)
        d_form_eval[component_id] = evals[component_id] * coefficients[indices[1]]
    end

    return d_form_eval, [[1]]
end

function _evaluate_codifferential(
    form::ExteriorDerivative{manifold_dim, 1, 0, FF},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, FF <: FormField{manifold_dim, 0}}
    field = get_form(form)
    space = get_form(field)
    evals, indices = _evaluate_codifferential(d(space), element_id, xi)

    T = eltype(evals[1])
    coefficients = get_coefficients(field)
    n_derivative_components = size(evals, 1) # == binomial(manifold_dim, form_rank + 1).

    d_form_eval = Vector{Vector{T}}(undef, n_derivative_components)
    for component_id in eachindex(d_form_eval)
        d_form_eval[component_id] = evals[component_id] * coefficients[indices[1]]
    end

    return d_form_eval, [[1]]
end

############################################################################################
#                                        Form Space                                        #
############################################################################################

# 1D 1-forms.
function _evaluate_codifferential(
    form_space::FormSpace{1, 1}, element_id::Int, xi::Points.AbstractPoints{1}
)
    # Evaluate derivatives of the basis functions. We need derivatives up to order 1.
    fem_evals, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )
    T = eltype(fem_evals[1][1][1][1])
    # n_coderivative_form_components = 1
    n_basis_functions = length(form_basis_indices[1])
    n_evaluation_points = Points.get_num_points(xi)
    # Preallocate memory for output array
    codiff_eval = [zeros(T, n_evaluation_points, n_basis_functions)]
    # Compute the metric terms, including derivative of the metric.
    J, inv_g, g, sqrt_g, (dgdu,), (dinv_g_du,), (dsqrt_g_du,), Hs = Geometry.metric_derivatives(
        get_geometry(form_space), element_id, xi
    )
    # Compute the codifferential.
    # α^1 = α¹ du
    # *d*α¹ = (1/sqrt(det(g))) * (α¹ d/dx(1/sqrt(det(g))) + (1/sqrt(det(g)))d/dx(α¹))
    # Note that d/dx(1/sqrt(det(g))) = - (d/dx(sqrt(g)))/(sqrt(g)^2)
    # sign: (-1)^{n(k+1)+1}, n = manifold_dim, k = form_rank. 1-forms -> always -1.
    sign = -one(T)
    idx_du = FunctionSpaces.get_derivative_idx((1,))
    for b in axes(codiff_eval[1], 2)
        for i in axes(codiff_eval[1], 1)
            inv_sqrtg = one(T) / sqrt_g[i]
            codiff_eval[1][i, b] =
                sign *
                inv_sqrtg *
                (
                    -fem_evals[1][1][1][i, b] * dsqrt_g_du[i] / (sqrt_g[i] ^ 2) +  # - α¹ * (d/dx(sqrt(g)))/(sqrt(g)^2)
                    fem_evals[2][idx_du][1][i, b] * inv_sqrtg  # ∂u α¹ * 1 / sqrt(g)
                )
        end
    end

    return codiff_eval, form_basis_indices
end

# 1D 1-forms where the 1-form is the exterior derivative of a 0-form (Laplacian).
function _evaluate_codifferential(
    form_space::F, element_id::Int, xi::Points.AbstractPoints{1}
) where {FS <: FormSpace{1, 0}, F <: ExteriorDerivative{1, 1, 1, FS}}
    # Evaluate derivatives of the basis functions. We need derivatives up to order 2. Since
    # we are evaluating the laplacian of 0-forms, we do not have to scale the derivatives.
    fem_evals, form_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(form_space), element_id, xi, 2
    )
    T = eltype(fem_evals[1][1][1][1])
    n_basis_functions = length(form_basis_indices)
    n_evaluation_points = Points.get_num_points(xi)
    # Preallocate memory for output array
    codiff_eval = [zeros(T, n_evaluation_points, n_basis_functions)]
    # Compute the metric terms, including derivative of the metric.
    J, inv_g, g, sqrt_g, (dgdu,), (dinv_g_du,), (dsqrt_g_du,), Hs = Geometry.metric_derivatives(
        get_geometry(form_space), element_id, xi
    )
    # Compute the laplacian.
    # α^1 = α¹ du
    # α¹ = d(β⁰) = ∂ᵤ β⁰ du
    # *d*α¹ = (1/sqrt(det(g))) * (α¹ d/dx(1/sqrt(det(g))) + (1/sqrt(det(g)))d/dx(α¹))
    # Note that d/dx(1/sqrt(det(g))) = - (d/dx(sqrt(g)))/(sqrt(g)^2)
    # sign: (-1)^{n(k+1)+1}, n = manifold_dim, k = form_rank. 1-forms -> always -1.
    sign = -one(T)
    idx_du = FunctionSpaces.get_derivative_idx((1,))
    idx_duu = FunctionSpaces.get_derivative_idx((2,))
    for b in axes(codiff_eval[1], 2)
        for i in axes(codiff_eval[1], 1)
            inv_sqrtg = one(T) / sqrt_g[i]
            codiff_eval[1][i, b] =
                sign *
                inv_sqrtg *
                (
                    -fem_evals[2][idx_du][1][i, b] * dsqrt_g_du[i] / (sqrt_g[i] ^ 2) +  # - α¹ * (d/dx(sqrt(g)))/(sqrt(g)^2)
                    fem_evals[3][idx_duu][1][i, b] * inv_sqrtg  # ∂u α¹ * 1 / sqrt(g)
                )
        end
    end

    return codiff_eval, [form_basis_indices]
end

# 2D 1-forms.
function _evaluate_codifferential(
    form_space::FormSpace{2, 1}, element_id::Int, xi::Points.AbstractPoints{2}
)
    # Evaluate derivatives of the basis functions. We need derivatives up to order 1.
    fem_evals, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )
    T = eltype(fem_evals[1][1][1][1])
    n_basis_functions = length(form_basis_indices[1])
    n_evaluation_points = Points.get_num_points(xi)
    # Preallocate memory for output array
    codiff_eval = [zeros(T, n_evaluation_points, n_basis_functions)]
    # Compute the metric terms, including derivative of the metric.
    J, inv_g, g, sqrt_g, (dgdu, dgdv), (dinv_g_du, dinv_g_dv), (dsqrt_g_du, dsqrt_g_dv), Hs = Geometry.metric_derivatives(
        get_geometry(form_space), element_id, xi
    )
    # Compute the coderivative.
    # α^1 = α¹ du + α² dv
    # d*α¹ = β⁰
    # sign: (-1)^{n(k+1)+1}, n = manifold_dim, k = form_rank. 1-forms -> always -1.
    sign = -one(T)
    idx_du = FunctionSpaces.get_derivative_idx((1, 0))
    idx_dv = FunctionSpaces.get_derivative_idx((0, 1))
    for b in axes(codiff_eval[1], 2)
        for i in axes(codiff_eval[1], 1)
            one_div_sqrtg = one(T) / sqrt_g[i]
            codiff_eval[1][i, b] =
                sign * (
                    fem_evals[2][idx_du][1][i, b] * inv_g[i][1, 1] +  # ∂u α¹ * g¹¹
                    fem_evals[1][1][1][i, b] * dinv_g_du[i][1, 1] +  # α¹ * ∂u g¹¹
                    fem_evals[2][idx_du][2][i, b] * inv_g[i][1, 2] +  # ∂u α² * g¹²
                    fem_evals[1][1][2][i, b] * dinv_g_du[i][1, 2] +  # α² * ∂u g¹²
                    fem_evals[2][idx_dv][1][i, b] * inv_g[i][2, 1] +  # ∂v α¹ * g²¹
                    fem_evals[1][1][1][i, b] * dinv_g_dv[i][2, 1] +  # α¹ * ∂v g²¹
                    fem_evals[2][idx_dv][2][i, b] * inv_g[i][2, 2] +  # ∂v α² * g²²
                    fem_evals[1][1][2][i, b] * dinv_g_dv[i][2, 2] +  # α² * ∂v g²²
                    one_div_sqrtg * (
                        fem_evals[1][1][1][i, b] * inv_g[i][1, 1] * dsqrt_g_du[i] +  # α¹ * g¹¹ * ∂u sqrt(g)
                        fem_evals[1][1][2][i, b] * inv_g[i][1, 2] * dsqrt_g_du[i] +  # α² * g¹² * ∂u sqrt(g)
                        fem_evals[1][1][1][i, b] * inv_g[i][2, 1] * dsqrt_g_dv[i] +  # α¹ * g²¹ * ∂v sqrt(g)
                        fem_evals[1][1][2][i, b] * inv_g[i][2, 2] * dsqrt_g_dv[i]  # α² * g²² * ∂v sqrt(g)
                    )
                )
        end
    end

    return codiff_eval, form_basis_indices
end

# Specialised version for the exterior derivative of 0-forms to 1-forms in 2D.
# This is equivalent to the Laplacian of 0-forms.
function _evaluate_codifferential(
    form_space::ExteriorDerivative{2, 1, 1, FS},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {FS <: FormSpace{2, 0}}
    # Evaluate derivatives of the basis functions. We need derivatives up to order 2. Since
    # we are evaluating the laplacian of 0-forms, the basis functions we do not have to
    # scale the derivatives.
    fem_evals, form_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(form_space), element_id, xi, 2
    )
    T = eltype(fem_evals[1][1][1][1])
    n_basis_functions = length(form_basis_indices)
    n_evaluation_points = Points.get_num_points(xi)
    # Preallocate memory for output array
    codiff_eval = [zeros(T, n_evaluation_points, n_basis_functions)]
    # Compute the metric terms, including derivative of the metric.
    J, inv_g, g, sqrt_g, (dgdu, dgdv), (dinv_g_du, dinv_g_dv), (dsqrt_g_du, dsqrt_g_dv), Hs = Geometry.metric_derivatives(
        get_geometry(form_space), element_id, xi
    )
    # Compute the laplacian, which is the coderivative of the exterior derivative.
    # α¹ = α¹ du + α² dv
    # α¹ = d(β⁰) = ∂ᵤ β⁰ du + ∂ᵥ β⁰ dv
    # sign: (-1)^{n(k+1)+1}, n = manifold_dim, k = form_rank. 1-forms -> always -1.
    sign = -one(T)
    idx_du = FunctionSpaces.get_derivative_idx((1, 0))
    idx_dv = FunctionSpaces.get_derivative_idx((0, 1))
    idx_duu = FunctionSpaces.get_derivative_idx((2, 0))
    idx_dvv = FunctionSpaces.get_derivative_idx((0, 2))
    idx_duv = FunctionSpaces.get_derivative_idx((1, 1))
    for b in axes(codiff_eval[1], 2)
        for i in 1:n_evaluation_points
            one_div_sqrtg = one(T) / sqrt_g[i]
            codiff_eval[1][i, b] =
                sign * (
                    fem_evals[3][idx_duu][1][i, b] * inv_g[i][1, 1] +  # ∂u α¹ * g¹¹
                    fem_evals[2][idx_du][1][i, b] * dinv_g_du[i][1, 1] +  # α¹ * ∂u g¹¹
                    fem_evals[3][idx_duv][1][i, b] * inv_g[i][1, 2] +  # ∂u α² * g¹²
                    fem_evals[2][idx_dv][1][i, b] * dinv_g_du[i][1, 2] +  # α² * ∂u g¹²
                    fem_evals[3][idx_duv][1][i, b] * inv_g[i][2, 1] +  # ∂v α¹ * g²¹
                    fem_evals[2][idx_du][1][i, b] * dinv_g_dv[i][2, 1] +  # α¹ * ∂v g²¹
                    fem_evals[3][idx_dvv][1][i, b] * inv_g[i][2, 2] +  # ∂v α² * g²²
                    fem_evals[2][idx_dv][1][i, b] * dinv_g_dv[i][2, 2] +  # α² * ∂v g²²
                    one_div_sqrtg * (
                        fem_evals[2][idx_du][1][i, b] * inv_g[i][1, 1] * dsqrt_g_du[i] +  # α¹ * g¹¹ * ∂u sqrt(g)
                        fem_evals[2][idx_dv][1][i, b] * inv_g[i][1, 2] * dsqrt_g_du[i] +  # α² * g¹² * ∂u sqrt(g)
                        fem_evals[2][idx_du][1][i, b] * inv_g[i][2, 1] * dsqrt_g_dv[i] +  # α¹ * g²¹ * ∂v sqrt(g)
                        fem_evals[2][idx_dv][1][i, b] * inv_g[i][2, 2] * dsqrt_g_dv[i]  # α² * g²² * ∂v sqrt(g)
                    )
                )
        end
    end

    return codiff_eval, [form_basis_indices]
end
