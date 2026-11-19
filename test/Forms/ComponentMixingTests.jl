module ComponentMixingTests

# Tests for the evaluation of forms whose components have different (and possibly
# overlapping) basis indices. Operators that combine components must evaluate the combined
# component on the union of the indices of the combined components. The results are compared
# against a dense reference, in which every component is expanded to all the basis functions
# supported on the element.

using Mantis

using Test

# Embeds the evaluation `values` of a component, with per-expression indices `indices`, into
# a dense array whose columns per expression are `global_indices`.
function to_dense(values, indices, global_indices)
    dense = zeros(size(values, 1), length.(global_indices)...)
    positions = [indexin(indices[e], global_indices[e]) for e in eachindex(indices)]
    for basis_id in CartesianIndices(size(values)[2:end])
        dense_id = ntuple(e -> positions[e][basis_id[e]], length(indices))
        dense[:, dense_id...] .= values[:, Tuple(basis_id)...]
    end

    return dense
end

sorted_union(indices...) = sort(union(indices...))

xi_2d = Points.TensorProductPoints(([0.1, 0.5, 0.9], [0.2, 0.7]))

@testset "Polar splines (overlapping component indices)" begin
    complex = Forms.create_polar_spline_de_rham_complex(
        (4, 3), FunctionSpaces.Bernstein.((2, 2)), (1, 1)
    )
    one_form_space = complex[2]
    fe_space = Forms.get_fe_space(one_form_space)
    geometry = Forms.get_geometry(one_form_space)
    der_idx_1 = FunctionSpaces.get_derivative_idx((1, 0))
    der_idx_2 = FunctionSpaces.get_derivative_idx((0, 1))

    # Elements 1 to 4 touch the pole, where the components share basis functions.
    for element_id in (1, 3, 5, 10)
        I_1 = collect(FunctionSpaces.get_basis_indices(fe_space, element_id, 1))
        I_2 = collect(FunctionSpaces.get_basis_indices(fe_space, element_id, 2))
        I = (collect(FunctionSpaces.get_basis_indices(fe_space, element_id)),)
        I_union = sorted_union(I_1, I_2)
        if element_id <= 4
            @test !isempty(intersect(I_1, I_2))
        end

        # Dense reference of the 1-form and its first derivatives.
        fem_evals, fem_indices = Forms._evaluate_form_in_canonical_coordinates(
            one_form_space, element_id, xi_2d, 1
        )
        @test fem_indices == [[I_1], [I_2]]
        α = [to_dense(fem_evals[1][1][c], fem_indices[c], I) for c in 1:2]
        ∂₁α = [to_dense(fem_evals[2][der_idx_1][c], fem_indices[c], I) for c in 1:2]
        ∂₂α = [to_dense(fem_evals[2][der_idx_2][c], fem_indices[c], I) for c in 1:2]

        # Form space: each component only has its own basis functions.
        form_eval, form_indices = Forms.evaluate(one_form_space, element_id, xi_2d)
        @test form_indices == [[I_1], [I_2]]
        @test size(form_eval[1], 2) == length(I_1)
        @test size(form_eval[2], 2) == length(I_2)

        # Exterior derivative: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂.
        d_eval, d_indices = Forms.evaluate(Forms.d(one_form_space), element_id, xi_2d)
        @test d_indices == [[I_union]]
        @test size(d_eval[1]) == (Points.get_num_points(xi_2d), length(I_union))
        @test isapprox(to_dense(d_eval[1], d_indices[1], I), ∂₁α[2] - ∂₂α[1]; atol=1e-12)

        # Hodge star: [-(α₁g²¹+α₂g²²)dξ₁ + (α₁g¹¹+α₂g¹²)dξ₂]√det(gᵢⱼ).
        inv_g, _, sqrt_g = Geometry.inv_metric(geometry, element_id, xi_2d)
        g_ij(i, j) = [inv_g[p][i, j] for p in eachindex(inv_g)]
        ★α = [
            -(α[1] .* g_ij(2, 1) .+ α[2] .* g_ij(2, 2)) .* sqrt_g,
            (α[1] .* g_ij(1, 1) .+ α[2] .* g_ij(1, 2)) .* sqrt_g,
        ]
        hodge_eval, hodge_indices = Forms.evaluate(
            Forms.★(one_form_space), element_id, xi_2d
        )
        @test hodge_indices == [[I_union], [I_union]]
        for c in 1:2
            @test isapprox(to_dense(hodge_eval[c], hodge_indices[c], I), ★α[c]; atol=1e-12)
        end

        # Wedge: (α₁β₂ - α₂β₁)dξ₁∧dξ₂, with β = ★α.
        wedge = Forms.Wedge(one_form_space, Forms.★(one_form_space))
        wedge_eval, wedge_indices = Forms.evaluate(wedge, element_id, xi_2d)
        @test wedge_indices == [[I_union, I_union]]
        wedge_reference = zeros(Points.get_num_points(xi_2d), length(I[1]), length(I[1]))
        for a in axes(wedge_reference, 2), b in axes(wedge_reference, 3)
            wedge_reference[:, a, b] .=
                α[1][:, a] .* ★α[2][:, b] .- α[2][:, a] .* ★α[1][:, b]
        end
        @test isapprox(
            to_dense(wedge_eval[1], wedge_indices[1], (I[1], I[1])),
            wedge_reference;
            atol=1e-12,
        )

        # Binary transformations of forms with different indices per component. For 1-forms
        # in 2D we have ★★α = -α, but ★★α is evaluated on the union of the indices.
        ★★α = Forms.★(Forms.★(one_form_space))
        _, hodge_hodge_indices = Forms.evaluate(★★α, element_id, xi_2d)
        @test hodge_hodge_indices == [[I_union], [I_union]]
        sum_eval, sum_indices = Forms.evaluate(one_form_space + ★★α, element_id, xi_2d)
        diff_eval, diff_indices = Forms.evaluate(one_form_space - ★★α, element_id, xi_2d)
        @test sum_indices == [[I_union], [I_union]]
        @test diff_indices == [[I_union], [I_union]]
        for c in 1:2
            @test isapprox(sum_eval[c], zeros(size(sum_eval[c])); atol=1e-12)
            @test isapprox(
                to_dense(diff_eval[c], diff_indices[c], I), 2 .* α[c]; atol=1e-12
            )
        end
    end
end

xi_3d = Points.TensorProductPoints(([0.1, 0.6], [0.3, 0.8], [0.25, 0.75]))

# The curvilinear mapping gives a non-diagonal metric, so every component of the Hodge-star
# depends on several components of the form.
@testset "Curvilinear 3D tensor-product splines (disjoint component indices)" begin
    complex = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2), (2, 2, 2), (1, 1, 1)
    )
    der_idxs = [FunctionSpaces.get_derivative_idx(ntuple(d -> Int(d == i), 3)) for i in 1:3]

    for form_rank in (1, 2), element_id in (1, 3, 8)
        form_space = complex[form_rank + 1]
        fe_space = Forms.get_fe_space(form_space)
        geometry = Forms.get_geometry(form_space)
        I_c = [
            collect(FunctionSpaces.get_basis_indices(fe_space, element_id, c)) for c in 1:3
        ]
        I = (collect(FunctionSpaces.get_basis_indices(fe_space, element_id)),)
        I_union = sorted_union(I_c...)

        # Dense reference of the form and its first derivatives, ∂α[i][c] = ∂αc/∂ξi.
        fem_evals, fem_indices = Forms._evaluate_form_in_canonical_coordinates(
            form_space, element_id, xi_3d, 1
        )
        @test fem_indices == [[I_c[c]] for c in 1:3]
        α = [to_dense(fem_evals[1][1][c], fem_indices[c], I) for c in 1:3]
        ∂α = [
            [to_dense(fem_evals[2][der_idxs[i]][c], fem_indices[c], I) for c in 1:3] for
            i in 1:3
        ]

        # Exterior derivative.
        d_eval, d_indices = Forms.evaluate(Forms.d(form_space), element_id, xi_3d)
        if form_rank == 1
            # (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
            # + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
            dα = [∂α[2][3] - ∂α[3][2], ∂α[3][1] - ∂α[1][3], ∂α[1][2] - ∂α[2][1]]
            @test d_indices == [
                [sorted_union(I_c[2], I_c[3])],
                [sorted_union(I_c[3], I_c[1])],
                [sorted_union(I_c[1], I_c[2])],
            ]
        else
            # (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
            dα = [∂α[1][1] + ∂α[2][2] + ∂α[3][3]]
            @test d_indices == [[I_union]]
        end
        for i in eachindex(dα)
            @test isapprox(to_dense(d_eval[i], d_indices[i], I), dα[i]; atol=1e-12)
        end

        # Hodge star: component i is ∑ⱼ M[i, j] αⱼ s, with M = g⁻¹ and s = √det(gᵢⱼ) for
        # 1-forms, and M = g and s = 1/√det(gᵢⱼ) for 2-forms.
        if form_rank == 1
            M, _, s = Geometry.inv_metric(geometry, element_id, xi_3d)
        else
            M, sqrt_g = Geometry.metric(geometry, element_id, xi_3d)
            s = 1 ./ sqrt_g
        end
        M_ij = (i, j) -> [M[p][i, j] for p in eachindex(M)]
        ★α = [sum(α[j] .* M_ij(i, j) for j in 1:3) .* s for i in 1:3]
        @test !isapprox(★α[1], α[1] .* M_ij(1, 1) .* s; atol=1e-12)
        hodge_eval, hodge_indices = Forms.evaluate(Forms.★(form_space), element_id, xi_3d)
        @test hodge_indices == [[I_union] for _ in 1:3]
        for i in 1:3
            @test isapprox(to_dense(hodge_eval[i], hodge_indices[i], I), ★α[i]; atol=1e-12)
        end
    end
end

@testset "Unsupported combinations" begin
    @test_throws ArgumentError Forms._d_terms(Val(4), Val(1))
    @test_throws ArgumentError Forms._hodge_rows(Val(4), Val(1))
end

end
