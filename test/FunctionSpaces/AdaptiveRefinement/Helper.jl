
function problem_data(rank::Int, geo::Geometry.AbstractGeometry)
    if rank != 1
        throw(ArgumentError("This problem data is only defined for 1-forms."))
    end

    function u_function(x)
        u₁ = @. sinpi(x[:, 1]) * tanh(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2))

        u₂ = @. sinpi(x[:, 2]) * tanh(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2))

        return [u₁, u₂]
    end

    function δu_function(x)
        du₁_dx = @. π *
                    cos(π * x[:, 1]) *
                    tanh(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2)) +
            200 *
                    (x[:, 1] - 0.5) *
                    sinpi(x[:, 1]) *
                    sech(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2))^2
        du₂_dy = @. π *
                    cos(π * x[:, 2]) *
                    tanh(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2)) +
            200 *
                    (x[:, 2] - 0.5) *
                    sinpi(x[:, 2]) *
                    sech(100 * ((x[:, 1] - 0.5)^2 + (x[:, 2] - 0.5)^2 - 0.3^2))^2
        return [-(du₁_dx + du₂_dy)]
    end

    function f_function(x)
        # f = [-Δu₁, -Δu₂]
        d2u₁_dx2 = @. -2 *
                      (200 * x[:, 1] - 100)^2 *
                      sinpi(x[:, 1]) *
                      sech(
                          100 * x[:, 1]^2 - 100 * x[:, 1] +
                          100 * (x[:, 2] - 1) * x[:, 2] +
                          41,
                      )^2 *
                      tanh(
                          100 * x[:, 1]^2 - 100 * x[:, 1] +
                          100 * (x[:, 2] - 1) * x[:, 2] +
                          41,
                      ) -
                      π^2 *
                      sinpi(x[:, 1]) *
                      tanh(
                          100 * x[:, 1]^2 - 100 * x[:, 1] +
                          100 * (x[:, 2] - 1) * x[:, 2] +
                          41,
                      ) +
            200 *
            sinpi(x[:, 1]) *
            sech(100 * x[:, 1]^2 - 100 * x[:, 1] + 100 * (x[:, 2] - 1) * x[:, 2] + 41)^2 +
            2 *
            π *
            (200 * x[:, 1] - 100) *
            cospi(x[:, 1]) *
            sech(100 * x[:, 1]^2 - 100 * x[:, 1] + 100 * (x[:, 2] - 1) * x[:, 2] + 41)^2
        d2u₁_dy2 = @. -200 *
            sinpi(x[:, 1]) *
            sech(100 * x[:, 2]^2 - 100 * x[:, 2] + 100 * (x[:, 1] - 1) * x[:, 1] + 41)^2 *
            (
                100 *
                (2 * x[:, 2] - 1)^2 *
                tanh(100 * x[:, 2]^2 - 100 * x[:, 2] + 100 * (x[:, 1] - 1) * x[:, 1] + 41) -
                1
            )
        d2u₂_dx2 = @. -200 *
            sinpi(x[:, 2]) *
            sech(100 * x[:, 1]^2 - 100 * x[:, 1] + 100 * (x[:, 2] - 1) * x[:, 2] + 41)^2 *
            (
                100 *
                (2 * x[:, 1] - 1)^2 *
                tanh(100 * x[:, 1]^2 - 100 * x[:, 1] + 100 * (x[:, 2] - 1) * x[:, 2] + 41) -
                1
            )
        d2u₂_dy2 = @. -2 *
                      (200 * x[:, 2] - 100)^2 *
                      sinpi(x[:, 2]) *
                      sech(
                          100 * x[:, 2]^2 - 100 * x[:, 2] +
                          100 * (x[:, 1] - 1) * x[:, 1] +
                          41,
                      )^2 *
                      tanh(
                          100 * x[:, 2]^2 - 100 * x[:, 2] +
                          100 * (x[:, 1] - 1) * x[:, 1] +
                          41,
                      ) -
                      π^2 *
                      sinpi(x[:, 2]) *
                      tanh(
                          100 * x[:, 2]^2 - 100 * x[:, 2] +
                          100 * (x[:, 1] - 1) * x[:, 1] +
                          41,
                      ) +
            200 *
            sinpi(x[:, 2]) *
            sech(100 * x[:, 2]^2 - 100 * x[:, 2] + 100 * (x[:, 1] - 1) * x[:, 1] + 41)^2 +
            2 *
            π *
            (200 * x[:, 2] - 100) *
            cospi(x[:, 2]) *
            sech(100 * x[:, 2]^2 - 100 * x[:, 2] + 100 * (x[:, 1] - 1) * x[:, 1] + 41)^2

        return [-(d2u₁_dx2 + d2u₁_dy2), -(d2u₂_dx2 + d2u₂_dy2)]
    end

    δu = Forms.AnalyticalFormField(0, δu_function, geo, "δu")
    u = Forms.AnalyticalFormField(1, u_function, geo, "u")
    f = Forms.AnalyticalFormField(1, f_function, geo, "f")

    return δu, u, f
end
