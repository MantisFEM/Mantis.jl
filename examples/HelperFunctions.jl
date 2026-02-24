using LinearAlgebra
using Printf

############################################################################################
#                                        Variables                                         #
############################################################################################
fpi = Float64(π)

############################################################################################
#                                     General methods                                      #
############################################################################################

function unit_square_biquadratic(num_ref::Int=0)
    deg = (2, 2)
    TP = tensor_product_bsplines(deg, (1, 1), (-1, -1), (1.0, 1.0))
    geom_coeffs = [
        0.0 0.0
        0.5 0.0
        1.0 0.0
        0.0 0.5
        0.95 0.95
        1.0 0.5
        0.0 1.0
        0.5 1.0
        1.0 1.0
    ]
    if num_ref == 0
        return Mantis.Geometry.FEGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(
            TP, (2^num_ref, 2^num_ref)
        )
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEGeometry(TP_refined, geom_coeffs_refined)
    end
end

function bilinear_wedge(num_ref::Int=0)
    deg = (1, 1)
    TP = tensor_product_bsplines(deg, (1, 1), (-1, -1), (1.0, 1.0))
    geom_coeffs = [
        0.0 0.0
        1.0 0.0
        0.0 1.0
        1.5 1.75
    ]
    if num_ref == 0
        return Mantis.Geometry.FEGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(
            TP, (2^num_ref, 2^num_ref)
        )
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEGeometry(TP_refined, geom_coeffs_refined)
    end
end

function quarter_annulus_nurbs(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    num_el = 1
    # B-spline space - angular direction
    Bθ = univariate_bsplines(deg, num_el, -1)
    # NURBS space - angular direction
    Rθ = Mantis.FunctionSpaces.RationalFESpace(Bθ, [1, 1 / sqrt(2), 1])
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, num_el, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Rθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEGeometry(TP, geom_coeffs)
end

function quarter_annulus_trigonometric_bspline(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    Wt = pi / 2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(
        Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt), num_el, -1
    )
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEGeometry(TP, geom_coeffs)
end

function quarter_bulging_annulus_trigonometric_bspline(
    radius_inner::Float64, radius_outer::Float64
)
    deg = 2
    Wt = pi / 2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(
        Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt), num_el, -1
    )
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEGeometry(TP, geom_coeffs)
end

############################################################################################
#                                     Hodge Laplacian                                      #
############################################################################################

function sinusoidal_data(
    form_rank::Int, geometry::Mantis.Geometry.AbstractGeometry, ω::Number=2 * pi
)
    function my_sol(x::Matrix{Float64})
        # [u] = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = @. sin(ω * x)
        return [vec(prod(y; dims=2))]
    end

    function grad_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = [ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        y = sin.(ω .* x)
        z = ω .* cos.(ω .* x)
        w = Vector{Vector{Float64}}(undef, size(x, 2))
        for i in 1:size(x, 2)
            w[i] = z[:, i] .* prod(y[:, setdiff(1:size(x, 2), i)]; dims=2)[:, 1]
        end
        return w
    end

    function flux_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = ⋆[ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        w = grad_my_sol(x)
        if size(x, 2) == 1
            # (a) -> (a)
            return [w[1]]

        elseif size(x, 2) == 2
            # (a, b) -> (-b, a)
            w = [-w[2], w[1]]
            return w

        elseif size(x, 2) == 3
            # (a, b, c) -> (a, b, c)
            return [w[1], w[2], w[3]]

        else
            throw(ArgumentError("Dimension not supported."))
        end
    end

    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x); dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end

    function vec_sol(x::Matrix{Float64})
        # [u¹,u²,…,uⁿ] = [sin(ωx¹), sin(ωx²),…,sin(ωxⁿ)]

        return [sin.(ω * x[:, i]) for i in 1:size(x, 2)]
    end

    function minus_div_sol(x::Matrix{Float64})
        # [-(u¹₁+u²₂+…+uⁿₙ)] = [-ω(cos(ωx¹)+cos(ωx²)+cos(ωxⁿ))]

        return [-ω .* sum(cos, ω .* x; dims=2)]
    end

    function vec_laplace_sol(x::Matrix{Float64})
        # [-Δu¹,-Δu²,…,-Δuⁿ] = [ω²sin(ωx¹),ω²sin(ωx²),…,ω²sin(ωxⁿ)]

        return [@. ω^2 * sin(ω * x[:, i]) for i in 1:size(x, 2)]
    end

    if form_rank == 0
        u⁰ = Mantis.Forms.AnalyticalFormField(0, my_sol, geometry, "u")
        du⁰ = Mantis.Forms.AnalyticalFormField(1, grad_my_sol, geometry, "du")
        f⁰ = Mantis.Forms.AnalyticalFormField(0, laplace_my_sol, geometry, "f")
        return u⁰, du⁰, f⁰
    elseif form_rank == 1
        u¹ = Mantis.Forms.AnalyticalFormField(1, vec_sol, geometry, "u")
        δu¹ = Mantis.Forms.AnalyticalFormField(0, minus_div_sol, geometry, "δu")
        f¹ = Mantis.Forms.AnalyticalFormField(1, vec_laplace_sol, geometry, "f")
        return u¹, δu¹, f¹
    elseif form_rank == 2
        u² = Mantis.Forms.AnalyticalFormField(2, my_sol, geometry, "u")
        δu² = Mantis.Forms.AnalyticalFormField(1, flux_my_sol, geometry, "δu")
        f² = Mantis.Forms.AnalyticalFormField(2, laplace_my_sol, geometry, "f")
        return u², δu², f²
    else
        throw(ArgumentError("Form rank not supported."))
    end
end

function circular_data(rank::Int, geo::Mantis.Geometry.AbstractGeometry; a=100, b=9 / 100)
    if rank != 1
        throw(ArgumentError("This problem data is only defined for 1-forms."))
    end

    ϕ(x, y) = (x * (1 - x) * y * (1 - y))^2

    ∇ϕ₁(x, y) = 2 * (y - 1)^2 * y^2 * (x - 1) * x * (2 * x - 1)
    d∇ϕ₁dx(x, y) = 2 * (y - 1)^2 * y^2 * (6 * x^2 - 6 * x + 1)
    d∇ϕ₁²dx₂(x, y) = 2 * (y - 1)^2 * y^2 * (12 * x - 6)
    d∇ϕ₁dy(x, y) = 4 * (x - 1) * x * (2 * x - 1) * (y - 1) * y * (2 * y - 1)
    d∇ϕ₁²dy₂(x, y) = 4 * (x - 1) * x * (2 * x - 1) * (6 * y^2 - 6 * y + 1)

    ∇ϕ₂(x, y) = 2 * (x - 1)^2 * x^2 * (y - 1) * y * (2 * y - 1)
    d∇ϕ₂dx(x, y) = 4 * (y - 1) * y * (2 * y - 1) * (x - 1) * x * (2 * x - 1)
    d∇ϕ₂²dx₂(x, y) = 4 * (y - 1) * y * (2 * y - 1) * (6 * x^2 - 6 * x + 1)
    d∇ϕ₂dy(x, y) = 2 * (x - 1)^2 * x^2 * (6 * y^2 - 6 * y + 1)
    d∇ϕ₂²dy₂(x, y) = 2 * (x - 1)^2 * x^2 * (12 * y - 6)

    φ(x, y) = tanh(a * ((x - 0.5)^2 + (y - 0.5)^2 - b))
    dφdx(x, y) = a * sech(a * ((x - 1 / 2)^2 + (y - 1 / 2)^2 - b))^2 * (2 * x - 1)
    dφ²dx₂(x, y) =
        -2 *
        a *
        sech((a * (2 * x^2 - 2 * x + 2 * (y - 1) * y - 2 * b + 1)) / 2)^2 *
        (
            a *
            (2 * x - 1)^2 *
            tanh((a * (2 * x^2 - 2 * x + 2 * (y - 1) * y - 2 * b + 1)) / 2) - 1
        )
    dφdy(x, y) = a * sech(a * ((y - 1 / 2)^2 + (x - 1 / 2)^2 - b))^2 * (2 * y - 1)
    dφ²dy₂(x, y) =
        -2 *
        a *
        sech((a * (2 * y^2 - 2 * y + 2 * (x - 1) * x - 2 * b + 1)) / 2)^2 *
        (
            a *
            (2 * y - 1)^2 *
            tanh((a * (2 * y^2 - 2 * y + 2 * (x - 1) * x - 2 * b + 1)) / 2) - 1
        )

    # u = [∇ϕ₁φ, ∇ϕ₁φ]
    u₁(x, y) = ∇ϕ₁(x, y) * φ(x, y)
    u₂(x, y) = ∇ϕ₂(x, y) * φ(x, y)

    du₁dx(x, y) = d∇ϕ₁dx(x, y) * φ(x, y) + ∇ϕ₁(x, y) * dφdx(x, y)
    du₁²dx₂(x, y) =
        d∇ϕ₁²dx₂(x, y) * φ(x, y) + 2 * d∇ϕ₁dx(x, y) * dφdx(x, y) + ∇ϕ₁(x, y) * dφ²dx₂(x, y)
    du₁dy(x, y) = d∇ϕ₁dy(x, y) * φ(x, y) + ∇ϕ₁(x, y) * dφdy(x, y)
    du₁²dy₂(x, y) =
        d∇ϕ₁²dy₂(x, y) * φ(x, y) + 2 * d∇ϕ₁dy(x, y) * dφdy(x, y) + ∇ϕ₁(x, y) * dφ²dy₂(x, y)

    du₂dx(x, y) = d∇ϕ₂dx(x, y) * φ(x, y) + ∇ϕ₂(x, y) * dφdx(x, y)
    du₂²dx₂(x, y) =
        d∇ϕ₂²dx₂(x, y) * φ(x, y) + 2 * d∇ϕ₂dx(x, y) * dφdx(x, y) + ∇ϕ₂(x, y) * dφ²dx₂(x, y)

    du₂dy(x, y) = d∇ϕ₂dy(x, y) * φ(x, y) + ∇ϕ₂(x, y) * dφdy(x, y)
    du₂²dy₂(x, y) =
        d∇ϕ₂²dy₂(x, y) * φ(x, y) + 2 * d∇ϕ₂dy(x, y) * dφdy(x, y) + ∇ϕ₂(x, y) * dφ²dy₂(x, y)

    function u_function(x)
        return [u₁.(x[:, 1], x[:, 2]), u₂.(x[:, 1], x[:, 2])]
    end

    # δu = -div u
    function δu_function(x)
        return [@. -(du₁dx(x[:, 1], x[:, 2]) + du₂dy(x[:, 1], x[:, 2]))]
    end

    # f = [-Δu₁, -Δu₂]
    function f_function(x)
        return [
            -(du₁²dx₂.(x[:, 1], x[:, 2]) + du₁²dy₂.(x[:, 1], x[:, 2])),
            -(du₂²dx₂.(x[:, 1], x[:, 2]) + du₂²dy₂.(x[:, 1], x[:, 2])),
        ]
    end

    δu = Forms.AnalyticalFormField(0, δu_function, geo, "δu")
    u = Mantis.Forms.AnalyticalFormField(1, u_function, geo, "u")
    f = Mantis.Forms.AnalyticalFormField(1, f_function, geo, "f")

    return δu, u, f
end

############################################################################################
#                                       Convenience                                        #
############################################################################################

function get_elements_in_box(
    geometry::Geometry.AbstractGeometry{2},
    first_element::NTuple{2, Int},
    last_element::NTuple{2, Int},
)
    lin_num_elements = Geometry.get_lin_num_elements(geometry)
    elements_in_box = Vector{Int}(
        undef,
        (last_element[1] - first_element[1] + 1) * (last_element[2] - first_element[2] + 1),
    )
    count = 1
    for y_element in first_element[2]:last_element[2]
        for x_element in first_element[1]:last_element[1]
			element = (x_element, y_element)
			elements_in_box[count] = lin_num_elements[element...]
			count += 1
        end
    end

    return elements_in_box
end
