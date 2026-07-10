"""
    butcher_tableau_to_glm(
        A::SMatrix{num_stages, num_stages, T},
        B::SVector{num_stages, T},
        C::SVector{num_stages, T},
        order::Int,
    ) where {num_stages, T}

Convert the given `A`, `B`, and `C` matrices from a Butcher Tableau into the GLM framework.
Will automatically choose the correct type for the integrator.
"""
function butcher_tableau_to_glm(
    A::SMatrix{num_stages, num_stages, T},
    B::SVector{num_stages, T},
    C::SVector{num_stages, T},
    order::Int,
) where {num_stages, T}
    # If the matrix A has non-zero elements in the upper triangular part (including the
    # diagonal) then it is an implicit scheme. We do make a distinction between diagonally
    # implicit (nonzero diagonal but all other uppper triangular entries are zero) and
    # fully implicit schemes.
    is_implicit, is_diagonally_implicit = check_implicit(A)

    U = ones(SMatrix{num_stages, 1, T})
    V = ones(SMatrix{1, 1, T})

    time_levels = TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    )

    if is_implicit
        return Implicit(
            A, SMatrix{1, num_stages}(transpose(B)), U, V, C, time_levels, order
        )
    elseif is_diagonally_implicit
        return DiagonallyImplicit(
            A, SMatrix{1, num_stages}(transpose(B)), U, V, C, time_levels, order
        )
    else
        return Explicit(
            A, SMatrix{1, num_stages}(transpose(B)), U, V, C, time_levels, order
        )
    end
end

############## Runge-Kutta ################

# Explicit:

# order 1
"""
    FORWARD_EULER

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const FORWARD_EULER = butcher_tableau_to_glm(
    SMatrix{1, 1}(0.0), SVector(1.0), SVector(0.0), 1
)
# order 2
"""
    EXPLICIT_MIDPOINT

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const EXPLICIT_MIDPOINT = butcher_tableau_to_glm(
    SMatrix{2, 2}(0.0, 1/2, 0.0, 0.0), SVector(0.0, 1.0), SVector(0.0, 1 / 2), 2
)

"""
    HEUN2

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const HEUN2 = butcher_tableau_to_glm(
    SMatrix{2, 2}(0.0, 1.0, 0.0, 0.0), SVector(1 / 2, 1 / 2), SVector(0.0, 1.0), 2
)

"""
    RALSTON2

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RALSTON2 = butcher_tableau_to_glm(
    SMatrix{2, 2}(0.0, 2/3, 0.0, 0.0), SVector(1 / 4, 3 / 4), SVector(0.0, 2 / 3), 2
)

# order 3
"""
    HEUN3

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const HEUN3 = butcher_tableau_to_glm(
    SMatrix{3, 3}(0.0, 1/3, 0.0, 0.0, 0.0, 2/3, 0.0, 0.0, 0.0),
    SVector(1 / 4, 0.0, 3 / 4),
    SVector(0.0, 1 / 3, 2 / 3),
    3,
)

"""
    RK3

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RK3 = butcher_tableau_to_glm(
    SMatrix{3, 3}(0.0, 1/2, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    SVector(1 / 6, 2 / 3, 1 / 6),
    SVector(0.0, 1 / 2, 1.0),
    3,
)

"""
    RALSTON3

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RALSTON3 = butcher_tableau_to_glm(
    SMatrix{3, 3}(0.0, 1/2, 0.0, 0.0, 0.0, 3/4, 0.0, 0.0, 0.0),
    SVector(2 / 9, 1 / 3, 4 / 9),
    SVector(0.0, 1 / 2, 3 / 4),
    3,
)

"""
    VDHW3

Van der Houwen's/Wray's third-order method

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const VDHW3 = butcher_tableau_to_glm(
    SMatrix{3, 3}(0.0, 8/15, 1/4, 0.0, 0.0, 5/12, 0.0, 0.0, 0.0),
    SVector(1 / 4, 0, 3 / 4),
    SVector(0.0, 8 / 15, 2 / 3),
    3,
)

"""
    SSPRK3

Third-order Strong Stability Preserving Runge-Kutta.

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const SSPRK3 = butcher_tableau_to_glm(
    SMatrix{3, 3}(0.0, 1.0, 1/4, 0.0, 0.0, 1/4, 0.0, 0.0, 0.0),
    SVector(1 / 6, 1 / 6, 2 / 3),
    SVector(0.0, 1.0, 1 / 2),
    3,
)

# order 4
"""
    RK4

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RK4 = butcher_tableau_to_glm(
    SMatrix{4, 4}(
        0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ),
    SVector(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    SVector(0.0, 0.5, 0.5, 1.0),
    4,
)

"""
    RK4_3_8

3/8-rule fourth-order method.

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RK4_3_8 = butcher_tableau_to_glm(
    SMatrix{4, 4}(
        0.0, 1/3, -1/3, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ),
    SVector(1 / 8, 3 / 8, 3 / 8, 1 / 8),
    SVector(0.0, 1 / 3, 2 / 3, 1.0),
    4,
)

"""
    RALSTON4

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RALSTON4 = butcher_tableau_to_glm(
    SMatrix{4, 4}(
        0.0,
        2/5,
        (-2889 + 1428*sqrt(5))/1024,
        (-3365 + 2094*sqrt(5))/6040,
        0.0,
        0.0,
        (3785 - 1620*sqrt(5))/1024,
        (-975 - 3046*sqrt(5))/2552,
        0.0,
        0.0,
        0.0,
        (467040 + 203968*sqrt(5))/240845,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    SVector(
        (263 + 24*sqrt(5)) / 1812,
        (125 - 1000*sqrt(5)) / 3828,
        (3426304 + 1661952*sqrt(5)) / 5924787,
        (30 - 4*sqrt(5)) / 123,
    ),
    SVector(0.0, 2 / 5, (14 - 3*sqrt(5)) / 16, 1.0),
    4,
)

# Implicit:

# order 1
"""
    BACKWARD_EULER

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const BACKWARD_EULER = butcher_tableau_to_glm(
    SMatrix{1, 1}(1.0), SVector(1.0), SVector(1.0), 1
)

"""
    RADAU_IA_1

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RADAU_IA_1 = butcher_tableau_to_glm(SMatrix{1, 1}(1.0), SVector(1.0), SVector(1.0), 1)

# order 2
"""
    IMPLICIT_MIDPOINT

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const IMPLICIT_MIDPOINT = butcher_tableau_to_glm(
    SMatrix{1, 1}(0.5), SVector(1.0), SVector(1 / 2), 2
)

"""
    CRANK_NICOLSON

Crank-Nicolson method of order 2. From
[Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const CRANK_NICOLSON = butcher_tableau_to_glm(
    SMatrix{2, 2}(0.0, 0.5, 0.0, 0.5), SVector(0.5, 0.5), SVector(0.0, 1.0), 2
)

const _α_DIRK2 = 1 - sqrt(2)/2
"""
    DIRK2

Strongly S-stable, two-stage, DIRK scheme of order 2. See [Alexander1977](@cite) page 1012.
For this scheme, ``\\alpha = 1 - sqrt(2)/2``.
"""
const DIRK2 = butcher_tableau_to_glm(
    SMatrix{2, 2}(_α_DIRK2, 1-_α_DIRK2, 0.0, _α_DIRK2),
    SVector(1-_α_DIRK2, _α_DIRK2),
    SVector(_α_DIRK2, 1),
    2,
)

# order 3
"""
    DIRK3

Crouzeix's two-stage, 3rd order, A-stable DIRK scheme. See [Alexander1977](@cite) page 1008.
"""
const DIRK3 = butcher_tableau_to_glm(
    SMatrix{2, 2}(1/2 + 1/(2*sqrt(3)), -1/sqrt(3), 0.0, 1/2+1/(2*sqrt(3))),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 + 1/(2*sqrt(3)), 1 / 2 - 1/(2*sqrt(3))),
    3,
)

const _ESDIRK32_g = 0.4358665215
"""
    ESDIRK32

Four-stage, 3rd order, stiffly accurate, A- and L-stable ESDIRK scheme. See
[Kvaerno2004](@cite) page 497.
"""
const ESDIRK32 = butcher_tableau_to_glm(
    SMatrix{4, 4}(
        0.0,
        _ESDIRK32_g,
        (-4.0*_ESDIRK32_g^2 + 6*_ESDIRK32_g - 1.0)/(4.0 * _ESDIRK32_g),
        (6.0*_ESDIRK32_g - 1.0)/(12.0 * _ESDIRK32_g),
        0.0,
        _ESDIRK32_g,
        (-2.0*_ESDIRK32_g+1.0)/(4.0*_ESDIRK32_g),
        -1.0/((24.0*_ESDIRK32_g-12.0)*_ESDIRK32_g),
        0.0,
        0.0,
        _ESDIRK32_g,
        (-6.0*_ESDIRK32_g^2 + 6*_ESDIRK32_g - 1.0)/(6.0*_ESDIRK32_g - 3.0),
        0.0,
        0.0,
        0.0,
        _ESDIRK32_g,
    ),
    SVector(
        (6.0*_ESDIRK32_g - 1.0)/(12.0 * _ESDIRK32_g),
        -1.0/((24.0*_ESDIRK32_g-12.0)*_ESDIRK32_g),
        (-6.0*_ESDIRK32_g^2 + 6*_ESDIRK32_g - 1.0)/(6.0*_ESDIRK32_g - 3.0),
        _ESDIRK32_g,
    ),
    SVector(0.0, 2.0*_ESDIRK32_g, 1.0, 1.0),
    3,
)

"""
    RADAU_IA_3

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const RADAU_IA_3 = butcher_tableau_to_glm(
    SMatrix{2, 2}(1/4, 1/4, -1/4, 5/12), SVector(1 / 4, 3 / 4), SVector(0, 2 / 3), 3
)

# order 4
const _α_DIRK4 = 2 / sqrt(3) * cos(pi / 18)
"""
    DIRK4

Crouzeix's three-stage, 4th order, A-stable DIRK scheme. See [Alexander1977](@cite) page
1008.
"""
const DIRK4 = butcher_tableau_to_glm(
    SMatrix{3, 3}(
        (1+_α_DIRK4)/2,
        -_α_DIRK4/2,
        1+_α_DIRK4,
        0.0,
        (1+_α_DIRK4)/2,
        -(1.0 + 2*_α_DIRK4),
        0.0,
        0.0,
        (1+_α_DIRK4)/2,
    ),
    SVector(1 / (6 * _α_DIRK4^2), 1.0 - 1 / (3 * _α_DIRK4^2), 1 / (6 * _α_DIRK4^2)),
    SVector((1 + _α_DIRK4) / 2, 1 / 2, (1 - _α_DIRK4) / 2),
    4,
)

"""
    GAUSS_LEGENDRE_4

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const GAUSS_LEGENDRE_4 = butcher_tableau_to_glm(
    SMatrix{2, 2}(1/4, 1/4+sqrt(3)/6, 1/4-sqrt(3)/6, 1/4),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6),
    4,
)

# order 6
"""
    GAUSS_LEGENDRE_6

From [Wikipedia's list of Runge-Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods).
"""
const GAUSS_LEGENDRE_6 = butcher_tableau_to_glm(
    SMatrix{3, 3}(
        5/36,
        5 / 36+sqrt(15) / 24,
        5 / 36+sqrt(15) / 30,
        2 / 9-sqrt(15) / 15,
        2/9,
        2 / 9+sqrt(15) / 15,
        5 / 36-sqrt(15) / 30,
        5 / 36-sqrt(15) / 24,
        5/36,
    ),
    SVector(5 / 18, 4 / 9, 5 / 18),
    SVector(1 / 2 - sqrt(15) / 10, 1 / 2, 1 / 2 + sqrt(15) / 10),
    6,
)

############## Multi-step  ################

# Explicit:

# Adams-Bashforth
"""
    AB1

Adams-Bashforth 1:
```math
y_{n+1} = y_{n} + \\Delta t f(y_{n})\\;.
```
"""
const AB1 = Explicit(
    SMatrix{1, 1}(0.0), # A
    SMatrix{1, 1}(1.0), # B
    SMatrix{1, 1}(1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(0.0),
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    1,
)

"""
    AB2

Adams-Bashforth 2:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{3}{2} f(y_{n}) - \\frac{1}{2} f(y_{n-1}))\\;.
```
"""
const AB2 = Explicit(
    SMatrix{1, 1}(0.0), # A
    SMatrix{2, 1}(3/2, 1.0), # B
    SMatrix{1, 2}(1.0, 0.0), # U
    SMatrix{2, 2}(1.0, 0.0, -1/2, 0.0), # V
    SVector(0.0),
    TimeLevels(
        [0], # y
        Int[], # Δt G
        [1],  # Δt F
    ),
    2,
)

"""
    AB3

Adams-Bashforth 3:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{23}{12} f(y_{n}) - \\frac{4}{3} f(y_{n-1}) + \\frac{5}{12} f(y_{n-2}))\\;.
```
"""
const AB3 = Explicit(
    SMatrix{1, 1}(0.0), # A
    SMatrix{3, 1}(23/12, 1.0, 0.0), # B
    SMatrix{1, 3}(1.0, 0.0, 0.0), # U
    SMatrix{3, 3}(1.0, 0.0, 0.0, -16/12, 0.0, 1.0, 5/12, 0.0, 0.0), # V
    SVector(0.0),
    TimeLevels(
        [0], # y(n)
        Int[], # Δt G
        [1, 2],  # Δt F(n-1), Δt F(n-2)
    ),
    3,
)

"""
    AB4

Adams-Bashforth 4:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{55}{24} f(y_{n}) - \\frac{59}{24} f(y_{n-1}) + \\frac{37}{24} f(y_{n-2}) - \\frac{9}{24} f(y_{n-3}))\\;.
```
"""
const AB4 = Explicit(
    SMatrix{1, 1}(0.0), # A
    SMatrix{4, 1}(55/24, 1.0, 0.0, 0.0), # B
    SMatrix{1, 4}(1.0, 0.0, 0.0, 0.0), # U
    SMatrix{4, 4}(
        1.0,
        0.0,
        0.0,
        0.0,
        -59/24,
        0.0,
        1.0,
        0.0,
        37/24,
        0.0,
        0.0,
        1.0,
        -9/24,
        0.0,
        0.0,
        0.0,
    ), # V
    SVector(0.0),
    TimeLevels(
        [0], # y
        Int[], # Δt G
        [1, 2, 3],  # Δt F
    ),
    4,
)

# Implicit:

# Adams-Moulton
"""
    AM0

Adams-Moulton 0:
```math
y_{n+1} = y_{n} + \\Delta t g(y_{n})\\;.
```
"""
const AM0 = DiagonallyImplicit(
    SMatrix{1, 1}(1.0), # A
    SMatrix{1, 1}(1.0), # B
    SMatrix{1, 1}(1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    1,
)

"""
    AM1

Adams-Moulton 1:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{1}{2} g(y_{n+1}) + \\frac{1}{2} g(y_{n}))\\;.
```
"""
const AM1 = DiagonallyImplicit(
    SMatrix{1, 1}(0.5), # A
    SMatrix{2, 1}(0.5, 1.0), # B
    SMatrix{1, 2}(1.0, 0.5), # U
    SMatrix{2, 2}(1.0, 0.0, 0.5, 0.0), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        [0], # Δt G
        Int[],  # Δt F
    ),
    2,
)

"""
    AM2

Adams-Moulton 2:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{5}{12} g(y_{n+1}) + \\frac{8}{12} g(y_{n}) - \\frac{1}{12} g(y_{n-1})\\;.
```
"""
const AM2 = DiagonallyImplicit(
    SMatrix{1, 1}(5/12), # A
    SMatrix{3, 1}(5/12, 1.0, 0.0), # B
    SMatrix{1, 3}(1.0, 8/12, -1/12), # U
    SMatrix{3, 3}(1.0, 0.0, 0.0, 8/12, 0.0, 1.0, -1/12, 0.0, 0.0), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        [0, 1], # Δt G
        Int[],  # Δt F
    ),
    3,
)

"""
    AM3

Adams-Moulton 3:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{9}{24} g(y_{n+1}) + \\frac{19}{24} g(y_{n}) - \\frac{5}{24} g(y_{n-1}) + \\frac{1}{24} g(y_{n-2}))\\;.
```
"""
const AM3 = DiagonallyImplicit(
    SMatrix{1, 1}(9/24), # A
    SMatrix{4, 1}(9/24, 1.0, 0.0, 0.0), # B
    SMatrix{1, 4}(1.0, 19/24, -5/24, 1/24), # U
    SMatrix{4, 4}(
        1.0, 0.0, 0.0, 0.0, 19/24, 0.0, 1.0, 0.0, -5/24, 0.0, 0.0, 1.0, 1/24, 0.0, 0.0, 0.0
    ), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        [0, 1, 2], # Δt G
        Int[],  # Δt F
    ),
    4,
)

"""
    AM4

Adams-Moulton 4:
```math
y_{n+1} = y_{n} + \\Delta t (\\frac{251}{720} g(y_{n+1}) + \\frac{646}{720} g(y_{n}) - \\frac{264}{720} g(y_{n-1}) + \\frac{106}{720} g(y_{n-2}) - \\frac{19}{720} g(y_{n-3}))\\;.
```
"""
const AM4 = DiagonallyImplicit(
    SMatrix{1, 1}(251/720), # A
    SMatrix{5, 1}(251/720, 1.0, 0.0, 0.0, 0.0), # B
    SMatrix{1, 5}(1.0, 646/720, -264/720, 106/720, -19/720), # U
    SMatrix{5, 5}(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        646/720,
        0.0,
        1.0,
        0.0,
        0.0,
        -264/720,
        0.0,
        0.0,
        1.0,
        0.0,
        106/720,
        0.0,
        0.0,
        0.0,
        1.0,
        -19/720,
        0.0,
        0.0,
        0.0,
        0.0,
    ), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        [0, 1, 2, 3], # Δt G
        Int[],  # Δt F
    ),
    5,
)

# backward differentiation formulas
"""
    BDF1

Backward differentiation formula 1:
```math
y_{n+1} = y_{n} + \\Delta t f(y_n)\\;.
```
"""
const BDF1 = DiagonallyImplicit(
    SMatrix{1, 1}(1.0), # A
    SMatrix{1, 1}(1.0), # B
    SMatrix{1, 1}(1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(1.0),
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    1,
)

"""
    BDF2

Backward differentiation formula 2:
```math
y_{n+1} = \\frac{4}{3} y_{n} - \\frac{1}{3} y_{n-1} + \\frac{2}{3} \\Delta t f(y_{n})\\;.
```
"""
const BDF2 = DiagonallyImplicit(
    SMatrix{1, 1}(2/3), # A
    SMatrix{2, 1}(2/3, 0.0), # B
    SMatrix{1, 2}(4/3, -1/3), # U
    SMatrix{2, 2}(4/3, 1.0, -1/3, 0.0), # V
    SVector(1.0),
    TimeLevels(
        [0, 1], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    2,
)

"""
    BDF3

Backward differentiation formula 3:
```math
y_{n+1} = \\frac{18}{11} y_{n} - \\frac{9}{11} y_{n-1} + \\frac{2}{11} y_{n-2} + \\frac{6}{11} \\Delta t f(y_{n})\\;.
```
"""
const BDF3 = DiagonallyImplicit(
    SMatrix{1, 1}(6/11), # A
    SMatrix{3, 1}(6/11, 0.0, 0.0), # B
    SMatrix{1, 3}(18/11, -9/11, 2/11), # U
    SMatrix{3, 3}(18/11, 1.0, 0.0, -9/11, 0.0, 1.0, 2/11, 0.0, 0.0), # V
    SVector(1.0),
    TimeLevels(
        [0, 1, 2], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    3,
)

"""
    BDF4

Backward differentiation formula 4:
```math
y_{n+1} = \\frac{48}{25} y_{n} - \\frac{36}{25} y_{n-1} + \\frac{16}{25} y_{n-2} - \\frac{3}{25} y_{n-3} + \\frac{12}{25} \\Delta t f(y_{n})\\;.
```
"""
const BDF4 = DiagonallyImplicit(
    SMatrix{1, 1}(12/25), # A
    SMatrix{4, 1}(12/25, 0.0, 0.0, 0.0), # B
    SMatrix{1, 4}(48/25, -36/25, 16/25, -3/25), # U
    SMatrix{4, 4}(
        48/25,
        1.0,
        0.0,
        0.0,
        -36/25,
        0.0,
        1.0,
        0.0,
        16/25,
        0.0,
        0.0,
        1.0,
        -3/25,
        0.0,
        0.0,
        0.0,
    ), # V
    SVector(1.0),
    TimeLevels(
        [0, 1, 2, 3], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    4,
)

##############    IMEX     ################

# Multi-stage
"""
    BACKWARD_FORWARD_EULER

See equation A10 in [Vos2011](@cite).
"""
const BACKWARD_FORWARD_EULER = IMEX(
    SMatrix{1, 1}(1.0), # A Implicit
    SMatrix{1, 1}(0.0), # A Explicit
    SMatrix{2, 1}(1.0, 0.0), # B Implicit
    SMatrix{2, 1}(0.0, 1.0), # B Explicit
    SMatrix{1, 2}(1.0, 1.0), # U
    SMatrix{2, 2}(1.0, 0.0, 1.0, 0.0), # V
    SVector(1.0), # C Implicit
    SVector(0.0), # C Explicit
    TimeLevels(
        [0], # y
        Int[], # Δt G
        [0],  # Δt F
    ),
    1,
)

"""
    MIDPOINT_IMEX

IMEX combination of the implicit and explicit midpoint rules. See equation 4.14 in
[Ern2023](@cite).
"""
const MIDPOINT_IMEX = IMEX(
    SMatrix{2, 2}(0.0, 0.0, 0.0, 1/2), # A Implicit
    SMatrix{2, 2}(0.0, 1/2, 0.0, 0.0), # A Explicit
    SMatrix{1, 2}(0.0, 1.0), # B Implicit
    SMatrix{1, 2}(0.0, 1.0), # B Explicit
    SMatrix{2, 1}(1.0, 1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(0.0, 1/2), # C Implicit
    SVector(0.0, 1/2), # C Explicit
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    2,
)

const _γ = (3 + sqrt(3)) / 6
"""
    RK3_IMEX

See equation A13 in [Vos2011](@cite).
"""
const RK3_IMEX = IMEX(
    SMatrix{3, 3}(0.0, 0.0, 0.0, 0.0, _γ, 1.0-2.0*_γ, 0.0, 0.0, _γ), # A Implicit
    SMatrix{3, 3}(0.0, _γ, _γ-1.0, 0.0, 0.0, 2.0*(1.0-_γ), 0.0, 0.0, 0.0), # A Explicit
    SMatrix{1, 3}(0.0, 0.5, 0.5), # B Implicit
    SMatrix{1, 3}(0.0, 0.5, 0.5), # B Explicit
    SMatrix{3, 1}(1.0, 1.0, 1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(0.0, _γ, 1 - _γ), # C Implicit
    SVector(0.0, _γ, 1 - _γ), # C Explicit
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    3,
)

const _γ331 = 0.5 + 1/(2*sqrt(3))
"""
    IMEX331

3-stage, 3rd order, A-stable IMEX scheme with optimal efficiency. See equation 4.18 in
[Ern2023](@cite).
"""
const IMEX331 = IMEX(
    SMatrix{3, 3}(0.0, 1/3-_γ331, _γ331, 0.0, _γ331, 2/3-2.0*_γ, 0.0, 0.0, _γ), # A Implicit
    SMatrix{3, 3}(0.0, 1/3, 0.0, 0.0, 0.0, 2.0/3.0, 0.0, 0.0, 0.0), # A Explicit
    SMatrix{1, 3}(0.25, 0.0, 0.75), # B Implicit
    SMatrix{1, 3}(0.25, 0.0, 0.75), # B Explicit
    SMatrix{3, 1}(1.0, 1.0, 1.0), # U
    SMatrix{1, 1}(1.0), # V
    SVector(0.0, 1/3, 2/3), # C Implicit
    SVector(0.0, 1/3, 2/3), # C Explicit
    TimeLevels(
        [0], # y
        Int[], # Δt G
        Int[],  # Δt F
    ),
    3,
)

# Multi-step
"""
    CNAB2

Second-order Crank-Nicolson/Adams-Bashforth linear multistep scheme. See equation A12 in
[Vos2011](@cite).
"""
const CNAB2 = IMEX(
    SMatrix{1, 1}(1/2), # A Implicit
    SMatrix{1, 1}(0.0), # A Explicit
    SMatrix{4, 1}(1 / 2, 1.0, 0.0, 0.0), # B Implicit
    SMatrix{4, 1}(0.0, 0.0, 1.0, 0.0), # B Explicit
    SMatrix{1, 4}(1.0, 1/2, 3/2, -1/2), # U
    SMatrix{4, 4}(
        1.0, 0.0, 0.0, 0.0, 1/2, 0.0, 0.0, 0.0, 3/2, 0.0, 0.0, 1.0, -1/2, 0.0, 0.0, 0.0
    ), # V
    SVector(1.0), # C Implicit
    SVector(1.0), # C Explicit
    TimeLevels(
        [0], # y
        [0], # Δt G
        [0, 1],  # Δt F
    ),
    2,
)

"""
    SSSS2

Stiffly stable splitting scheme of order 2. See [Karniadakis1991](@cite) table IV and
[Vos2011](@cite) equation 59.
"""
const SSSS2 = IMEX(
    SMatrix{1, 1}(2/3),
    SMatrix{1, 1}(0.0),
    SMatrix{4, 1}(2 / 3, 0.0, 0.0, 0.0),
    SMatrix{4, 1}(0.0, 0.0, 1.0, 0.0),
    SMatrix{1, 4}(4/3, -1/3, 4/3, -2/3),
    SMatrix{4, 4}(
        4/3, 1.0, 0.0, 0.0, -1/3, 0.0, 0.0, 0.0, 4/3, 0.0, 0.0, 1.0, -2/3, 0.0, 0.0, 0.0
    ),
    SVector(1.0),
    SVector(1.0),
    TimeLevels(
        [0, 1], # y
        Int[], # Δt G
        [0, 1],  # Δt F
    ),
    2,
)
