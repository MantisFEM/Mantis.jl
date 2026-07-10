module TimeIntegrationAdvectionTests

import SparseArrays
import LinearAlgebra
using Mantis
using Test

# We can test (some) numerical schemes by considering a simple advection equation. For
# example, the forward Euler scheme with CFL exactly equal to 1 is exact in this case, so
# we can easily compare the solution to the initial condition.

# Consider the advection equation on a 1D domain (x_L, x_R) as
# dc/dt = Ac
# where c is the vector of unknowns, and A is the (N-1, N-1) discretisation matrix.

# Special case backward FD + Forward Euler at CFL = 1 --------------------------------------
# If we construct A as first-order upwind finite difference method, the scheme with forward
# Euler time integration with CFL 1 will be exact.
const dx = 0.01
const L = 1.0
const velocity = 0.1
const dt = dx / velocity  # gives CFL = 1
const grid = 0.0:dx:L
const A =
    (velocity / dx) *
    SparseArrays.spdiagm(0 => ones(length(grid)), -1 => -1 .* ones(length(grid)-1))
# Set periodic boundary conditions.
A[1, end] = -(velocity / dx)

function discretised_advection_equation(output, c, t)
    LinearAlgebra.mul!(output, -A, c)
    return nothing
end

c_0 = sinpi.(2.0 .* grid)

linear_advection_ode = TimeIntegrators.define_explicit_ode(discretised_advection_equation)

@testset "Forward Euler CFL 1 Advection Tests" verbose = true begin
    c_n = TimeIntegrators.initialise_scheme(c_0, TimeIntegrators.FORWARD_EULER)

    # Pick the end time such that the IC traverses our domain exactly 10 times.
    for t in 0.0:dt:(20.0 + dt)
        TimeIntegrators.time_integrate!(c_n, linear_advection_ode, t, dt)
    end

    @test isapprox(c_0, TimeIntegrators.get_solution(c_n), atol=1e-15, rtol=1e-15)
end

end
