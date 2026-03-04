module ConstantFormSpaceTests

using Mantis
using Test
import LinearAlgebra

const manifold_dim = 2
const L = 1.0
const num_elements_per_dim = 2
const num_points_per_dim = 3
const num_quad_points_per_dim = 2

# Create a simple Cartesian geometry with random breakpoints
breakpoints = ntuple(manifold_dim) do i
    el_sizes = rand(num_elements_per_dim)
    el_sizes ./= sum(el_sizes).* L
    return cumsum([0.0; el_sizes])
end
geom = Geometry.CartesianGeometry(breakpoints)
canonical_qrule = Quadrature.tensor_product_rule(
    ntuple(i->num_quad_points_per_dim, manifold_dim), Quadrature.gauss_legendre
)
dΩ = Mantis.Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geom))

# Setup the form spaces, evaluate and check the results
ξ = Points.CartesianPoints(ntuple(i->range(0.0, rand(), num_points_per_dim), manifold_dim))
for form_rank in 0:manifold_dim
    if form_rank ∉ Set([0, manifold_dim])
        @test_throws ArgumentError Forms.ConstantFormSpace(form_rank, geom, "c")
    else
        form_space = Forms.ConstantFormSpace(form_rank, geom, "a")

        # Check number of basis functions
        @test Forms.get_num_basis(form_space) == 1

        # Check number of basis functions for each element
        for element_id in 1:Forms.get_num_elements(form_space)
            @test Forms.get_num_basis(form_space, element_id) == 1
        end

        # Check max local dimension
        @test Forms.get_max_local_dim(form_space) == 1

        # Check estimated nnz per element
        @test Forms.get_estimated_nnz_per_elem(form_space) == 1

        # Check evaluate
        for element_id in 1:Forms.get_num_elements(form_space)
            eval, inds = Forms.evaluate(form_space, element_id, ξ)
            if form_rank == 0
                @test eval == [reshape([1.0 for _ in 1:Points.get_num_points(ξ)], :, 1)]
                @test inds == [[1]]
            elseif form_rank == manifold_dim
                J = Geometry.jacobian(Forms.get_geometry(form_space), element_id, ξ)  # Jₖⱼ = ∂Φᵏ\\∂ξⱼ
                detJ =LinearAlgebra.det.(J)
                @test eval == [reshape([detJ[i] for i in 1:Points.get_num_points(ξ)], :, 1)]
                @test inds == [[1]]
            end
        end

        # Check integral
        ip = ∫(form_space ∧ ★(form_space), dΩ)
        ip_eval = 0.0
        for element_id in 1:Forms.get_num_elements(form_space)
            ip_eval += sum(Forms.evaluate(ip, element_id)[1])
        end
        @test isapprox(ip_eval, L^manifold_dim, atol=1e-12)
    end

end



end