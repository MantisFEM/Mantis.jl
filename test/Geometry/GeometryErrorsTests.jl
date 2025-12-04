module GeometryErrorsTests

# These tests are designed to test the error behaviour when creating a new AbstractGeometry
# subtype but without correctly defining all methods.

using Mantis
using Test

# Since v1.12, accessing a non-defined field will return a FieldError. In earlier versions
# this was an ErrorException, so we have to account for that in (CI) testing.
@static if Base.VERSION >= v"1.12"
    const fielderror = FieldError
else
    const fielderror = ErrorException
end

struct NonExistentGeometry{manifold_dim, image_dim, num_patches} <:
       Geometry.AbstractGeometry{manifold_dim, image_dim, num_patches}
    function NonExistentGeometry(m, i, n)
        return new{m, i, n}()
    end
end

# 1D, single patch
neg_1D = NonExistentGeometry(1, 1, 1)

@test Geometry.get_manifold_dim(neg_1D) == 1
@test Geometry.get_image_dim(neg_1D) == 1
@test Geometry.get_num_patches(neg_1D) == 1
@test_throws fielderror Geometry.get_patch_id(neg_1D, 1)
@test_throws fielderror Geometry.get_patch_and_local_element_id(neg_1D, 1)
@test_throws fielderror Geometry.get_global_element_id(neg_1D, 1, 1)
@test_throws fielderror Geometry.get_num_elements(neg_1D)
@test_throws fielderror Geometry.get_num_elements(neg_1D, 1)
@test_throws fielderror Geometry.get_num_elements_per_patch(neg_1D)
@test_throws MethodError Geometry.get_geometry(neg_1D)
@test_throws MethodError Geometry.get_parametric_geometry(neg_1D)
@test_throws MethodError Geometry.get_parametric_geometry(neg_1D, 1)
@test_throws MethodError Geometry.get_element_measure(neg_1D, 1)
@test_throws MethodError Geometry.get_element_lengths(neg_1D, 1)
@test_throws MethodError Geometry.get_element_vertices(neg_1D, 1)
@test_throws MethodError Geometry.evaluate(neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],)))
@test_throws MethodError Geometry.jacobian(neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],)))
@test_throws MethodError Geometry.hessian(neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],)))
@test_throws MethodError Geometry.metric(neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],)))
@test_throws MethodError Geometry.inv_metric(
    neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],))
)
@test_throws MethodError Geometry.metric_derivatives(
    neg_1D, 1, Points.CartesianPoints(([0.0, 1.0],))
)

# 1D, multi-patch
neg_1D_3 = NonExistentGeometry(1, 1, 3)

@test Geometry.get_manifold_dim(neg_1D_3) == 1
@test Geometry.get_image_dim(neg_1D_3) == 1
@test Geometry.get_num_patches(neg_1D_3) == 3
@test_throws fielderror Geometry.get_patch_id(neg_1D_3, 1)
@test_throws fielderror Geometry.get_patch_and_local_element_id(neg_1D_3, 1)
@test_throws fielderror Geometry.get_global_element_id(neg_1D_3, 1, 1)
@test_throws fielderror Geometry.get_num_elements(neg_1D_3)
@test_throws fielderror Geometry.get_num_elements(neg_1D_3, 1)
@test_throws fielderror Geometry.get_num_elements_per_patch(neg_1D_3)
@test_throws MethodError Geometry.get_geometry(neg_1D_3)
@test_throws MethodError Geometry.get_parametric_geometry(neg_1D_3)
@test_throws MethodError Geometry.get_parametric_geometry(neg_1D_3, 1)
@test_throws MethodError Geometry.get_element_measure(neg_1D_3, 1)
@test_throws MethodError Geometry.get_element_lengths(neg_1D_3, 1)
@test_throws MethodError Geometry.get_element_vertices(neg_1D_3, 1)
@test_throws MethodError Geometry.evaluate(
    neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],))
)
@test_throws MethodError Geometry.jacobian(
    neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],))
)
@test_throws MethodError Geometry.hessian(
    neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],))
)
@test_throws MethodError Geometry.metric(neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],)))
@test_throws MethodError Geometry.inv_metric(
    neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],))
)
@test_throws MethodError Geometry.metric_derivatives(
    neg_1D_3, 1, Points.CartesianPoints(([0.0, 1.0],))
)

# 3D, multi-patch
neg_3D_5 = NonExistentGeometry(3, 2, 5)

@test Geometry.get_manifold_dim(neg_3D_5) == 3
@test Geometry.get_image_dim(neg_3D_5) == 2
@test Geometry.get_num_patches(neg_3D_5) == 5
@test_throws fielderror Geometry.get_patch_id(neg_3D_5, 1)
@test_throws fielderror Geometry.get_patch_and_local_element_id(neg_3D_5, 1)
@test_throws fielderror Geometry.get_global_element_id(neg_3D_5, 1, 1)
@test_throws fielderror Geometry.get_num_elements(neg_3D_5)
@test_throws fielderror Geometry.get_num_elements(neg_3D_5, 1)
@test_throws fielderror Geometry.get_num_elements_per_patch(neg_3D_5)
@test_throws MethodError Geometry.get_geometry(neg_3D_5)
@test_throws MethodError Geometry.get_parametric_geometry(neg_3D_5)
@test_throws MethodError Geometry.get_parametric_geometry(neg_3D_5, 1)
@test_throws MethodError Geometry.get_element_measure(neg_3D_5, 1)
@test_throws MethodError Geometry.get_element_lengths(neg_3D_5, 1)
@test_throws MethodError Geometry.get_element_vertices(neg_3D_5, 1)
@test_throws MethodError Geometry.evaluate(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)
@test_throws MethodError Geometry.jacobian(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)
@test_throws MethodError Geometry.hessian(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)
@test_throws MethodError Geometry.metric(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)
@test_throws MethodError Geometry.inv_metric(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)
@test_throws MethodError Geometry.metric_derivatives(
    neg_3D_5, 1, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
)

end
