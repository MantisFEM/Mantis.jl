using Mantis
import LinearAlgebra

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

@eval Mantis.Forms begin
    function Forms.get_form(form::Forms.BinaryFormTransformation)
        forms = Forms.get_forms(form)
        if isa(first(forms), Forms.AbstractFormSpace)
            return first(forms)
        end

        return last(forms)
    end
end

Base.zero(form::Forms.BinaryFormTransformation) = 0.0 * form

# Define a function-valued matrix type
struct FuncMatrix{T}
    funcs::Matrix{T}
end

Base.size(F::FuncMatrix) = size(F.funcs)

function Base.:*(A::Matrix, D::FuncMatrix)
    @assert size(A) == (1, 3) "Wrong matrix size."
    @assert size(D) == (3, 2) "Wrong matrix size."
    D = D.funcs
    result = Forms.AbstractForm[
        (D[1, 1](A[1, 1])+D[2, 1](A[1, 2])+D[3, 1](A[1, 3])) (D[1, 2](A[1, 1])+D[2, 2](A[1, 2])+D[3, 2](A[1, 3]))
    ]

    return result
end
