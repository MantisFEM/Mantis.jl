module APITests

using Mantis

import Base: copy, deepcopy, fill!, copyto!, map, map!

using Test

############################################################################################
#                                          Status                                          #
############################################################################################

@test Caching.Flag <: Mantis.Caching.AbstractStatus

flag = Caching.Flag(false)
@test flag() == false
Caching.set!(flag, true)
@test flag() == true

############################################################################################
#                                          Buffer                                          #
############################################################################################

struct NewObject
    mat::Matrix{Float64}
end

sz = (2, 2)
obj = NewObject(zeros(sz))

@test_throws MethodError Caching.preallocate(obj)
@test_throws MethodError Caching.preallocate(obj, sz)

struct NewBuffer <: Caching.AbstractBuffer
    mat::Matrix{Float64}
    status::Caching.Flag
    function NewBuffer(mat::Matrix{Float64})
        new(mat, Caching.Flag(false))
    end
end

Caching.preallocate(::NewObject, sz=(1, 1)) = NewBuffer(Matrix{Float64}(undef, sz))

buff = Caching.preallocate(obj)
@test isa(buff, NewBuffer)
@test isa(buff, Caching.AbstractBuffer)
buff = Caching.preallocate(obj, sz)
@test isa(buff, NewBuffer)
@test isa(buff, Caching.AbstractBuffer)

@test isa(Caching.status(buff), Caching.Flag)
@test Caching.isfilled(buff) == false
Caching.setfilled!(buff, true)
@test Caching.isfilled(buff) == true
 
@test_throws MethodError Caching.peek(obj)
Caching.peek(buff::NewBuffer) = buff.mat
@test isa(Caching.peek(buff), Matrix{Float64})
@test isa(buff(), Matrix{Float64})

@test_throws MethodError copy(obj)
copy(buff::NewBuffer) = NewBuffer(buff())
@test isa(copy(buff), NewBuffer)
buff = Caching.preallocate(obj, sz)
copy_buff = copy(buff)
@test buff() === copy_buff()

# @test_throws MethodError deepcopy(obj)
deepcopy(buff::NewBuffer) = NewBuffer(buff() |> deepcopy)
@test isa(deepcopy(buff), NewBuffer)
buff = Caching.preallocate(obj, sz)
deepcopy_buff = deepcopy(buff)
@test !(buff === deepcopy_buff)
@test !(buff() === deepcopy_buff())

@test_throws MethodError Caching.update!(obj, 1.0)
@test_throws MethodError fill!(obj, 1.0)
function fill!(buff::NewBuffer, val) 
    mat = buff()
    fill!(mat, val)

    return buff
end
@test isa(fill!(buff, 1.0), NewBuffer)
buff = Caching.preallocate(obj, sz)
fill!(buff, 2.0)
@test buff() == fill(2.0, sz)
@test Caching.isfilled(buff) == false
Caching.update!(buff, 1.0)
@test buff() == fill(1.0, sz)
@test Caching.isfilled(buff) == true
# 10.0 is ignored because isfilled(buff) == true
Caching.update!(buff, 10.0)
@test buff() == fill(1.0, sz)

@test_throws MethodError Caching.clear!(buff)
Caching.clear!(buff::NewBuffer) = fill!(buff, 0.0)
@test isa(Caching.clear!(buff), NewBuffer)
@test buff() == zeros(sz)

dest_buff = NewBuffer(zeros(sz))
src_buff = NewBuffer(ones(sz))
@test_throws MethodError copyto!(dest_buff, src_buff)
function copyto!(dest_buff::NewBuffer, src_buff::NewBuffer)
    copyto!(dest_buff(), src_buff())

    return dest_buff
end
@test isa(copyto!(dest_buff, src_buff), NewBuffer)
@test dest_buff() == ones(sz)

############################################################################################
#                                          Cache                                           #
############################################################################################

struct AnotherObject
    label::String
end

obj = AnotherObject("test")
sz = (2, 2)

@test_throws MethodError Caching.Cache(obj)
@test_throws MethodError Caching.Cache(obj, sz)

obj = NewObject(zeros(sz))
cache = Caching.Cache(obj)
@test Caching.get_object(cache) === obj
@test isa(Caching.get_buffer(cache), NewBuffer)
ext = Caching.extract(cache)
@test ext[1] === obj
@test isa(ext[2], NewBuffer)

buff = NewBuffer(zeros(sz))
obj = NewObject(buff())
cache = Caching.Cache(obj, buff)
@test cache() === buff()
fill!(buff, 1.0)
@test cache() == ones(sz)

fill!(buff::NewBuffer, ::NewObject, args...) = fill!(buff, args...)

fill!(cache, 2.0)
@test cache() == fill(2.0, sz)
@test cache() === buff()

cache(1.0)
@test cache() == ones(sz)

Caching.clear!(cache)
@test cache() == zeros(sz)

############################################################################################
#                                    CompositeBuffer                                       #
############################################################################################

# Construction with a vararg of sub-buffers.
buff1 = NewBuffer(zeros(sz))
buff2 = NewBuffer(ones(sz))
comp = Caching.CompositeBuffer(buff1, buff2)
@test isa(comp, Caching.CompositeBuffer)
@test isa(comp, Caching.AbstractBuffer)

# Construction from a tuple directly.
comp_from_tuple = Caching.CompositeBuffer((buff1, buff2))
@test isa(comp_from_tuple, Caching.CompositeBuffer)
@test isa(comp_from_tuple, Caching.AbstractBuffer)

# Reject non-AbstractBuffer elements.
@test_throws ArgumentError Caching.CompositeBuffer((buff1, "not_a_buffer"))

# get_buffers returns the original tuple.
@test Caching.get_buffers(comp) === (buff1, buff2)

# map(f, comp) applies f to each sub-buffer and returns a tuple.
statuses = map(Caching.status, comp)
@test isa(statuses, NTuple{2, Caching.Flag})

# isfilled delegates to all sub-buffers: true only when every sub-buffer is filled.
comp = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
@test Caching.isfilled(comp) == false
# map(isfilled, comp) checks each sub-buffer individually.
@test map(Caching.isfilled, comp)[1] == false

# map!(f, comp) applies f to every sub-buffer in-place, returning comp.
result = map!(b -> Caching.setfilled!(b, true), comp)
@test result === comp
@test Caching.isfilled(comp) == true
@test map(Caching.isfilled, comp)[1] == true
@test map(Caching.isfilled, comp)[2] == true

# map!(f, comp, i) applies f only to the i-th sub-buffer, returning comp.
comp = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
result = map!(b -> Caching.setfilled!(b, true), comp, 1)
@test result === comp
@test map(Caching.isfilled, comp)[1] == true
@test map(Caching.isfilled, comp)[2] == false
# isfilled(comp) is false because not all sub-buffers are filled.
@test Caching.isfilled(comp) == false

# map(peek, comp) returns a tuple of each sub-buffer's contents.
comp = Caching.CompositeBuffer(buff1, buff2)
peeked = comp()
@test isa(peeked, NTuple{2, Matrix{Float64}})
@test peeked[1] === Caching.peek(buff1)
@test peeked[2] === Caching.peek(buff2)

# map!(f, comp) broadcasts a fill operation to every sub-buffer.
comp = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
map!(b -> fill!(b, 3.0), comp)
@test comp()[1] == fill(3.0, sz)
@test comp()[2] == fill(3.0, sz)

# map!(f, comp, i) fills only the i-th sub-buffer.
comp = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
map!(b -> fill!(b, 7.0), comp, 1)
@test comp()[1] == fill(7.0, sz)
@test comp()[2] == zeros(sz)

# map!(clear!, comp) clears each sub-buffer.
comp = Caching.CompositeBuffer(NewBuffer(ones(sz)), NewBuffer(ones(sz)))
map!(Caching.clear!, comp)
@test comp()[1] == zeros(sz)
@test comp()[2] == zeros(sz)

# map!(update!, comp)
comp = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
map!(b -> Caching.update!(b, 5.0), comp)
@test comp()[1] == fill(5.0, sz)
@test comp()[2] == fill(5.0, sz)
@test Caching.isfilled(comp) == true
# 9.0 is ignored because isfilled(comp) == true
map!(b -> Caching.update!(b, 9.0), comp)
@test comp()[1] == fill(5.0, sz)
map!(b -> Caching.update!(b, 9.0), comp, 1)
@test comp()[1] == fill(5.0, sz)

# copy produces a shallow copy; wrap in CompositeBuffer to copy the composite.
comp = Caching.CompositeBuffer(NewBuffer(ones(sz)), NewBuffer(ones(sz)))
comp_copy = copy(comp)
@test isa(comp_copy, Caching.CompositeBuffer)
@test comp_copy()[1] === comp()[1]

# deepcopy produces independent sub-buffers.
comp_deep = deepcopy(comp)
@test isa(comp_deep, Caching.CompositeBuffer)
@test !(comp_deep()[1] === comp()[1])
@test comp_deep()[1] == comp()[1]

# copyto! copies contents from src to dest (same buffer type B required).
dest = Caching.CompositeBuffer(NewBuffer(zeros(sz)), NewBuffer(zeros(sz)))
src  = Caching.CompositeBuffer(NewBuffer(ones(sz)),  NewBuffer(ones(sz)))
copyto!(dest, src)
@test dest()[1] == ones(sz)
@test dest()[2] == ones(sz)

end
