require 'dp'

--Feature Tensor
local data = torch.rand(3,4)
local axes = {'b','f'}
local sizes = {3, 4}
local d = dp.DataTensor{data=data, axes=axes, sizes=sizes}
local t = d:feature()
function test() return d:image() end
assert(pcall(test) == false)

--Image Tensor
local data = torch.rand(3,32,32,3)
local axes = {'b','h','w', 'c'}
local d = dp.DataTensor{data=data, axes=axes}
local t = d:feature()
local i = d:image()

--Class Tensor
local data = torch.rand(48,4)
local axes = {'b','t'}
local d = dp.DataTensor{data=data, axes=axes}
local t = d:multiclass()
local i = d:class()


