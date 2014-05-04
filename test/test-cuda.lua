local mytester 
local dptest = {}

function dptest.datatensor()
   local data = torch.rand(3,4)
   local axes = {'b','f'}
   local sizes = {3, 4}
   local dt = dp.DataTensor{data=data, axes=axes, sizes=sizes}
   local f = dt:feature('torch.CudaTensor')
   mytester:assert(f:type() == 'torch.CudaTensor')
   mytester:asserteq(f:dim(),2)
end
function dptest.imagetensor()
   local size = {8,4,4,3}
   local feature_size = {8,4*4*3}
   local data = torch.rand(unpack(size))
   local axes = {'b','h','w','c'}
   local dt = dp.ImageTensor{data=data, axes=axes}
   -- convert to cuda image (shouldn't change anything)
   local i = dt:image('torch.CudaTensor')
   local data2 = data:transpose(1, 4)
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dt:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assert(t:type() == 'torch.CudaTensor')
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to cuda image (should expand last dim)
   local i = dt:image('torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dt:image()
   mytester:assertTensorEq(i:float(), data:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to cuda image (should expand last dim)
   local i = dt:image('torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dt:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assert(t:type() == 'torch.CudaTensor')
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dt:image()
   mytester:assertTensorEq(i:float(), data:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
end

function dp.testCuda(tests)
   require 'cutorch'
   require 'cunn'
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end
