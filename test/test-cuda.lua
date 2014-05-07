local mytester 
local dptest = {}
local mediator = dp.Mediator()

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
function dptest.neural()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataTensor{data=tensor}
   local layer = dp.Neural{input_size=10, output_size=2, transfer=nn.Tanh()}
   local params = layer:parameters()
   layer:cuda()
   for param_name, param_table in pairs(layer:parameters()) do
      mytester:assertTensorEq(param_table.param:float(), params[param_name].param:float(), 0.00001)
      mytester:assertTensorEq(param_table.grad:float(), params[param_name].grad:float(), 0.00001)
   end
   mytester:assert(torch.type(layer._affine.weight) == 'torch.CudaTensor')
   mytester:assert(layer:inputType() == 'torch.CudaTensor')
   mytester:assert(layer:outputType() == 'torch.CudaTensor')
   mytester:assert(layer:moduleType() == 'torch.CudaTensor')
   local act, carry = layer:forward(input, {nSample=5})
   local grad = layer:backward(dp.DataTensor{data=grad_tensor}, carry)
   layer:setup{mediator=mediator, id=dp.ObjectID('layer')}
   -- nn
   local mlp = nn.Sequential()
   local m = nn.Linear(10,2):cuda()
   m:share(layer._affine, 'weight', 'bias')
   m:double()
   mlp:add(m)
   mlp:add(nn.Tanh())
   local mlp_act = mlp:forward(tensor)
   -- update
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   mytester:assertTensorEq(tensor, input:feature('torch.DoubleTensor'), 0.00001)
   local mlp_grad = mlp:backwardUpdate(tensor, grad_tensor, 0.1)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act:double(), act:feature('torch.DoubleTensor'), 0.00001)
   mytester:assertTensorEq(mlp_grad:double(), grad:feature('torch.DoubleTensor'), 0.00001)
   params = layer:parameters()
   mytester:assertTensorEq(params.weight.param:double(), m.weight, 0.00001)
   mytester:assertTensorEq(params.bias.param:double(), m.bias, 0.00001)
end
function dptest.sequential()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataTensor{data=tensor}
   local model = dp.Sequential{
      models = {
         dp.Neural{input_size=10, output_size=4, transfer=nn.Tanh()},
         dp.Neural{input_size=4, output_size=2, transfer=nn.LogSoftMax()}
      }
   }
   model:cuda()
   local act, carry = model:forward(input, {nSample=5})
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assert(carry.nSample == 5, "Carry lost an attribute")
   -- nn
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(10,4))
   mlp:get(1):share(model:get(1)._affine:double(), 'weight', 'bias')
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(4,2))
   mlp:get(3):share(model:get(2)._affine:double(), 'weight', 'bias')
   mlp:add(nn.LogSoftMax())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:feature('torch.DoubleTensor'), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature('torch.DoubleTensor'), 0.00001)
end
function dptest.nll()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.randperm(10):sub(1,5)
   -- dp
   local input = dp.DataTensor{data=input_tensor}
   local target = dp.ClassTensor{data=target_tensor}
   local loss = dp.NLL()
   -- this shouldn't change anything since nn.ClassNLLCriterion doesn't work with cuda
   loss:cuda()
   local err, carry = loss:forward(input, target, {nSample=5})
   local grad = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad:float(), grad:feature('torch.FloatTensor'), 0.00001)
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
