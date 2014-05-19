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
   local i = dt:imageCHWB('torch.CudaTensor')
   local data2 = data:transpose(1, 4)
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dt:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assert(t:type() == 'torch.CudaTensor')
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to cuda image (should expand last dim)
   local i = dt:imageCHWB('torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dt:imageBHWC()
   mytester:assertTensorEq(i:float(), data:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to cuda image (should expand last dim)
   local i = dt:imageCHWB('torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dt:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assert(t:type() == 'torch.CudaTensor')
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dt:imageBHWC()
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
   mytester:assertTensorEq(mlp_act, act:feature('torch.DoubleTensor'), 0.0001)
   mytester:assertTensorEq(mlp_grad, grad:feature('torch.DoubleTensor'), 0.0001)
end
function dptest.softmaxtree()
   local input_tensor = torch.randn(5,10):float()
   local target_tensor = torch.IntTensor{20,24,27,10,12}
   local grad_tensor = torch.randn(5):float()
   local root_id = 29
   local hierarchy={
      [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   -- dp
   local input = dp.DataTensor{data=input_tensor}
   local target = dp.ClassTensor{data=target_tensor}
   local model = dp.SoftmaxTree{input_size=10, hierarchy=hierarchy, root_id=root_id}
   model:cuda()
   -- nn
   require 'nnx'
   local mlp = nn.SoftMaxTree(10, hierarchy, root_id)
   mlp:float()
   mlp.weight = model._module.weight:float():clone()
   mlp.bias = model._module.bias:float():clone()
   -- forward backward
   --- dp
   local act, carry = model:forward(input, {nSample=5, targets=target})
   local gradWeight = model._module.gradWeight:float()
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTableEq(act:feature():size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(grad:feature():size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model._module.gradWeight:float()
   mytester:assertTensorNe(gradWeight, gradWeight2, 0.00001)
   --- nn
   local mlp_act = mlp:forward{input_tensor, target_tensor}
   local mlp_grad = mlp:backward({input_tensor, target_tensor}, grad_tensor:select(2,1))
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:feature('torch.FloatTensor'), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature('torch.FloatTensor'), 0.00001)
   -- share
   local model2 = model:sharedClone()   
   -- update
   local weight = model._module.weight:clone()
   local act_ten = act:feature():clone()
   local grad_ten = grad:feature():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   model:accept(visitor)
   local weight2 = model._module.weight
   mytester:assertTensorNe(weight, weight2, 0.00001)
   model:doneBatch()
   -- forward backward
   local act2, carry2 = model2:forward(input, {nSample=5, targets=target})
   local grad2, carry2 = model2:backward(dp.DataTensor{data=grad_tensor}, carry2)
   mytester:assertTensorNe(act_ten, act2:feature('torch.FloatTensor'), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:feature('torch.FloatTensor'), 0.00001)
   local act, carry = model:forward(input, {nSample=5, targets=target})
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTensorEq(act:feature('torch.FloatTensor'), act2:feature('torch.FloatTensor'), 0.00001)
   mytester:assertTensorEq(grad:feature('torch.FloatTensor'), grad2:feature('torch.FloatTensor'), 0.00001)
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
function dptest.treenll()
   local input_tensor = torch.randn(5,10):add(100)
   local target_tensor = torch.ones(5) --all targets are 1
   -- dp
   local input = dp.DataTensor{data=input_tensor:narrow(2,1,1)}
   local target = dp.ClassTensor{data=target_tensor}
   local loss = dp.TreeNLL()
   loss:cuda()
   -- the targets are actually ignored (SoftmaxTree uses them before TreeNLL)
   local err, carry = loss:forward(input, target, {nSample=5})
   local grad = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad:narrow(2,1,1):float(), grad:feature():float(), 0.00001)
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
