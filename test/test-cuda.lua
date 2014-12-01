local mytester 
local dptest = {}
local mediator = dp.Mediator()

function dptest.dataview()
   local data = torch.rand(3,4)
   local sizes = {3, 4}
   local dv = dp.DataView('bf', data)
   local f = dv:forward('bf', 'torch.CudaTensor')
   mytester:assert(f:type() == 'torch.CudaTensor')
   mytester:asserteq(f:dim(),2)
end

function dptest.imageview()
   local size = {8,4,4,3}
   local feature_size = {8,4*4*3}
   local data = torch.rand(unpack(size))
   local dv = dp.ImageView('bhwc', data)
   -- convert to cuda image (shouldn't change anything)
   local i = dv:forward('chwb', 'torch.CudaTensor')
   local data2 = data:transpose(1, 4)
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dv:forward('bf')
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to cuda image (should expand last dim)
   local i = dv:forward('chwb', 'torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dv:forward('bhwc')
   mytester:assertTensorEq(i:float(), data:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to cuda image (should expand last dim)
   local i = dv:forward('chwb', 'torch.CudaTensor')
   mytester:assertTensorEq(i:float(), data2:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), data2:size():totable(), 0.0001)
   -- convert to cuda feature (should colapse last dims)
   local t = dv:forward('bf')
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(t:float(), data:reshape(unpack(feature_size)):float(), 0.00001)
   -- convert to bhwc image (should transpose first and last dim)
   local i = dv:forward('bhwc')
   mytester:assertTensorEq(i:float(), data:float(), 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
end

function dptest.neural()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataView('bf', tensor)
   local layer = dp.Neural{input_size=10, output_size=2, transfer=nn.Tanh()}
   local params, grads = layer:parameters()
   layer:cuda()
   local params2, grads2 = layer:parameters()
   for i=1,#params do
      mytester:assertTensorEq(params2[i]:float(), params[i]:float(), 0.00001)
      mytester:assertTensorEq(grads2[i]:float(), grads2[i]:float(), 0.00001)
   end
   mytester:assert(torch.type(layer._linear.weight) == 'torch.CudaTensor')
   mytester:assert(layer:inputType() == 'torch.CudaTensor')
   mytester:assert(layer:outputType() == 'torch.CudaTensor')
   mytester:assert(layer:moduleType() == 'torch.CudaTensor')
   local output, carry = layer:forward(input, dp.Carry{nSample=5})
   output:backward('bf', grad_tensor:cuda())
   input = layer:backward(output, carry)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.Linear(10,2):cuda()
   m:share(layer._linear, 'weight', 'bias')
   m:double()
   mlp:add(m)
   mlp:add(nn.Tanh())
   local mlp_act = mlp:forward(tensor)
   -- update
   layer:updateParameters(0.1)
   mytester:assertTensorEq(tensor, input:forward('bf', 'torch.DoubleTensor'), 0.00001)
   local mlp_grad = mlp:backwardUpdate(tensor, grad_tensor, 0.1)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act:double(), output:forward('bf', 'torch.DoubleTensor'), 0.00001)
   mytester:assertTensorEq(mlp_grad:double(), input:backward('bf', 'torch.DoubleTensor'), 0.00001)
   mytester:assertTensorEq(layer._linear.weight:double(), m.weight, 0.00001)
   mytester:assertTensorEq(layer._linear.bias:double(), m.bias, 0.00001)
end

function dptest.sequential()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataView('bf', tensor)
   local model = dp.Sequential{
      models = {
         dp.Neural{input_size=10, output_size=4, transfer=nn.Tanh()},
         dp.Neural{input_size=4, output_size=2, transfer=nn.LogSoftMax()}
      }
   }
   model:cuda()
   local output, carry = model:forward(input, dp.Carry{nSample=5})
   output:backward('bf', grad_tensor:cuda())
   input, carry = model:backward(output, carry)
   mytester:assert(carry:getObj('nSample') == 5, "Carry lost an attribute")
   -- nn
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(10,4))
   mlp:get(1):share(model:get(1)._linear:double(), 'weight', 'bias')
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(4,2))
   mlp:get(3):share(model:get(2)._linear:double(), 'weight', 'bias')
   mlp:add(nn.LogSoftMax())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf', 'torch.DoubleTensor'), 0.0001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf', 'torch.DoubleTensor'), 0.0001)
end

function dptest.dictionary()
   local size = {8,10}
   local output_size = {8,10,50}
   local data = torch.randperm(80):resize(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size)):float()
   -- dp
   local input = dp.ClassView('bt', data)
   local layer = dp.Dictionary{dict_size=100, output_size=50}
    -- nn
   local mlp = nn.LookupTable(100,50)
   mlp:share(layer._module, 'weight')
   -- dp
   layer:cuda()
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bwc'):size():totable(), output_size, 0.00001)
   output:backward('bwc', grad_tensor:cuda())
   input = layer:backward(output, carry)
   -- should be able to get input gradients
   local function f() 
      input:backward('bt') 
   end 
   mytester:assert(not pcall(f))
   -- nn
   mlp:float()
   local mlp_act = mlp:forward(input:forward('bt'))
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bwc'):float(), 0.00001)
   -- update
   local act_ten = output:forward('bwc'):clone()
   layer:updateParameters(0.1)
   -- forward backward
   output, carry2 = layer:forward(input, dp.Carry{nSample=5})
   output:backward('bwc', grad_tensor:cuda())
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten:float(), output:forward('bwc'):float(), 0.00001)
end

function dptest.convolution2D()
   local size = {8,32,32,3}
   local output_size = {8,32,15,15}
   local data = torch.rand(unpack(size)):float()
   local grad_tensor = torch.randn(unpack(output_size)):float()
   -- dp
   local input = dp.ImageView('bhwc', data)
   local layer = dp.Convolution2D{
      input_size=3, output_size=32, kernel_size={3,3}, 
      kernel_stride={1,1}, pool_size={2,2}, pool_stride={2,2},
      transfer=nn.Tanh()
   }
   layer:cuda()
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bchw'):size():totable(), output_size, 0.00001)
   output:backward('bchw', grad_tensor:cuda())
   input = layer:backward(output, carry)
   mytester:assertTableEq(input:backward('bhwc'):size():totable(), size, 0.00001)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.SpatialConvolutionMM(3,32,3,3,1,1)
   m:cuda()
   m:share(layer._conv, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   mlp:add(nn.SpatialMaxPooling(2,2,2,2))
   mlp:cuda()
   local mlp_act = mlp:forward(input:forward('bchw', 'torch.CudaTensor'))
   local mlp_grad = mlp:backward(input:forward('bchw', 'torch.CudaTensor'), grad_tensor:cuda())
   -- compare nn and dp
   mlp_grad = dp.ImageView('bchw', mlp_grad):forward('bhwc', 'torch.FloatTensor')
   mytester:assertTensorEq(mlp_act:float(), output:forward('bchw', 'torch.FloatTensor'), 0.00001)
   mytester:assertTableEq(mlp_grad:size():totable(), input:backward('bhwc'):size():totable(), 0.00001)
   mytester:assertTensorEq(mlp_grad:float(), input:backward('bhwc', 'torch.FloatTensor'), 0.00001)
   -- update
   local act_ten = output:forward('bhwc', 'torch.FloatTensor'):clone()
   local grad_ten = input:backward('bhwc', 'torch.FloatTensor'):clone()
   layer:updateParameters(0.1)
   layer:doneBatch()
   -- forward backward
   output, carry2 = layer:forward(input, dp.Carry{nSample=8})
   output:backward('bchw', grad_tensor:cuda())
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten, output:forward('bhwc', 'torch.FloatTensor'), 0.00001)
   mytester:assertTensorNe(grad_ten, input:backward('bhwc', 'torch.FloatTensor'), 0.00001)
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
   local input = dp.DataView()
   input:forward('bf', input_tensor)
   local target = dp.ClassView()
   target:forward('b', target_tensor)
   local model = dp.SoftmaxTree{input_size=10, hierarchy=hierarchy, root_id=root_id}
   model:float()
   -- nn
   local mlp = nn.SoftMaxTree(10, hierarchy, root_id)
   mlp.weight = model._module.weight:clone()
   mlp.bias = model._module.bias:clone()
   mlp:float()
   -- forward backward
   --- dp
   model:cuda()
   local output, carry = model:forward(input, dp.Carry{nSample=5, targets=target})
   local gradWeight = model._module.gradWeight:clone()
   output:backward('b', grad_tensor:cuda())
   input, carry = model:backward(output, carry)
   cutorch.synchronize()
   mytester:assertTableEq(output:forward('bf'):size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model._module.gradWeight
   mytester:assertTensorNe(gradWeight:float(), gradWeight2:float(), 0.00001)
   --- nn
   local mlp_act = mlp:forward{input_tensor, target_tensor}
   local mlp_grad = mlp:backward({input_tensor, target_tensor}, grad_tensor)[1]
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'):float(), 0.001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'):float(), 0.001)
   -- share
   local model2 = model:sharedClone()
   -- update
   local weight = model._module.weight:clone()
   local act_ten = output:forward('bf'):clone()
   local grad_ten = input:backward('bf'):clone()
   model:updateParameters(0.1)
   local weight2 = model._module.weight
   mytester:assertTensorNe(weight:float(), weight2:float(), 0.00001)
   model:doneBatch()
   -- forward backward
   local output2, carry2 = model2:forward(input:clone(), dp.Carry{nSample=5, targets=target})
   output2:backward('b', grad_tensor:cuda())
   local input2, carry2 = model2:backward(output2, carry2)
   mytester:assertTensorNe(act_ten:float(), output2:forward('bf'):float(), 0.00001)
   mytester:assertTensorNe(grad_ten:float(), input2:backward('bf'):float(), 0.00001)
   local output, carry = model:forward(input2:clone(), dp.Carry{nSample=5, targets=target})
   output:backward('b', grad_tensor:cuda())
   local input, carry = model:backward(output, carry)
   mytester:assertTensorEq(output:forward('bf'):float(), output2:forward('bf'):float(), 0.00001)
   mytester:assertTensorEq(input:backward('bf'):float(), input2:backward('bf'):float(), 0.00001)
end

function dptest.blocksparse()
   local inputSize = 23
   local nBlock = {12, 14, 16}
   local hiddenSize = {64, 32, 64}
   local gaterSize = {17}
   local windowSize = {4, 8, 4}
   local outputSize = 15
   local batchSize = 8
   local accUpdate = true
   
   local input_tensor = torch.randn(batchSize, inputSize):cuda()
   local gradOutput_tensor = torch.randn(batchSize, outputSize):cuda()
   local input = dp.DataView()
   input:forward('bf', input_tensor)

   local model = dp.BlockSparse{
      input_size=inputSize, output_size=outputSize, hidden_size=hiddenSize, n_block=nBlock,
      gater_size=gaterSize, window_size=windowSize, noise_std={1,1}, acc_update=accUpdate
   }
   model:cuda()
   
   local output, carry = model:forward(input, dp.Carry{nSample=batchSize})
   local params = model:parameters()
   params = _.map(params, function(k,v) return v:float() end )
   output:backward('bf', gradOutput_tensor:cuda())
   input, carry = model:backward(output, carry)
   mytester:assertTableEq(output:forward('bf'):size():totable(), {batchSize,outputSize}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {batchSize,inputSize}, 0.000001, "Wrong grad size")
   model:updateParameters(0.1)
   local params2 = model:parameters()
   for i=1,#params do
      mytester:assertTensorNe(params[i]:float(), params2[i]:float(), 0.00001)
   end
end

function dptest.nll()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.randperm(10):sub(1,5)
   -- dp
   local input = dp.DataView('bf', input_tensor)
   local target = dp.ClassView('b', target_tensor)
   local loss = dp.NLL{size_average=false} -- else loss isn't avg
   -- this shouldn't change anything since nn.ClassNLLCriterion doesn't work with cuda
   loss:cuda()
   local err, carry = loss:forward(input, target, dp.Carry{nSample=5})
   input = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad:float(), input:backward('bf'):float(), 0.00001)
end

function dptest.treenll()
   local input_tensor = torch.randn(5,10):add(100)
   local target_tensor = torch.ones(5) --all targets are 1
   -- dp
   local input = dp.DataView('b', input_tensor:select(2,1))
   local target = dp.ClassView('b', target_tensor)
   local loss = dp.TreeNLL{size_average=false} -- else loss isn't avg
   loss:cuda()
   -- the targets are actually ignored (SoftmaxTree uses them before TreeNLL)
   local err, carry = loss:forward(input, target, dp.Carry{nSample=5})
   input = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad:select(2,1):float(), input:backward('b'):float(), 0.00001)
end

function dp.testCuda(tests)
   require 'cutorch'
   require 'cunn'
   require 'cunnx'
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end
