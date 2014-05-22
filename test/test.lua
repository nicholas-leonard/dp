local mytester 
local dptest = {}
local mediator = dp.Mediator()

function dptest.uid()
   local uid1 = dp.uniqueID()
   mytester:asserteq(type(uid1), 'string', 'type(uid1) == string')
   local uid2 = dp.uniqueID()
   local uid3 = dp.uniqueID('mynamespace')
   mytester:assertne(uid1, uid2, 'uid1 ~= uid2')
   mytester:assertne(uid2, uid3, 'uid2 ~= uid3')
end
function dptest.dataview()
   local data = torch.rand(3,4)
   local sizes = {3, 4}
   local v = dp.DataView()
   -- forward
   v:forward('bf', data)
   local f = v:forward('bf', 'torch.DoubleTensor')
   mytester:asserteq(f:dim(),2)
   mytester:assertTensorEq(f,data, 0.00001)
   local f2 = v:forward('bf', 'torch.FloatTensor')
   mytester:assertTensorEq(f:float(),f2, 0.00001)
   -- backward
   local g = f:clone()
   g[1][1] = 0
   local g2 = f2:clone()
   g2[1][2] = 0
   v:backward('bf', g)
   v:backward('bf', g2)
   local b = v:backward('bf', 'torch.DoubleTensor')
   mytester:asserteq(b:dim(),2)
   local r = g2:clone():double()
   r:add(g)
   mytester:assertTensorEq(b,r, 0.00001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local v2 = v:index(indices)
   mytester:assertTensorEq(v2:forward('bf', 'torch.DoubleTensor'), data:index(1, indices), 0.000001)
   local v3 = dp.DataView()
   v3:forward('bf', torch.zeros(8,4))
   local v3 = v:index(v3, indices)
   mytester:assertTensorEq(v2:forward('bf', 'torch.DoubleTensor'), v3:forward('bf', 'torch.DoubleTensor'), 0.0000001)
   mytester:assertTensorEq(v3._input, v2:forward('bf', 'torch.DoubleTensor'), 0.0000001)
   local v4 = v:index(nil, indices)
   mytester:assertTensorEq(v4:forward('bf', 'torch.DoubleTensor'), v2:forward('bf', 'torch.DoubleTensor'), 0.0000001)
end
function dptest.imageview()
   local size = {8,32,32,3}
   local feature_size = {8,32*32*3}
   local data = torch.rand(unpack(size))
   local v = dp.ImageView()
   v:forward('bhwc', data)
   -- convert to image (shouldn't change anything)
   local i = v:forward('bhwc', 'torch.DoubleTensor')
   mytester:assertTensorEq(i, data, 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to feature (should colapse last dims)
   local f = v:forward('bf', 'torch.DoubleTensor')
   mytester:assertTableEq(f:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(f, data, 0.00001)
   -- convert to image (shouldn't change anything)
   local i2 = v:forward('bhwc', 'torch.DoubleTensor')
   mytester:assertTensorEq(i2, data, 0.0001)
   mytester:assertTableEq(i2:size():totable(), size, 0.0001)
   -- convert to image bchw
   local i3 = v:forward('bchw', 'torch.DoubleTensor')
   mytester:assertTensorEq(i3, data:transpose(2,4):transpose(3,4), 0.0001)
   mytester:assertTableEq(i3:size():totable(), {8,3,32,32}, 0.0001)
   -- create from bchw
   local v2 = dp.ImageView()
   v2:forward('bchw', i3)
   local i4 = v:forward('bhwc', 'torch.DoubleTensor')
   mytester:assertTensorEq(i4, data, 0.0001)
   mytester:assertTableEq(i4:size():totable(), size, 0.0001)
   -- convert to bchw()
   local i5 = v:forward('bchw', 'torch.DoubleTensor')
   mytester:assertTensorEq(i5, data:transpose(2,4):transpose(3,4), 0.0001)
   mytester:assertTableEq(i5:size():totable(), {8,3,32,32}, 0.0001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local v3 = v:index(indices)
   mytester:assertTensorEq(v3:forward('bf', 'torch.DoubleTensor'), data:reshape(unpack(feature_size)):index(1, indices), 0.000001)
   local v4 = dp.ImageView()
   v4:forward('bhwc', torch.zeros(2,32,32,3))
   v:index(v4, indices)
   mytester:assertTensorEq(v4:forward('bf', 'torch.DoubleTensor'), v3:forward('bf', 'torch.DoubleTensor'), 0.0000001)
   mytester:assertTensorEq(v4._input, v3:forward('bhwc', 'torch.DoubleTensor'), 0.0000001)
   local v5 = v:index(nil, indices)
   mytester:assertTensorEq(v5:forward('bhwc', 'torch.DoubleTensor'), v._input:index(1, indices), 0.0000001)
end
function dptest.sequenceview()
   local size = {8,10,50}
   local feature_size = {8,10*50}
   local data = torch.rand(unpack(size))
   local v = dp.SequenceView()
   v:forward('bwc', data)
   -- convert to sequence (shouldn't change anything)
   local s = v:forward('bwc', 'torch.DoubleTensor')
   mytester:assertTensorEq(s, data, 0.0001)
   mytester:assertTableEq(s:size():totable(), size, 0.0001)
   -- convert to feature (should colapse last dims)
   local f = v:forward('bf', 'torch.DoubleTensor')
   mytester:assertTableEq(f:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(f, data, 0.00001)
   -- convert to sequence (should expand last dim)
   local s2 = v:forward('bwc', 'torch.DoubleTensor')
   mytester:assertTensorEq(s2, data, 0.0001)
   mytester:assertTableEq(s2:size():totable(), size, 0.0001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local v2 = v:index(indices)
   mytester:assertTensorEq(v2:forward('bf', 'torch.DoubleTensor'), data:reshape(unpack(feature_size)):index(1, indices), 0.000001)
   local v3 = dp.SequenceView()
   v3:forward('bwc', torch.zeros(8,10,50))
   local v4 = v:index(v3, indices)
   mytester:assertTensorEq(v4:forward('bf', 'torch.DoubleTensor'), v2:forward('bf', 'torch.DoubleTensor'), 0.0000001)
   mytester:assertTensorEq(v2._input, v4:forward('bwc', 'torch.DoubleTensor'), 0.0000001)
   local v5 = v:index(nil, indices)
   mytester:assertTensorEq(v4:forward('bf', 'torch.DoubleTensor'), v5:forward('bf', 'torch.DoubleTensor'), 0.0000001)
end
function dptest.classview()
   local size = {48,4}
   local data = torch.rand(unpack(size))
   local v = dp.ClassView()
   v:forward('bt', data)
   -- multiclass
   local c = v:forward('bt', 'torch.DoubleTensor')
   mytester:assertTableEq(c:size():totable(), size, 0.0001)
   mytester:assertTensorEq(c, data, 0.00001)
   local c2 = v:forward('b', 'torch.DoubleTensor')
   mytester:asserteq(c2:dim(),1)
   mytester:assertTableEq(c2:size(1), size[1], 0.0001)
   mytester:assertTensorEq(c2, data:select(2,1), 0.00001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local v2 = v:index(indices)
   mytester:assertTensorEq(v2:forward('bt', 'torch.DoubleTensor'), data:index(1, indices), 0.000001)
   local v3 = dp.ClassView()
   v3:forward('bt', torch.zeros(8,4))
   local v4 = v:index(v3, indices)
   mytester:assertTensorEq(v4:forward('bt', 'torch.DoubleTensor'), v2:forward('bt', 'torch.DoubleTensor'), 0.0000001)
   mytester:assertTensorEq(v3._input, v4:forward('bt', 'torch.DoubleTensor'), 0.0000001)
   local v5 = v:index(nil, indices)
   mytester:assertTensorEq(v5:forward('bt', 'torch.DoubleTensor'), v2:forward('bt', 'torch.DoubleTensor'), 0.0000001)
   mytester:assertTableEq(v:classes(), v5:classes())
end
function dptest.listview()
   -- image tensor
   local image_size = {8,32,32,3}
   local feature_size = {8,32*32*3}
   local image_data = torch.rand(unpack(image_size))
   local image_v = dp.ImageView()
   -- data tensor
   local data = torch.rand(8,4)
   local data_v = dp.DataView()
   -- composite tensor
   local list_v = dp.ListView({image_v,data_v})
   list_v:forward({'bhwc', 'bf'}, {image_data, data}) 
   local t = list_v:forward('bf', 'torch.FloatTensor')
   local size = {8,(32*32*3)+4}
   mytester:assertTableEq(t:size():totable(), size, 0.0001)
   local c = torch.concat({
      image_v:forward('bf', 'torch.FloatTensor'), 
      data_v:forward('bf', 'torch.FloatTensor')
   }, 2)
   mytester:assertTensorEq(t, c, 0.00001)
end
function dptest.dataset()
   -- class tensor
   local class_data = torch.randperm(8)
   local classes = {1,2,3,4,5,6,7,8,9,10}
   local class_v = dp.ClassView()
   class_v:forward('b', class_data)
   class_v:setClasses(classes)
   -- image tensor
   local image_data = torch.rand(8,32,32,3)
   local image_v = dp.ImageView()
   image_v:forward('bhwc', image_data)
   -- dataset
   local ds = dp.DataSet{which_set='train', inputs=image_v, targets=class_v}
   local batch = ds:index(torch.LongTensor{1,2,3})
   local batch2 = ds:sub(1, 3)
   mytester:assertTensorEq(batch:inputs():forward('bhwc'), batch2:inputs():forward('bhwc'), 0.00001)
   batch2 = ds:index(batch, torch.LongTensor{2,3,4})
   mytester:assertTensorEq(batch:inputs():forward('bhwc'), batch2:inputs():forward('bhwc'), 0.00001)
   mytester:assertTensorEq(batch:targets():forward('bt'), batch2:targets():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch:targets():forward('bt'), ds:targets():forward('bt'):narrow(1,2,3), 0.00001)
end
function dptest.sentenceset()
   local tensor = torch.IntTensor(80, 2):zero()
   for i=1,8 do
      -- one sentence
      local sentence = tensor:narrow(1, ((i-1)*10)+1, 10)
      -- fill it with a sequence of words
      sentence:select(2,2):copy(torch.randperm(17):sub(1,10))
      -- fill it with start sentence delimiters
      sentence:select(2,1):fill(((i-1)*10)+1)
   end
   -- 18 words in vocabulary ("<S>" isn't found in tensor since its redundant to "</S>")
   local words = {"</S>", "<UNK>", "the", "it", "is", "to", "view", "huh", "hi", "ho", "oh", "I", "you", "we", "see", "do", "have", "<S>"}
   -- dataset
   local ds = dp.SentenceSet{which_set='train', data=tensor, words=words, context_size=3, start_id=1, end_id=2}
   local batch = ds:index(torch.LongTensor{1,2,3})
   mytester:assertTableEq(batch:inputs():forward('bt'):size():totable(), {3, 3})
   local batch2 = ds:sub(1, 3)
   mytester:assertTensorEq(batch:inputs():forward('bt'), batch2:inputs():forward('bt'), 0.00001)
   batch2 = ds:index(batch, torch.LongTensor{2,3,4})
   mytester:assertTensorEq(batch:inputs():forward('bt'), batch2:inputs():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch:targets():forward('b'), batch2:targets():forward('b'), 0.00001)
   mytester:assertTensorEq(batch:targets():forward('b'), tensor:select(2,2):narrow(1,2,3), 0.00001)
end 
function dptest.gcn()
   --[[ zero_vector ]]--
   -- Global Contrast Normalization
   -- Test that passing in the zero vector does not result in
   -- a divide by 0 error
   local dv = dp.DataView()
   dv:forward('bf', torch.zeros(1,1))
   local dataset = dp.DataSet{which_set='train', inputs=dv}

   --std_bias = 0.0 is the only value for which there 
   --should be a risk of failure occurring
   local preprocess = dp.GCN{sqrt_bias=0.0, use_std=true}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs():input():sum()

   mytester:assert(not _.isNaN(result))
   mytester:assert(_.isFinite(result))

   --[[ unit_norm ]]--
   -- Global Contrast Normalization
   -- Test that using std_bias = 0.0 and use_norm = True
   -- results in vectors having unit norm
   local dv = dp.DataView('bf', torch.rand(3,9))
   local dataset = dp.DataSet{which_set='train', inputs=dv}
   
   local preprocess = dp.GCN{std_bias=0.0, use_std=false}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs():input()
   local norms = torch.pow(result, 2):sum(2):sqrt()
   local max_norm_error = torch.abs(norms:add(-1)):max()
   mytester:assert(max_norm_error < 3e-5)
end
function dptest.zca()
   -- Confirm that ZCA.inv_P_ is the correct inverse of ZCA._P.
   local dv = dp.DataView('bf', torch.randn(15,10))
   local dataset = dp.DataSet{which_set='train', inputs=dv}
   local preprocess = dp.ZCA()
   preprocess._unit_test = true
   dataset:preprocess{input_preprocess=preprocess}
   local function is_identity(matrix)
      local identity = torch.eye(matrix:size(1))
      local abs_diff = torch.abs(identity:add(-matrix))
      return (torch.lt(abs_diff,.00001):int():min() ~= 0)
   end
   local data = dataset:inputs():input()
   mytester:assert(table.eq(preprocess._P:size():totable(),{data:size(2),data:size(2)}))
   mytester:assert(not is_identity(preprocess._P))
   mytester:assert(is_identity(preprocess._P*preprocess._inv_P))
end
function dptest.neural()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local layer = dp.Neural{input_size=10, output_size=2, transfer=nn.Tanh()}
   local input = dp.DataView()
   input:forward('bf', tensor)
   local output, carry = layer:forward(input, {nSample=5})
   output:backward('bf', grad_tensor)
   input = layer:backward(output, carry)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.Linear(10,2)
   m:share(layer._affine, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'), 0.00001)
   -- update
   local act_ten = output:forward('bf'):clone()
   local grad_ten = input:backward('bf'):clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   layer:doneBatch()
   -- forward backward
   output, carry2 = layer:forward(input, {nSample=5})
   output:backward('bf', grad_tensor)
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten, output:forward('bf'), 0.00001)
   mytester:assertTensorNe(grad_ten, input:backward('bf'), 0.00001)
end
function dptest.sequential()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataView()
   input:forward('bf', tensor)
   local model = dp.Sequential{
      models = {
         dp.Neural{input_size=10, output_size=4, transfer=nn.Tanh()},
         dp.Neural{input_size=4, output_size=2, transfer=nn.LogSoftMax()}
      }
   }
   local output, carry = model:forward(input, {nSample=5})
   output:backward('bf', grad_tensor)
   input, carry = model:backward(output, carry)
   mytester:assert(carry.nSample == 5, "Carry lost an attribute")
   -- nn
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(10,4))
   mlp:get(1):share(model:get(1)._affine, 'weight', 'bias')
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(4,2))
   mlp:get(3):share(model:get(2)._affine, 'weight', 'bias')
   mlp:add(nn.LogSoftMax())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'), 0.00001)
end
function dptest.softmaxtree()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.IntTensor{20,24,27,10,12}
   local grad_tensor = torch.randn(5)
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
   -- nn
   require 'nnx'
   local mlp = nn.SoftMaxTree(10, hierarchy, root_id)
   mlp.weight = model._module.weight:clone()
   mlp.bias = model._module.bias:clone()
   -- forward backward
   --- dp
   local output, carry = model:forward(input, {nSample=5, targets=target})
   local gradWeight = model._module.gradWeight:clone()
   output:backward('b', grad_tensor)
   input, carry = model:backward(output, carry)
   mytester:assertTableEq(output:forward('bf'):size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model._module.gradWeight
   mytester:assertTensorNe(gradWeight, gradWeight2, 0.00001)
   --- nn
   local mlp_act = mlp:forward{input_tensor, target_tensor}
   local mlp_grad = mlp:backward({input_tensor, target_tensor}, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'), 0.00001)
   -- share
   local model2 = model:sharedClone()   
   -- update
   local weight = model._module.weight:clone()
   local act_ten = output:forward('bf'):clone()
   local grad_ten = input:backward('bf'):clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   model:accept(visitor)
   local weight2 = model._module.weight
   mytester:assertTensorNe(weight, weight2, 0.00001)
   model:doneBatch()
   -- forward backward
   local output2, carry2 = model2:forward(input:clone(), {nSample=5, targets=target})
   output2:backward('b', grad_tensor)
   local input2, carry2 = model2:backward(output2, carry2)
   mytester:assertTensorNe(act_ten, output2:forward('bf'), 0.00001)
   mytester:assertTensorNe(grad_ten, input2:backward('bf'), 0.00001)
   local output, carry = model:forward(input2:clone(), {nSample=5, targets=target})
   output:backward('b', grad_tensor)
   local input, carry = model:backward(output, carry)
   mytester:assertTensorEq(output:forward('bf'), output2:forward('bf'), 0.00001)
   mytester:assertTensorEq(input:backward('bf'), input2:backward('bf'), 0.00001)
end
function dptest.convolution1D()
   local size = {8,10,50}
   local output_size = {8,4,100}
   local data = torch.rand(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   -- dp
   local input = dp.SequenceTensor{data=data}
   local layer = dp.Convolution1D{
      input_size=50, output_size=100, kernel_size=2, 
      kernel_stride=1, pool_size=2, pool_stride=2,
      transfer=nn.Tanh()
   }
   local act, carry = layer:forward(input, {nSample=8})
   mytester:assertTableEq(act:conv1D():size():totable(), output_size, 0.00001)
   local grad = layer:backward(dp.SequenceTensor{data=grad_tensor}, carry)
   mytester:assertTableEq(grad:conv1D():size():totable(), size, 0.00001)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.TemporalConvolution(50,100,2,1)
   m:share(layer._conv, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   mlp:add(nn.TemporalMaxPooling(2,2))
   local mlp_act = mlp:forward(input:conv1D())
   local mlp_grad = mlp:backward(input:conv1D(), grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:conv1D(), 0.00001)
   mytester:assertTableEq(mlp_grad:size():totable(), grad:conv1D():size():totable(), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:conv1D(), 0.00001)
   -- update
   local act_ten = act:expand():clone()
   local grad_ten = grad:expand():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   layer:doneBatch()
   -- forward backward
   local act2, carry2 = layer:forward(input, {nSample=8})
   local grad2, carry2 = layer:backward(dp.SequenceTensor{data=grad_tensor}, carry2)
   mytester:assertTensorNe(act_ten, act2:sequence(), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:sequence(), 0.00001)
end
function dptest.convolution2D()
   local size = {8,32,32,3}
   local output_size = {8,20,15,15}
   local data = torch.rand(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   local axes = {'b','h','w','c'}
   local output_axes = {'b','c','h','w'}
   -- dp
   local input = dp.ImageTensor{data=data, axes=axes}
   local layer = dp.Convolution2D{
      input_size=3, output_size=20, kernel_size={3,3}, 
      kernel_stride={1,1}, pool_size={2,2}, pool_stride={2,2},
      transfer=nn.Tanh()
   }
   local act, carry = layer:forward(input, {nSample=8})
   mytester:assertTableEq(act:conv2D():size():totable(), output_size, 0.00001)
   local grad = layer:backward(dp.ImageTensor{data=grad_tensor,axes=output_axes}, carry)
   mytester:assertTableEq(grad:conv2D():size():totable(), {8,3,32,32}, 0.00001)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.SpatialConvolution(3,20,3,3,1,1)
   m:share(layer._conv, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   mlp:add(nn.SpatialMaxPooling(2,2,2,2))
   local mlp_act = mlp:forward(input:imageBCHW())
   local mlp_grad = mlp:backward(input:imageBCHW(), grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:conv2D(), 0.00001)
   mytester:assertTableEq(mlp_grad:size():totable(), grad:conv2D():size():totable(), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:conv2D(), 0.00001)
   -- update
   local act_ten = act:image():clone()
   local grad_ten = grad:image():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   layer:doneBatch()
   -- forward backward
   local act2, carry2 = layer:forward(input, {nSample=8})
   local grad2, carry2 = layer:backward(dp.ImageTensor{data=grad_tensor,axes=output_axes}, carry2)
   mytester:assertTensorNe(act_ten, act2:image(), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:image(), 0.00001)
end
function dptest.dictionary()
   local size = {8,10}
   local output_size = {8,10,50}
   local data = torch.randperm(80):resize(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   local axes = {'b','t'}
   local output_axes = {'b','s','f'}
   -- dp
   local input = dp.WordTensor{data=data, axes=axes}
   local layer = dp.Dictionary{dict_size=100, output_size=50}
   local act, carry = layer:forward(input, {nSample=8})
   mytester:assertTableEq(act:conv1D():size():totable(), output_size, 0.00001)
   local grad = layer:backward(dp.SequenceTensor{data=grad_tensor,axes=output_axes}, carry)
   mytester:assert(grad == nil)
   -- nn
   local mlp = nn.LookupTable(100,50)
   mlp:share(layer._module, 'weight')
   local mlp_act = mlp:forward(input:context())
   mlp:backward(input:context(), grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:conv1D(), 0.00001)
   -- update
   local act_ten = act:expand():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   layer:doneBatch()
   -- forward backward
   local act2, carry2 = layer:forward(input, {nSample=5})
   local grad2, carry2 = layer:backward(dp.SequenceTensor{data=grad_tensor,axes=output_axes}, carry2)
   mytester:assertTensorNe(act_ten, act2:conv1D(), 0.00001)
end
function dptest.nll()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.randperm(10):sub(1,5)
   -- dp
   local input = dp.DataView{data=input_tensor}
   local target = dp.ClassTensor{data=target_tensor}
   local loss = dp.NLL()
   local err, carry = loss:forward(input, target, {nSample=5})
   local grad = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad, grad:feature(), 0.00001)
end
function dptest.treenll()
   local input_tensor = torch.randn(5,10):add(100) -- add for log nans
   local target_tensor = torch.ones(5) --all targets are 1
   -- dp
   local input = dp.DataView{data=input_tensor:narrow(2,1,1)}
   local target = dp.ClassTensor{data=target_tensor}
   local loss = dp.TreeNLL()
   -- the targets are actually ignored (SoftmaxTree uses them before TreeNLL)
   local err, carry = loss:forward(input, target, {nSample=5})
   local grad = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local c_err = criterion:forward(input_tensor, target_tensor)
   local c_grad = criterion:backward(input_tensor, target_tensor)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(c_grad:narrow(2,1,1), grad:feature(), 0.00001)
end


function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

