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
function dptest.datatensor()
   local data = torch.rand(3,4)
   local axes = {'b','f'}
   local sizes = {3, 4}
   local dt = dp.DataTensor{data=data, axes=axes, sizes=sizes}
   local f = dt:feature()
   mytester:asserteq(f:dim(),2)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local fi = dt:index(indices)
   mytester:assertTensorEq(fi:feature(), data:index(1, indices), 0.000001)
   local dt2 = dp.DataTensor{data=torch.zeros(8,4), axes=axes}
   local fi2 = dt:index(dt2, indices)
   mytester:assertTensorEq(fi2:feature(), fi:feature(), 0.0000001)
   mytester:assertTensorEq(dt2._data, fi2:feature(), 0.0000001)
   local fi3 = dt:index(nil, indices)
   mytester:assertTensorEq(fi2:feature(), fi:feature(), 0.0000001)
end
function dptest.imagetensor()
   local size = {3,32,32,3}
   local feature_size = {3,32*32*3}
   local data = torch.rand(unpack(size))
   local axes = {'b','h','w','c'}
   local dt = dp.ImageTensor{data=data, axes=axes}
   -- convert to image (shouldn't change anything)
   local i = dt:image()
   mytester:assertTensorEq(i, data, 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- convert to feature (should colapse last dims)
   local t = dt:feature()
   mytester:assertTableEq(t:size():totable(), feature_size, 0.0001)
   mytester:assertTensorEq(i, data, 0.00001)
   -- convert to image (should expand last dim)
   local i = dt:image()
   mytester:assertTensorEq(i, data, 0.0001)
   mytester:assertTableEq(i:size():totable(), size, 0.0001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local fi = dt:index(indices)
   mytester:assertTensorEq(fi:feature(), data:reshape(unpack(feature_size)):index(1, indices), 0.000001)
   local dt2 = dp.ImageTensor{data=torch.zeros(8,32,32,3), axes=axes}
   local fi2 = dt:index(dt2, indices)
   mytester:assertTensorEq(fi2:feature(), fi:feature(), 0.0000001)
   mytester:assertTensorEq(dt2._data, fi2:feature(), 0.0000001)
   local fi3 = dt:index(nil, indices)
   mytester:assertTensorEq(fi2:feature(), fi:feature(), 0.0000001)
end
function dptest.classtensor()
   local size = {48,4}
   local data = torch.rand(unpack(size))
   local axes = {'b','t'}
   local dt = dp.ClassTensor{data=data, axes=axes}
   local t = dt:multiclass()
   mytester:assertTableEq(t:size():totable(), size, 0.0001)
   mytester:assertTensorEq(t, data, 0.00001)
   local i = dt:class()
   mytester:asserteq(i:dim(),1)
   mytester:assertTableEq(i:size(1), size[1], 0.0001)
   mytester:assertTensorEq(i, data:select(2,1), 0.00001)
   -- indexing
   local indices = torch.LongTensor{2,3}
   local fi = dt:index(indices)
   mytester:assertTensorEq(fi:multiclass(), data:index(1, indices), 0.000001)
   local dt2 = dp.ClassTensor{data=torch.zeros(8,4), axes=axes}
   local fi2 = dt:index(dt2, indices)
   mytester:assertTensorEq(fi2:multiclass(), fi:multiclass(), 0.0000001)
   mytester:assertTensorEq(dt2._data, fi2:multiclass(), 0.0000001)
   local fi3 = dt:index(nil, indices)
   mytester:assertTensorEq(fi2:multiclass(), fi:multiclass(), 0.0000001)
end
function dptest.compositetensor()
   -- class tensor
   local class_size = {8}
   local class_data = torch.randperm(8)
   local classes = {1,2,3,4,5,6,7,8,9,10}
   local class_tensor = dp.ClassTensor{data=class_data, classes=classes}
   -- image tensor
   local image_size = {8,32,32,3}
   local feature_size = {8,32*32*3}
   local image_data = torch.rand(unpack(image_size))
   local image_tensor = dp.ImageTensor{data=image_data}
   -- data tensor
   local data = torch.rand(8,4)
   local data_tensor = dp.DataTensor{data=data}
   -- composite tensor
   local composite_tensor = dp.CompositeTensor{
      components={class_tensor,image_tensor,data_tensor}
   }
   local t = composite_tensor:feature()
   local size = {8,(32*32*3)+10+4}
   mytester:assertTableEq(t:size():totable(), size, 0.0001)
   local c = torch.concat({class_tensor:feature(), image_tensor:feature(), data_tensor:feature()}, 2)
   mytester:assertTensorEq(t, c, 0.00001)
end
function dptest.dataset()
   -- class tensor
   local class_data = torch.randperm(8)
   local classes = {1,2,3,4,5,6,7,8,9,10}
   local class_tensor = dp.ClassTensor{data=class_data, classes=classes}
   -- image tensor
   local image_data = torch.rand(8,32,32,3)
   local image_tensor = dp.ImageTensor{data=image_data}
   -- dataset
   local ds = dp.DataSet{which_set='train', inputs=image_tensor, targets=class_tensor}
   local batch = ds:index(torch.LongTensor{1,2,3})
   local batch2 = ds:sub(1, 3)
   mytester:assertTensorEq(batch:inputs():image(), batch2:inputs():image(), 0.00001)
   batch2 = ds:index(batch, torch.LongTensor{2,3,4})
   mytester:assertTensorEq(batch:inputs():image(), batch2:inputs():image(), 0.00001)
   mytester:assertTensorEq(batch:targets():class(), batch2:targets():class(), 0.00001)
   mytester:assertTensorEq(batch:targets():class(), ds:targets():class():narrow(1,2,3), 0.00001)
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
   mytester:assertTableEq(batch:inputs():context():size():totable(), {3, 3})
   local batch2 = ds:sub(1, 3)
   mytester:assertTensorEq(batch:inputs():context(), batch2:inputs():context(), 0.00001)
   batch2 = ds:index(batch, torch.LongTensor{2,3,4})
   mytester:assertTensorEq(batch:inputs():context(), batch2:inputs():context(), 0.00001)
   mytester:assertTensorEq(batch:targets():class(), batch2:targets():class(), 0.00001)
   mytester:assertTensorEq(batch:targets():class(), tensor:select(2,2):narrow(1,2,3), 0.00001)
end 
function dptest.gcn_zero_vector()
   -- Global Contrast Normalization
   -- Test that passing in the zero vector does not result in
   -- a divide by 0 error
   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.zeros(1, 1)}
   }

   --std_bias = 0.0 is the only value for which there 
   --should be a risk of failure occurring
   local preprocess = dp.GCN{sqrt_bias=0.0, use_std=true}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs(1):data():sum()

   mytester:assert(not _.isNaN(result))
   mytester:assert(_.isFinite(result))
end
function dptest.gcn_unit_norm()
   -- Global Contrast Normalization
   -- Test that using std_bias = 0.0 and use_norm = True
   -- results in vectors having unit norm

   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.rand(3,9)}
   }
   
   local preprocess = dp.GCN{std_bias=0.0, use_std=false}
   dataset:preprocess{input_preprocess=preprocess}
   local result = dataset:inputs(1):data()
   local norms = torch.pow(result, 2):sum(2):sqrt()
   local max_norm_error = torch.abs(norms:add(-1)):max()
   mytester:assert(max_norm_error < 3e-5)
end
function dptest.zca()
   -- Confirm that ZCA.inv_P_ is the correct inverse of ZCA._P.
   local dataset = dp.DataSet{
      which_set='train', inputs=dp.DataTensor{data=torch.randn(15,10)}
   }
   local preprocess = dp.ZCA()
   preprocess._unit_test = true
   dataset:preprocess{input_preprocess=preprocess}
   local function is_identity(matrix)
      local identity = torch.eye(matrix:size(1))
      local abs_diff = torch.abs(identity:add(-matrix))
      return (torch.lt(abs_diff,.00001):int():min() ~= 0)
   end
   local data = dataset:inputs(1):data()
   mytester:assert(table.eq(preprocess._P:size():totable(),{data:size(2),data:size(2)}))
   mytester:assert(not is_identity(preprocess._P))
   mytester:assert(is_identity(preprocess._P*preprocess._inv_P))
end
function dptest.neural()
   local tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5, 2)
   -- dp
   local input = dp.DataTensor{data=tensor}
   local layer = dp.Neural{input_size=10, output_size=2, transfer=nn.Tanh()}
   local act, carry = layer:forward(input, {nSample=5})
   local grad = layer:backward(dp.DataTensor{data=grad_tensor}, carry)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.Linear(10,2)
   m:share(layer._affine, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   local mlp_act = mlp:forward(tensor)
   local mlp_grad = mlp:backward(tensor, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:feature(), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature(), 0.00001)
   -- update
   local act_ten = act:feature():clone()
   local grad_ten = grad:feature():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   layer:accept(visitor)
   layer:doneBatch()
   -- forward backward
   local act2, carry2 = layer:forward(input, {nSample=5})
   local grad2, carry2 = layer:backward(dp.DataTensor{data=grad_tensor}, carry2)
   mytester:assertTensorNe(act_ten, act2:feature(), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:feature(), 0.00001)
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
   local act, carry = model:forward(input, {nSample=5})
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
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
   mytester:assertTensorEq(mlp_act, act:feature(), 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature(), 0.00001)
end
function dptest.softmaxtree()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.IntTensor{20,24,27,10,12}
   local grad_tensor = torch.randn(5)
   local hierarchy={
      [-1]=torch.IntTensor{0,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   -- dp
   local input = dp.DataTensor{data=input_tensor}
   local target = dp.ClassTensor{data=target_tensor}
   local model = dp.SoftmaxTree{input_size=10, hierarchy=hierarchy}
   -- nn
   local concat = nn.ConcatTable()
   local indices = {3,3,4}
   for i,k in ipairs{-1,2,8} do
      local s = nn.Sequential()
      s:add(model._parents[k][1]:clone())
      s:add(nn.Narrow(1,indices[i],1))
      concat:add(s)
   end
   local mlp = nn.Sequential()
   mlp:add(concat)
   mlp:add(nn.CMulTable())
   -- forward backward
   --- dp
   local act, carry = model:forward(input, {nSample=5, targets=target})
   local gradWeight = model:parameters().weight1.grad:clone()
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTableEq(act:feature():size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(grad:feature():size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model:parameters().weight1.grad:clone()
   mytester:assertTensorNe(gradWeight, gradWeight2, 0.00001)
   --- nn
   local mlp_act = mlp:forward(input_tensor[3])
   local mlp_grad = mlp:backward(input_tensor[3], grad_tensor[3])
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, act:feature()[3], 0.00001)
   mytester:assertTensorEq(mlp_grad, grad:feature()[3], 0.00001)
   -- share
   local model2 = model:sharedClone()   
   -- update
   local weight = model:parameters().weight1.param:clone()
   local act_ten = act:feature():clone()
   local grad_ten = grad:feature():clone()
   local visitor = dp.Learn{learning_rate=0.1}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   model:accept(visitor)
   local weight2 = model:parameters().weight1.param:clone()
   mytester:assertTensorNe(weight, weight2, 0.00001)
   model:doneBatch()
   -- forward backward
   local act2, carry2 = model2:forward(input, {nSample=5, targets=target})
   local grad2, carry2 = model2:backward(dp.DataTensor{data=grad_tensor}, carry2)
   mytester:assertTensorNe(act_ten, act2:feature(), 0.00001)
   mytester:assertTensorNe(grad_ten, grad2:feature(), 0.00001)
   local act, carry = model:forward(input, {nSample=5, targets=target})
   local grad, carry = model:backward(dp.DataTensor{data=grad_tensor}, carry)
   mytester:assertTensorEq(act:feature(), act2:feature(), 0.00001)
   mytester:assertTensorEq(grad:feature(), grad2:feature(), 0.00001)
end
function dptest.nll()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.randperm(10):sub(1,5)
   -- dp
   local input = dp.DataTensor{data=input_tensor}
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
   local input_tensor = torch.randn(5,10):add(100)
   local target_tensor = torch.ones(5) --all targets are 1
   -- dp
   local input = dp.DataTensor{data=input_tensor:narrow(2,1,1)}
   local target = dp.ClassTensor{data=target_tensor}
   local loss = dp.TreeNLL()
   -- the targets are actually ignored (SoftmaxTree uses them before TreeNLL)
   local err, carry = loss:forward(input, target, {nSample=5})
   local grad = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion()
   local lg = nn.Log()
   local l_act = lg:forward(input_tensor)
   local c_err = criterion:forward(l_act, target_tensor)
   local c_grad = criterion:backward(l_act, target_tensor)
   local l_grad = lg:backward(input_tensor, c_grad)
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.00001)
   mytester:assertTensorEq(l_grad:narrow(2,1,1), grad:feature(), 0.00001)
end


function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

