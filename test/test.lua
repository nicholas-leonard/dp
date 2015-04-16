local mytester 
local dptest = {}

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
   -- sub
   local v5 = v:sub(2,3)
   mytester:assertTensorEq(v2:forward('bf', 'torch.DoubleTensor'), v5:forward('bf', 'torch.DoubleTensor'), 0.000001)
   local v6 = v:sub(nil, 2,3)
   mytester:assertTensorEq(v2:forward('bf', 'torch.DoubleTensor'), v6:forward('bf', 'torch.DoubleTensor'), 0.000001)
   v:sub(v6, 1, 2)
   mytester:assertTensorEq(v6:forward('bf', 'torch.DoubleTensor'), data:sub(1,2), 0.000001)
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
   mytester:assertTableEq(t[1]:size():totable(), feature_size, 0.0001)
   local list_v = dp.ListView({dp.ImageView(), dp.ImageView()})
   list_v:forward('bhwc', {image_data, image_data:clone():add(1)})
   local t = list_v:forward('bchw')
   mytester:assertTensorEq(t[1], image_data:transpose(4,2):transpose(3,4), 0.00001)
   
   local list = dp.ListView{dp.ImageView('bchw',torch.randn(1,2,3,4)), dp.ImageView('bchw',torch.randn(1,2,3,4))}
   local t = list:forwardGet('bhwc')
   mytester:assertTableEq(t[1]:size():totable(), {1,3,4,2}, 0.00001)
   list:forwardPut('bhwc',{torch.randn(1,3,4,2),torch.randn(1,3,4,2)})
   local t = list:forwardGet('bchw')
   mytester:assertTableEq(t[1]:size():totable(), {1,2,3,4}, 0.00001)
   -- indexing
   local data1, data2 = torch.randn(5,2,3,4), torch.randn(5,2,3,4)
   local v = dp.ListView{dp.ImageView('bchw',data1), dp.ImageView('bchw',data2)}
   local indices = torch.LongTensor{2,3}
   local v2 = v:index(indices)
   local tbl = v2:forward('bchw', 'torch.DoubleTensor')
   local tbl2 = {data1, data2}
   mytester:assert(#tbl == 2)
   for i, d in ipairs(tbl) do
      mytester:assertTensorEq(d, tbl2[i]:index(1, indices), 0.000001)
   end
   local v3 = dp.ListView{dp.ImageView('bchw',torch.randn(1,2,3,4)), dp.ImageView('bchw',torch.randn(1,2,3,4))}
   local v4 = v:index(v3, indices)
   local tbl = v4:forward('bchw', 'torch.DoubleTensor')
   mytester:assert(#tbl == 2)
   for i, d in ipairs(tbl) do
      mytester:assertTensorEq(d, tbl2[i]:index(1, indices), 0.000001)
   end
   -- sub
   local v5 = v:sub(2,3)
   local tbl = v5:forward('bchw', 'torch.DoubleTensor')
   mytester:assert(#tbl == 2)
   for i, d in ipairs(tbl) do
      mytester:assertTensorEq(d, tbl2[i]:index(1, indices), 0.000001)
   end
   local v6 = v:sub(nil, 2,3)
   local tbl = v6:forward('bchw', 'torch.DoubleTensor')
   mytester:assert(#tbl == 2)
   for i, d in ipairs(tbl) do
      mytester:assertTensorEq(d, tbl2[i]:index(1, indices), 0.000001)
   end
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
   ds:index(batch2, torch.LongTensor{2,3,4})
   local batch4 = ds:sub(2, 4)
   mytester:assertTensorEq(batch4:inputs():forward('bhwc'), batch2:inputs():forward('bhwc'), 0.00001)
   mytester:assertTensorEq(batch4:targets():forward('bt'), batch2:targets():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch4:targets():forward('bt'), ds:targets():forward('bt'):narrow(1,2,3), 0.00001)
   local batch3 = ds:sub(nil, 2, 4)
   mytester:assertTensorEq(batch3:inputs():forward('bhwc'), batch2:inputs():forward('bhwc'), 0.00001)
   mytester:assertTensorEq(batch3:targets():forward('bt'), batch2:targets():forward('bt'), 0.00001)
   ds:sub(batch3, 1, 3)
   mytester:assertTensorEq(batch3:inputs():forward('bhwc'), batch:inputs():forward('bhwc'), 0.00001)
   mytester:assertTensorEq(batch3:targets():forward('bt'), batch:targets():forward('bt'), 0.00001)
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
   tensor[tensor:size(1)][2] = 2 -- must finish with end_id
   -- 18 words in vocabulary ("<S>" isn't found in tensor since its redundant to "</S>")
   local words = {"</S>", "<UNK>", "the", "it", "is", "to", "view", "huh", "hi", "ho", "oh", "I", "you", "we", "see", "do", "have", "<S>"}
   -- dataset
   local ds = dp.SentenceSet{which_set='train', data=tensor, words=words, context_size=3, start_id=1, end_id=2}
   local batch = ds:index(torch.LongTensor{1,2,3})
   mytester:assertTableEq(batch:inputs():forward('bt'):size():totable(), {3, 3})
   local batch2 = ds:sub(1, 3)
   mytester:assertTensorEq(batch:inputs():forward('bt'), batch2:inputs():forward('bt'), 0.00001)
   ds:index(batch2, torch.LongTensor{2,3,4})
   local batch4 = ds:sub(2, 4)
   mytester:assertTensorEq(batch4:inputs():forward('bt'), batch2:inputs():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch4:targets():forward('b'), batch2:targets():forward('b'), 0.00001)
   mytester:assertTensorEq(batch4:targets():forward('b'), tensor:select(2,2):narrow(1,2,3), 0.00001)
   local batch3 = ds:sub(nil, 2, 4)
   mytester:assertTensorEq(batch3:inputs():forward('bt'), batch2:inputs():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch3:targets():forward('b'), batch2:targets():forward('b'), 0.00001)
   ds:sub(batch3, 1, 3)
   mytester:assertTensorEq(batch3:inputs():forward('bt'), batch:inputs():forward('bt'), 0.00001)
   mytester:assertTensorEq(batch3:targets():forward('b'), batch:targets():forward('b'), 0.00001)
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
   local preprocess = dp.GCN{sqrt_bias=0.0, use_std=true, progress=false}
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
   
   local preprocess = dp.GCN{std_bias=0.0, use_std=false, progress=false}
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
   local preprocess = dp.ZCA{compute_undo=true,progress=false}
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
function dptest.lecunlcn()
   -- Test on a random image to confirm that it loads without error
   -- and it doesn't result in any NaN or Inf values
   local input_tensor = torch.randn(16, 3, 32, 32)
   local input = dp.ImageView('bchw', input_tensor:clone())
   local pp = dp.LeCunLCN{batch_size=5,progress=false}
   pp:apply(input)
   mytester:assert(_.isFinite(input:forward('default'):sum()), "LeCunLCN isn't finite")

   -- Test on zero-value image if cause any division by zero
   input:forward('bchw', input_tensor:clone():zero()) 
   pp:apply(input)
   local output1 = input:forward('default')
   mytester:assert(_.isFinite(output1:sum()), "LeCunLCN isn't finite (div by zero)")

   -- Test if it works fine with different number of channel as argument
   pp = dp.LeCunLCN{batch_size=5,channels={1, 2},progress=false}
   input:forward('bchw', input_tensor:clone())
   pp:apply(input)
   mytester:assert(_.isFinite(input:forward('default'):sum()), "LeCunLCN isn't finite (less channels)")
   
   -- Divide by standard deviation
   local pp = dp.LeCunLCN{batch_size=5,progress=false,divide_by_std=true}
   input:forward('bchw', input_tensor:clone())
   pp:apply(input)
   local output2 = input:forward('default')
   mytester:assert(_.isFinite(output2:sum()), "LeCunLCN isn't finite")
   mytester:assertTensorNe(output1, output2, 0.000001, "LeCunLCN is not dividing by std")

   -- Test on zero-value image if cause any division by zero
   input:forward('bchw', input_tensor:clone():zero())
   pp:apply(input)
   mytester:assert(_.isFinite(input:forward('default'):sum()), "LeCunLCN isn't finite (div by zero)")

   -- Save a test image
   input_tensor = mytester.lenna:clone()
   input_tensor = input_tensor:view(1,3,512,512)
   input:forward("bchw", input_tensor)
   pp = dp.LeCunLCN{batch_size=1,progress=false}
   pp:apply(input)
   image.savePNG(paths.concat(dp.UNIT_DIR, 'lecunlcn.png'), input:forward('default')[1]) 
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
   local mlp = nn.SoftMaxTree(10, hierarchy, root_id)
   mlp.weight = model._module.weight:clone()
   mlp.bias = model._module.bias:clone()
   -- forward backward
   --- dp
   local output, carry = model:forward(input, dp.Carry{nSample=5, targets=target})
   local gradWeight = model._module.gradWeight:clone()
   output:backward('b', grad_tensor)
   input, carry = model:backward(output, carry)
   mytester:assertTableEq(output:forward('bf'):size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local gradWeight2 = model._module.gradWeight
   mytester:assertTensorNe(gradWeight, gradWeight2, 0.00001)
   --- nn
   local mlp_act = mlp:forward{input_tensor, target_tensor}
   local mlp_grad = mlp:backward({input_tensor, target_tensor}, grad_tensor)[1]
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'), 0.00001)
   -- share
   local model2 = model:sharedClone()   
   -- update
   local weight = model._module.weight:clone()
   local act_ten = output:forward('bf'):clone()
   local grad_ten = input:backward('bf'):clone()
   model:updateParameters(0.1)
   local weight2 = model._module.weight
   mytester:assertTensorNe(weight, weight2, 0.00001)
   model:doneBatch()
   -- forward backward
   local output2, carry2 = model2:forward(input:clone(), dp.Carry{nSample=5, targets=target})
   output2:backward('b', grad_tensor)
   local input2, carry2 = model2:backward(output2, carry2)
   mytester:assertTensorNe(act_ten, output2:forward('bf'), 0.00001)
   mytester:assertTensorNe(grad_ten, input2:backward('bf'), 0.00001)
   local output, carry = model:forward(input2:clone(), dp.Carry{nSample=5, targets=target})
   output:backward('b', grad_tensor)
   local input, carry = model:backward(output, carry)
   mytester:assertTensorEq(output:forward('bf'), output2:forward('bf'), 0.00001)
   mytester:assertTensorEq(input:backward('bf'), input2:backward('bf'), 0.00001)
   -- accUpdate
   local layer = model
   local layer2 = dp.SoftmaxTree{input_size=10, hierarchy=hierarchy, root_id=root_id, acc_update=true}
   layer2._smt.weight = layer._smt.weight:clone()
   layer2._smt.bias = layer._smt.bias:clone()
   layer2:zeroGradParameters()
   layer:zeroGradParameters()
   local input2 = dp.DataView()
   input2:forward('bf', input_tensor)
   input:forward('bf', input_tensor)
   local output2, carry2 = layer2:forward(input2, dp.Carry{nSample=5, targets=target})
   local output, carry = layer:forward(input, dp.Carry{nSample=5, targets=target})
   output2:backward('b', grad_tensor)
   output:backward('b', grad_tensor)
   input = layer:backward(output, carry)
   input2 = layer2:backward(output2, carry2)
   mytester:assertTensorEq(output2:forward('bf'), output:forward('bf'), 0.00001)
   mytester:assertTensorEq(input2:backward('bf'), input:backward('bf'), 0.00001)
   mytester:assertTensorEq(layer2._smt.weight, layer._smt.weight, 0.00001)
   mytester:assertTensorEq(layer2._smt.bias, layer._smt.bias, 0.00001)
   layer2:updateParameters(0.1)
   layer:updateParameters(0.1)
   mytester:assertTensorEq(layer2._smt.weight, layer._smt.weight, 0.00001)
   mytester:assertTensorEq(layer2._smt.bias, layer._smt.bias, 0.00001)
end
function dptest.softmaxforest()
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
   local model = dp.SoftmaxForest{
      input_size=10, hierarchy={hierarchy,hierarchy,hierarchy}, 
      root_id={root_id,root_id,root_id}
   }
   for i=1,3 do
      local params2, gradParams2 = model._experts:get(i):parameters()
      for k,v in pairs(gradParams2) do
         mytester:assert(math.abs(v:sum()) < 0.0001)
      end
   end
   local output, carry = model:forward(input, dp.Carry{nSample=5, targets=target})
   local params, gradParams = model:parameters()
   local gradParams = table.recurse({}, gradParams, function(t,k,v)
      t[k] = v:clone()
   end)
   output:backward('b', grad_tensor)
   input, carry = model:backward(output, carry)
   mytester:assertTableEq(output:forward('bf'):size():totable(), {5,1}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local params2, gradParams2 = model:parameters()
   table.recurse(gradParams, gradParams2, function(t,k,v)
      mytester:assertTensorNe(t[k], v, 0.0000001)
   end)
   -- nn
   -- experts
   local experts = nn.ConcatTable()
   experts:add(nn.SoftMaxTree(10, hierarchy, root_id))
   experts:add(nn.SoftMaxTree(10, hierarchy, root_id))
   experts:add(nn.SoftMaxTree(10, hierarchy, root_id))
   experts:get(1).weight = model._smts[1].weight:clone()
   experts:get(2).weight = model._smts[2].weight:clone()
   experts:get(3).weight = model._smts[3].weight:clone()
   experts:get(1).bias = model._smts[1].bias:clone()
   experts:get(2).bias = model._smts[2].bias:clone()
   experts:get(3).bias = model._smts[3].bias:clone()
   -- gater
   local gater = nn.Sequential()
   gater:add(nn.SelectTable(1)) -- ignore targets
   gater:add(nn.Linear(10,3))
   gater:add(nn.SoftMax())
   gater:get(2).weight = model._gater:get(2).weight:clone()
   gater:get(2).bias = model._gater:get(2).bias:clone()
   -- mixture
   local trunk = nn.ConcatTable()
   trunk:add(gater)
   trunk:add(experts)
   local mixture = nn.MixtureTable()
   local mlp = nn.Sequential()
   mlp:add(trunk)
   mlp:add(mixture)
   mlp:zeroGradParameters()
   for i=1,3 do
      local params2, gradParams2 = experts:get(i):parameters()
      for k,v in pairs(gradParams2) do
         mytester:assert(math.abs(v:sum()) < 0.0001)
      end
   end
   local mlp_act = mlp:forward{input_tensor, target_tensor}
   local mlp_grad = mlp:backward({input_tensor, target_tensor}, grad_tensor)[1]
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bf'), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bf'), 0.00001)
   -- update
   mlp:updateParameters(0.1)
   model:updateParameters(0.1)
   for i=1,experts:size() do
      local expert = experts:get(i)
      local params, gradParams = expert:parameters()
      local params2, gradParams2 = model._experts:get(i):parameters()
      table.recurse(params, params2, function(t,k,v)
         mytester:assertTensorEq(t[k], v, 0.00001)
      end)
   end
   local params, gradParams = gater:parameters()
   local params2, gradParams2 = model._gater:parameters()
   table.recurse(params, params2, function(t,k,v)
      mytester:assertTensorEq(t[k], v, 0.00001)
   end)
end
function dptest.mixtureofexperts()
   local input_tensor = torch.randn(5,10)
   local grad_tensor = torch.randn(5,6)
   -- dp
   local input = dp.DataView()
   input:forward('bf', input_tensor)
   local model = dp.MixtureOfExperts{
      input_size=10, n_expert=3, expert_size={7}, 
      gater_size={8}, output_size=6
   }
   -- forward backward
   --- dp
   local output, carry = model:forward(input, dp.Carry{nSample=5})
   local params, gradParams = model:parameters()
   local gradParams = table.recurse({}, gradParams, function(t,k,v)
      t[k] = v:clone()
   end)
   output:backward('bf', grad_tensor)
   input, carry = model:backward(output, carry)
   mytester:assertTableEq(output:forward('bf'):size():totable(), {5,6}, 0.000001, "Wrong act size")
   mytester:assertTableEq(input:backward('bf'):size():totable(), {5,10}, 0.000001, "Wrong grad size")
   local params2, gradParams2 = model:parameters()
   table.recurse(gradParams, gradParams2, function(t,k,v)
      mytester:assertTensorNe(t[k], v, 0.00001)
   end)
end
function dptest.convolution1D()
   local size = {8,10,50}
   local output_size = {8,4,100}
   local data = torch.rand(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   -- dp
   local input = dp.SequenceView('bwc', data)
   local layer = dp.Convolution1D{
      input_size=50, output_size=100, kernel_size=2, 
      kernel_stride=1, pool_size=2, pool_stride=2,
      transfer=nn.Tanh()
   }
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bwc'):size():totable(), output_size, 0.00001)
   output:backward('bwc', grad_tensor)
   input = layer:backward(output, carry)
   mytester:assertTableEq(input:backward('bwc'):size():totable(), size, 0.00001)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.TemporalConvolution(50,100,2,1)
   m:share(layer._conv, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   mlp:add(nn.TemporalMaxPooling(2,2))
   local mlp_act = mlp:forward(data)
   local mlp_grad = mlp:backward(data, grad_tensor)
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bwc'), 0.00001)
   mytester:assertTableEq(mlp_grad:size():totable(), input:backward('bwc'):size():totable(), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bwc'), 0.00001)
   -- update
   local act_ten = output:forward('bwc'):clone()
   local grad_ten = input:backward('bwc'):clone()
   layer:updateParameters(0.1)
   layer:doneBatch()
   -- forward backward
   output, carry2 = layer:forward(input, dp.Carry{nSample=8})
   output:backward('bwc', grad_tensor)
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten, output:forward('bwc'), 0.00001)
   mytester:assertTensorNe(grad_ten, input:backward('bwc'), 0.00001)
end
function dptest.convolution2D()
   local size = {8,32,32,3}
   local output_size = {8,20,15,15}
   local data = torch.rand(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   -- dp
   local input = dp.ImageView('bhwc', data)
   local layer = dp.Convolution2D{
      input_size=3, output_size=20, kernel_size={3,3}, 
      kernel_stride={1,1}, pool_size={2,2}, pool_stride={2,2},
      transfer=nn.Tanh()
   }
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bchw'):size():totable(), output_size, 0.00001)
   output:backward('bchw', grad_tensor)
   input = layer:backward(output, carry)
   mytester:assertTableEq(input:backward('bhwc'):size():totable(), size, 0.00001)
   -- nn
   local mlp = nn.Sequential()
   local m = nn.SpatialConvolutionMM(3,20,3,3,1,1)
   m:share(layer._conv, 'weight', 'bias')
   mlp:add(m)
   mlp:add(nn.Tanh())
   mlp:add(nn.SpatialMaxPooling(2,2,2,2))
   local mlp_act = mlp:forward(input:forward('bchw'))
   local mlp_grad = mlp:backward(input:forward('bchw'), grad_tensor)
   -- compare nn and dp
   mlp_grad = dp.ImageView('bchw', mlp_grad):forward('bhwc')
   mytester:assertTensorEq(mlp_act, output:forward('bchw'), 0.00001)
   mytester:assertTableEq(mlp_grad:size():totable(), input:backward('bhwc'):size():totable(), 0.00001)
   mytester:assertTensorEq(mlp_grad, input:backward('bhwc'), 0.00001)
   -- update
   local act_ten = output:forward('bhwc'):clone()
   local grad_ten = input:backward('bhwc'):clone()
   layer:updateParameters(0.1)
   layer:doneBatch()
   -- forward backward
   output, carry2 = layer:forward(input, dp.Carry{nSample=8})
   output:backward('bchw', grad_tensor)
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten, output:forward('bhwc'), 0.00001)
   mytester:assertTensorNe(grad_ten, input:backward('bhwc'), 0.00001)
end
function dptest.inception()
   local size = {8,32,32,3} --bhwc
   local output_size = {8,16+24+8+12,32,32} --bchw
   local data = torch.rand(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   -- dp
   local input = dp.ImageView('bhwc', data)
   local layer = dp.Inception{
      input_size=3, output_size={16,24}, reduce_size={14,16,8,12}, kernel_size={5,3}, 
      pool_size=3, pool_stride=1, transfer=nn.ReLU(), sparse_init=true
   }
   for i, param in ipairs(layer:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception init error')
   end
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bchw'):size():totable(), output_size, 0.00001)
   mytester:assert(_.isFinite(output:forward('bchw'):sum()))
   output:backward('bchw', grad_tensor)
   input = layer:backward(output, carry)
   mytester:assertTableEq(input:backward('bhwc'):size():totable(), size, 0.00001)
   mytester:assert(_.isFinite(input:backward('bhwc'):sum()))
   layer:updateParameters(0.1)
   for i, param in ipairs(layer:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception update error')
   end
   layer:maxNorm(1)
   for i, param in ipairs(layer:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception maxNorm error')
   end
end
function dptest.dictionary()
   local size = {8,10}
   local output_size = {8,10,50}
   local data = torch.randperm(80):resize(unpack(size))
   local grad_tensor = torch.randn(unpack(output_size))
   -- dp
   local input = dp.ClassView('bt', data)
   local layer = dp.Dictionary{dict_size=100, output_size=50}
   local output, carry = layer:forward(input, dp.Carry{nSample=8})
   mytester:assertTableEq(output:forward('bwc'):size():totable(), output_size, 0.00001)
   output:backward('bwc', grad_tensor)
   input = layer:backward(output, carry)
   -- should be able to get input gradients
   local function f() 
      input:backward('bt') 
   end 
   mytester:assert(not pcall(f))
   -- nn
   local mlp = nn.LookupTable(100,50)
   mlp:share(layer._module, 'weight')
   local mlp_act = mlp:forward(input:forward('bt'))
   -- compare nn and dp
   mytester:assertTensorEq(mlp_act, output:forward('bwc'), 0.00001)
   -- update
   local act_ten = output:forward('bwc'):clone()
   layer:updateParameters(0.1)
   layer:doneBatch()
   -- forward backward
   output, carry2 = layer:forward(input, dp.Carry{nSample=8})
   output:backward('bwc', grad_tensor)
   input, carry2 = layer:backward(output, carry2)
   mytester:assertTensorNe(act_ten, output:forward('bwc'), 0.00001)
end

function dptest.recurrentdictionary()
   local batchSize = 8
   local dictSize = batchSize * 30
   local hiddenSize = 10
   local data = torch.randperm(dictSize):view(batchSize, -1)
   local gradData = torch.randn(dictSize, batchSize, hiddenSize)
   -- dp
   local input = dp.ClassView()
   local layer = dp.RecurrentDictionary{dict_size=dictSize, output_size=hiddenSize}
   layer:zeroGradParameters()
   -- nn
   local rnn = nn.Recurrent(hiddenSize, nn.LookupTable(dictSize, hiddenSize), nn.Linear(hiddenSize, hiddenSize))
   rnn.feedbackModule.weight = layer._feedback.weight:clone()
   rnn.feedbackModule.bias = layer._feedback.bias:clone()
   rnn.inputModule.weight = layer._lookup.weight:clone()
   rnn.startModule.bias = layer._recurrent.startModule.bias:clone()
   rnn:zeroGradParameters()
   -- compare
   for step=1,10 do
      -- forward
      local inputTensor = data:select(2,step)
      input:forward('b', inputTensor)
      local output, carry = layer:forward(input, dp.Carry{nSample=batchSize})
      local outputTensor = rnn:forward(inputTensor)
      mytester:assertTensorEq(outputTensor, output:forward('bf'), 0.000001)
      -- backward
      local gradOutputTensor = gradData[step]
      output:backward('bf', gradOutputTensor)
      layer:backward(output, carry)
      rnn:backward(inputTensor, gradOutputTensor)
   end
   -- updateParameters  (and BPTT)
   layer:updateParameters(0.1)
   rnn:updateParameters(0.1)
   local params = layer:parameters()
   local params2 = rnn:parameters()
   mytester:assert(table.length(params) ~= #params2, "missing specific case for lookuptable")
   layer.forwarded = false
   local params = layer:parameters()
   mytester:assert(#params == #params2, #params.." should equal "..#params2)
   for i, param in ipairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.000001, "error in update "..i)
   end
end
function dptest.nll()
   local input_tensor = torch.randn(5,10)
   local target_tensor = torch.randperm(10):sub(1,5)
   -- dp
   local input = dp.DataView('bf', input_tensor)
   local target = dp.ClassView('b', target_tensor)
   local loss = dp.NLL{size_average=false} -- else loss isn't avg
   -- test conversion
   loss:float()
   local err, carry = loss:forward(input, target, dp.Carry{nSample=5})
   input = loss:backward(input, target, carry)
   -- nn
   local criterion = nn.ClassNLLCriterion():float()
   local c_err = criterion:forward(input_tensor:float(), target_tensor:float())
   local c_grad = criterion:backward(input_tensor:float(), target_tensor:float())
   -- compare nn and dp
   mytester:asserteq(c_err, err, 0.000001)
   mytester:assertTensorEq(c_grad, input:backward('bf'):float(), 0.00001)
end
function dptest.sentencesampler()
   local nIndice = 1000
   local batchSize = 10
   
   -- create dummy sentence dataset
   local data = torch.IntTensor(nIndice, 2):zero()
   data:select(2,2):copy(torch.range(1,nIndice))
   local start_id, end_id = nIndice+1, nIndice+2
   local startIdx = 1
   local nSentence = 1
   local count = 0
   local maxSize = 20
   for i=1,nIndice do
      data[i][1] = startIdx
      if i < nIndice - 5 and count > 3 and (math.random() < 0.1 or maxSize == count) then
         data[i][2] = end_id
         startIdx = i+1
         nSentence = nSentence + 1
         count = 0
      end
      count = count + 1
   end
   data[nIndice][2] = end_id
   
   local epochSize = nIndice / 10
   local dataset = dp.SentenceSet{data=data,which_set='train',start_id=start_id,end_id=end_id}
   local nWord = 0
   for sentenceSize,s in pairs(dataset:groupBySize()) do
      nWord = nWord + (s.count*sentenceSize)
   end
   mytester:assert(nWord == nIndice, "error in groupBySize "..nWord..' vs '..nIndice)
   
   local sampler = dp.SentenceSampler{batch_size=batchSize,epoch_size=epochSize,evaluate=false}
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local nSampled = 0
   local batch
   while true do
      batch = batchSampler(batch)
      if not batch then
         break
      end
      mytester:assert(batch.isBatch)
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 1)
      mytester:assert(targets:dim() == 1)
      mytester:assert(inputs:size(1) == targets:size(1))
      for i=1,inputs:size(1) do
         local wordIdx = inputs[i]
         if wordIdx ~= start_id and wordIdx ~= end_id then
            local exists = sampled[wordIdx]
            mytester:assert(not exists, 'word sampled twice '..wordIdx)
            sampled[wordIdx] = true
         end
      end
      nSampled = nSampled + inputs:size(1)
   end 
   mytester:assert(nSampled < epochSize + maxSize * batchSize, "iterator not stoping")
   
   local epochSize = nIndice
   local dataset = dp.SentenceSet{data=data,which_set='train',start_id=start_id,end_id=end_id}
   local sampler = dp.SentenceSampler{batch_size=batchSize,epoch_size=-1,evaluate=false}
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local nSampled = 0
   local sampledTwice = 0
   while nSampled < epochSize do
      batch = batchSampler(batch)
      mytester:assert(batch.isBatch)
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 1)
      mytester:assert(targets:dim() == 1)
      mytester:assert(inputs:size(1) == targets:size(1))
      for i=1,inputs:size(1) do
         local wordIdx = inputs[i]
         if wordIdx ~= start_id and wordIdx ~= end_id then
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      nSampled = nSampled + inputs:size(1)
   end
   
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(table.length(sampled) == nIndice-nSentence, "not all words were sampled")
   
   local sampled = {}
   local nSampled = 0
   local sampledTwice = 0
   while nSampled < epochSize do
      batch = batchSampler(batch)
      mytester:assert(batch.isBatch)
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 1)
      mytester:assert(targets:dim() == 1)
      mytester:assert(inputs:size(1) == targets:size(1))
      for i=1,inputs:size(1) do
         local wordIdx = inputs[i]
         if wordIdx ~= start_id and wordIdx ~= end_id then
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      nSampled = nSampled + inputs:size(1)
   end
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(table.length(sampled) == nIndice-nSentence, "not all words were sampled")
end

function dptest.errorminima()
   local report={validator={loss={avgError=11}},epoch=1}
   local mediator = dp.Mediator()
   local m = dp.ErrorMinima{start_epoch=5}
   m:setup{mediator=mediator}
   local mt = {}
   local recv_count = 0
   local minima, minima_epoch
   mediator:subscribe('errorMinima', mt, 'errorMinima')
   
   -- test start_epoch
   function mt:errorMinima(found_minima, m)
      if found_minima then
         minima, minima_epoch = m:minima()
         recv_count = recv_count + 1
      end
   end
   for epoch=1,10 do
      report.epoch = epoch
      m:doneEpoch(report)
   end
   mytester:assert(recv_count == 1, "ErrorMinima recv_count error")
   mytester:assert(minima == 11, "ErrorMinima minima error")
   mytester:assert(minima_epoch == 5, "ErrorMinima start_epoch error")

   
   -- test that minimas is found
   local recv_count = 0
   local losses = {10,8,8,9,7}
   local cme = {11,12,15}
   local cm = {10,8,7}
   function mt:errorMinima(found_minima, m)
      if found_minima then
         minima, minima_epoch = m:minima()
         recv_count = recv_count + 1
         mytester:assert(losses[cme[recv_count]-10] == cm[recv_count], "ErrorMinima wrong minima")
      end
   end
   for epoch=11,15 do
      report.epoch = epoch
      report.validator.loss.avgError = losses[epoch-10]
      m:doneEpoch(report)
   end
   mytester:assert(recv_count == 3, "ErrorMinima recv_count error")
   mytester:assert(minima == 7, "ErrorMinima minima error")
   mytester:assert(minima_epoch == 15, "ErrorMinima epoch error")
end

function dptest.adaptivelearningrate()
   local lr = 20
   local mediator = dp.Mediator()
   local m = dp.AdaptiveLearningRate{max_wait=1,decay_factor=0.1}
   local visitor = dp.Learn{learning_rate=lr, verbose=false, observer=m}
   visitor:setup{mediator=mediator, id=dp.ObjectID('learn')}
   
   for epoch=1,10 do
      if epoch % 3 == 0 then
         mediator:publish('errorMinima', true)
      else
         mediator:publish('errorMinima', false)
      end
   end
   lr = lr*(0.1^3)
   mytester:assert(visitor:learningRate() == lr, "AdaptiveLearningRate learningRate error 1")
   
   mediator:publish('errorMinima', true)
   mytester:assert(visitor:learningRate() == lr, "AdaptiveLearningRate learningRate error 2")
   for epoch=1,10 do
      mediator:publish('errorMinima', false)
   end
   mytester:assert(math.abs(visitor:learningRate() - lr*(0.1^5)) < 0.00001, "AdaptiveLearningRate learningRate error 3")
end

function dptest.savetofile()
   local mediator = dp.Mediator()
   local stf = dp.SaveToFile{in_memory=true,verbose=false}
   local subject = {
      tensor=torch.randn(3,4), stf=stf,
      id=function()return{toPath=function()return'test462346';end};end
   }
   stf:setup(subject, mediator)
   local path = paths.concat(stf._save_dir, stf._filename)
   if paths.filep(path) then
      os.execute('rm '..path)
   end
   stf:save(subject)
   mediator:publish("doneExperiment")
   mytester:assert(paths.filep(path), "file doesn't exist")
   local saved_subject = torch.load(path)
   mytester:assert(not saved_subject.stf._save_cache, "cache was saved")
   mytester:assertTensorEq(subject.tensor, saved_subject.tensor, 0.000001, "tensor not saved")
   
   stf = saved_subject.stf
   mediator = saved_subject.stf._mediator
   subject = saved_subject
   subject.tensor = torch.randn(3,4)
   stf:save(subject)
   mediator:publish("doneExperiment")
   local saved_subject = torch.load(path)
   mytester:assertTensorEq(subject.tensor, saved_subject.tensor, 0.000001, "tensor not re saved")
end

function dptest.topcrop()
   local fb = dp.TopCrop{n_top={1,3}, n_crop=3,center=1,verbose=false}
   fb._id = dp.ObjectID('topcrop')
   local carry = dp.Carry{nSample=18}
   local preds = {{1,0,0,0,0,0},{1,0,0,0,0,0},{1,0,0,0,0,0}, -- 1,1
                   {0,1,0,0,0,0},{0,1,0,0,0,0},{0,1,0,0,0,0}, -- 1,1
                   {0,1,0,0,0,0},{0,0,1,0,0,0},{0,0,1,0,0,0}, -- 1,1
                   {1,0,0,2,0,0},{1,0,0,2,0,0},{1,0,0,2,0,0}, -- 0,1
                   {0,-1,0,0,2,0},{0,0,0,0,2,0},{0,-1,0,0,2,0}, -- 0,0
                   {0,0,-1,0,0,1},{0,0,-1,0,0,1},{0,0,-1,0,0,1}} -- 0,0
   preds = torch.FloatTensor(preds)
   local targets = torch.IntTensor{1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3}
   local inputs = torch.randn(18,1,5,5)
   local targetView = dp.ClassView('b', targets)
   local inputView = dp.ImageView('bchw', inputs)
   local batch = dp.Batch{inputs=inputView, targets=targetView, carry=carry}
   fb:add(batch, preds, {epoch=1})
   fb:add(batch, preds, {epoch=1})
   fb:doneEpoch({epoch=1})
   local report = fb:report()
   mytester:assert(report.topcrop.all[1] == 50, "topcrop all 1 error")
   mytester:assert(math.round(report.topcrop.all[3]) == 67, "topcrop all 3 error")
   mytester:assert(math.round(report.topcrop.center[1]) == 33, "topcrop center 1 error")
   mytester:assert(math.round(report.topcrop.center[3]) == 67, "topcrop center 3 error")
end

function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester.lenna = image.lena()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

