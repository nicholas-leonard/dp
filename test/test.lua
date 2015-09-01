local mytester 
local dptest = {}

function dptest.uniqueID()
   local uid1 = dp.uniqueID()
   mytester:asserteq(type(uid1), 'string', 'type(uid1) == string')
   local uid2 = dp.uniqueID()
   local uid3 = dp.uniqueID('mynamespace')
   mytester:assertne(uid1, uid2, 'uid1 ~= uid2')
   mytester:assertne(uid2, uid3, 'uid2 ~= uid3')
end

function dptest.DataView()
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

function dptest.ImageView()
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

function dptest.SequenceView()
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

function dptest.ClassView()
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

function dptest.ListView()
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

function dptest.DataSet()
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

function dptest.SentenceSet()
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

function dptest.DataSource()
   if true then return end -- TODO
   local input = torch.randn(3,4,5,6)
   local target = torch.randn(3)
   local inputView = dp.ImageView('bchw', input)
   local targetView = dp.ClassView('b', target)
   local valid = dp.DataSet{which_set='valid', inputs=input, targets=target}
   local ds = dp.DataSource{valid_set=valid}
   -- test ioShape
   local iShape, oShape = ds:ioShape()
   mytester:assert(iShape == 'bchw')
   mytester:assert(oShape == 'b')
   -- test iAxes
   
   -- test iSize
end

function dptest.TextSet()
   local data = torch.IntTensor(200):random(1,40)
   data[1] = 41 -- 41st word in vocabulary
   data[200] = 42 -- 42nd word in vocabulary
   local inputFreq, targetFreq = {}, {}
   data:narrow(1,1,190):apply(function(x) inputFreq[x] = (inputFreq[x] or 0) + 1 end)
   data:narrow(1,11,190):apply(function(x) targetFreq[x] = (targetFreq[x] or 0) + 1 end)
   local ts = dp.TextSet{data=data,which_set='train',context_size=10,recurrent=true}
   -- test TextSet:index() for recurrent = true
   local sampler = dp.ShuffleSampler{batch_size=4}
   local inputWC, targetWC = {}, {}
   local batch
   local sampleBatch = sampler:sampleEpoch(ts)
   local nSample = 0
   while true do
      batch = sampleBatch(batch)
      if not batch then
         break
      end
      local inputs = batch:inputs():forward('bt')
      local targets = batch:targets():forward('bt')
      mytester:assert(inputs:isSameSizeAs(targets))
      mytester:assert(inputs:size(1) <= 4)
      mytester:assert(inputs:size(2) == 10)
      mytester:assertTensorEq(inputs:narrow(2,2,inputs:size(2)-1), targets:narrow(2,1,targets:size(2)-1), 0.000001)
      inputs:select(2,1):apply(function(x) inputWC[x] = (inputWC[x] or 0) + 1 end)
      targets:select(2,10):apply(function(x) targetWC[x] = (targetWC[x] or 0) + 1 end)
      nSample = nSample + targets:size(1)
   end
   mytester:assert(nSample == 200-10)
   mytester:assertTableEq(inputFreq, inputWC, 0.00001)
   mytester:assertTableEq(targetFreq, targetWC, 0.00001)
   
   -- test TextSet:sub() for recurrent = true
   local inputFreq, targetFreq = {}, {}
   data:narrow(1,1,199):apply(function(x) inputFreq[x] = (inputFreq[x] or 0) + 1 end)
   data:narrow(1,2,199):apply(function(x) targetFreq[x] = (targetFreq[x] or 0) + 1 end)
   ts:contextSize(1)
   local sampler = dp.Sampler{batch_size=10}
   local inputWC, targetWC = {}, {}
   local batch = nil
   local sampleBatch = sampler:sampleEpoch(ts)
   local nSample = 0
   while true do
      batch = sampleBatch(batch)
      if not batch then
         break
      end
      local inputs = batch:inputs():forward('bt')
      local targets = batch:targets():forward('bt')
      mytester:assert(inputs:isSameSizeAs(targets))
      mytester:assert(inputs:size(1) == 1)
      mytester:assert(inputs:size(2) <= 10)
      mytester:assertTensorEq(inputs:narrow(2,2,inputs:size(2)-1), targets:narrow(2,1,targets:size(2)-1), 0.000001)
      inputs:apply(function(x) inputWC[x] = (inputWC[x] or 0) + 1 end)
      targets:apply(function(x) targetWC[x] = (targetWC[x] or 0) + 1 end)
      nSample = nSample + targets:nElement()
   end
   
   mytester:assert(nSample == 200-1)
   mytester:assertTableEq(inputFreq, inputWC, 0.00001)
   mytester:assertTableEq(targetFreq, targetWC, 0.00001)
end

function dptest.TextSource()
   local trainStr = [[ no it was n't black monday 
 but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of it in the final hour it barely managed to stay this side of chaos 
 some circuit breakers installed after the october N crash failed their first test traders say unable to cool the selling panic in both stocks and futures 
 the N stock specialist firms on the big board floor the buyers and sellers of last resort who were criticized after the N crash once again could n't handle the selling pressure 
 big investment banks refused to step up to the plate to support the beleaguered floor traders by buying big blocks of stock traders say 
 heavy selling of standard & poor 's 500-stock index futures in chicago <unk> beat stocks downward 
]]
   local validStr = [[ no it was n't black monday 
 but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of it in the final hour it barely managed to stay this side of chaos 
]]
   local testStr = [[ big investment banks refused to step up to the plate to support the beleaguered floor traders by buying big blocks of stock traders say 
 heavy selling of standard & poor 's 500-stock index futures in chicago <unk> beat stocks downward 
]]
   local ts = dp.TextSource{
      name='unit-test', context_size=5, string=true, recurrent=true,
      train=trainStr, valid=validStr, test=testStr
   }
   local train = trainStr..validStr
   local stringx = require('pl.stringx')
   train = stringx.replace(train, '\n', '<eos>')
   train = stringx.split(train)
   local vocab = {}
   local wordFreq = {}
   local word_seq = 0
   for i,word in ipairs(train) do
      local word_id = vocab[word]
      if not word_id then 
         word_seq = word_seq + 1
         vocab[word] = word_seq
         word_id = word_seq
      end
      wordFreq[word_id] = (wordFreq[word_id] or 0) + 1
   end
   local wordFreq2 = ts:wordFrequency()
   mytester:assertTableEq(wordFreq2, wordFreq, 0.000001)
   
   local nTrain = ts:trainSet():nSample() + 5
   local nValid = ts:validSet():nSample() + 5
   local nTest = ts:testSet():nSample() + 5
   
   local test = stringx.replace(testStr, '\n', '<eos>')
   test = stringx.split(test)
   mytester:assert(nTrain+nValid == #train)
   mytester:assert(nTest == #test)
end

function dptest.GCN()
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

function dptest.ZCA()
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

function dptest.LeCunLCN()
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

function dptest.Sampler()
   local sampler = dp.Sampler{batch_size=64}
   local inputs = torch.randn(1000,5)
   local targets = torch.IntTensor(1000):random(1,10)
   local inputs_v = dp.DataView('bf', inputs)
   local targets_v = dp.ClassView('b', targets)
   local ds = dp.DataSet{which_set='train',inputs=inputs_v, targets=targets_v}
   local batchSampler = sampler:sampleEpoch(ds)
   local n = 0
   local batch = batchSampler(batch)
   while batch do
      local binputs = batch:inputs():input()
      local btargets = batch:targets():input()
      mytester:assert(batch:nSample() == binputs:size(1))
      mytester:assert(batch:nSample() == btargets:size(1))
      mytester:assertTensorEq(inputs:narrow(1,n+1,binputs:size(1)), binputs, 0.00001)
      mytester:assertTensorEq(targets:narrow(1,n+1,btargets:size(1)), btargets, 0.00001)
      n = n + batch:nSample()
      batch = batchSampler(batch)
   end
   mytester:assert(ds:nSample() == n)
end

function dptest.ShuffleSampler()
   torch.manualSeed(777)
   local shuffle = torch.randperm(1000):long()
   local inputs = torch.randn(1000,5)
   local targets = torch.IntTensor(1000):random(1,10)
   local inputs_v = dp.DataView('bf', inputs)
   local targets_v = dp.ClassView('b', targets)
   local ds = dp.DataSet{which_set='train',inputs=inputs_v, targets=targets_v}
   mytester:assert(ds:nSample() == 1000)
   local sampler = dp.ShuffleSampler{batch_size=64, random_seed=777}
   local batchSampler = sampler:sampleEpoch(ds)
   local n = 0
   local batch = batchSampler(batch)
   while batch do
      local binputs = batch:inputs():input()
      local btargets = batch:targets():input()
      mytester:assert(batch:nSample() == binputs:size(1))
      mytester:assert(batch:nSample() == btargets:size(1))
      local indices = shuffle:narrow(1,n+1,binputs:size(1))
      local inputs_ = inputs:index(1,indices)
      local targets_ = targets:index(1,indices)
      mytester:assertTensorEq(inputs_, binputs, 0.00001)
      mytester:assertTensorEq(targets_, btargets, 0.00001)
      n = n + batch:nSample()
      batch = batchSampler(batch)
      collectgarbage()
   end
   mytester:assert(ds:nSample() == n)
end

function dptest.SentenceSampler()
   local nIndice = 1000
   local batchSize = 3
   
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
   
   local epochSize = nSentence - 3
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
   local nSampled2 = 0
   local batch
   while true do
      batch = batchSampler(batch)
      if not batch then
         break
      end
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if wordIdx ~= start_id and wordIdx ~= end_id then
               local exists = sampled[wordIdx]
               mytester:assert(not exists, 'word sampled twice '..wordIdx)
               sampled[wordIdx] = true
            end
         end
      end
      nSampled = nSampled + inputs:size(1)
      nSampled2 = nSampled2 + inputs:nElement()
   end 
   mytester:assert(nSampled <= epochSize + 3, "iterator not stoping "..nSampled.." ~= "..epochSize.." + 3")
   
   local epochSize = nSentence
   local dataset = dp.SentenceSet{data=data,which_set='train',start_id=start_id,end_id=end_id}
   local sampler = dp.SentenceSampler{batch_size=batchSize,epoch_size=-1}
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local nSampled = 0
   local nSampled2 = 0
   local sampledTwice = 0
   while nSampled < epochSize do
      batch = batchSampler(batch)
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if wordIdx ~= start_id and wordIdx ~= end_id then
               if sampled[wordIdx] then
                  sampledTwice = sampledTwice + 1
               end
               sampled[wordIdx] = true
            end
         end
      end
      nSampled = nSampled + inputs:size(1)
      nSampled2 = nSampled2 + inputs:nElement()
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
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if wordIdx ~= start_id and wordIdx ~= end_id then
               if sampled[wordIdx] then
                  sampledTwice = sampledTwice + 1
               end
               sampled[wordIdx] = true
            end
         end
      end
      nSampled = nSampled + inputs:size(1)
   end
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(table.length(sampled) == nIndice-nSentence, "not all words were sampled")
end

function dptest.TextSampler()
   local nIndice = 1000
   local batchSize = 3
   local dictSize = 200
   local contextSize = 20
   
   -- create dummy sentence dataset
   local data = torch.IntTensor(nIndice):range(1,nIndice)
   local dataset = dp.TextSet{data=data,which_set='train',context_size=contextSize,recurrent=true}
   
   local epochSize = 10*batchSize*contextSize
   local sampler = dp.TextSampler{batch_size=batchSize,epoch_size=epochSize}
   
   local slicedData = torch.IntTensor(batchSize, math.floor(nIndice/batchSize))
   local j = 1
   for i=1,batchSize do
      slicedData[i]:copy(data:narrow(1,j, math.floor(nIndice/batchSize)))
      j = j + math.floor(nIndice/batchSize)
   end
   
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local sampledTwice = 0
   local nSampled = 0
   local batch
   local i = 1
   while true do
      batch = batchSampler(batch)
      if not batch then
         break
      end
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      mytester:assertTensorEq(inputs, slicedData:narrow(2,i,contextSize), 0.0000001)
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      nSampled = nSampled + inputs:nElement()
      i = i + contextSize
   end 
   mytester:assert(nSampled == epochSize, "iterator not stoping "..nSampled.." ~= "..epochSize)
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   
   local epochSize = dataset:textSize()
   local sampler = dp.TextSampler{batch_size=batchSize,epoch_size=-1}
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local sampledTwice = 0
   local nSampled = 0
   local i = 1
   while nSampled < 999 do
      batch = batchSampler(batch)
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      local stop = i+contextSize-1
      if stop > slicedData:size(2) then
         stop = slicedData:size(2)
      end
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      local slice = slicedData[{{},{i,stop}}]
      mytester:assertTensorEq(inputs, slice, 0.0000001, "TextSampler full epoch sample "..i.." "..nSampled)
      nSampled = nSampled + inputs:nElement()
      i = i + contextSize
   end
   
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(nSampled == 999, "not all words were sampled "..nSampled.." ~= 999")
   
   local epochSize = dataset:nSample()
   local sampler = dp.TextSampler{batch_size=1,epoch_size=-1}
   local batchSampler = sampler:sampleEpoch(dataset)
   local sampled = {}
   local sampledTwice = 0
   local nSampled = 0
   local i = 1
   local nBatch = 1
   batch = batchSampler(batch)
   while batch do
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      local stop = math.min(data:size(1)-1, i+contextSize-1)
      local slice = data[{{i,stop}}]
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      mytester:assertTensorEq(inputs, slice, 0.0000001, "TextSampler 1D full epoch sample "..i.." "..nSampled)
      nSampled = nSampled + inputs:nElement()
      i = i + contextSize
      batch = batchSampler(batch)
      nBatch = nBatch + 1
   end
   
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(nSampled == nIndice-1, "not all words were sampled "..nSampled.." ~= "..(nIndice-1))
   
   batchSize = 7
   local slicedData = torch.IntTensor(batchSize, math.floor(nIndice/batchSize))
   local j = 1
   for i=1,batchSize do
      slicedData[i]:copy(data:narrow(1,j, math.floor(nIndice/batchSize)))
      j = j + math.floor(nIndice/batchSize)
   end
   
   local epochSize = dataset:textSize()
   local sampler = dp.TextSampler{batch_size=batchSize,epoch_size=-1}
   local batchSampler = sampler:sampleEpoch(dataset)
   local nSampled = 0
   local i = 1
   local sampled = {}
   local sampledTwice = 0
   while nSampled < 994 do
      batch = batchSampler(batch)
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      local stop = i+contextSize-1
      if stop > slicedData:size(2) then
         stop = slicedData:size(2)
      end
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      local slice = slicedData[{{},{i,stop}}]
      mytester:assertTensorEq(inputs, slice, 0.0000001, "TextSampler full epoch sample "..i.." "..nSampled)
      nSampled = nSampled + inputs:nElement()
      i = i + contextSize
   end
   
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(nSampled == 994, "not all words were sampled "..nSampled.." ~= 994")
   
   local batchSampler = sampler:sampleEpoch(dataset)
   local nSampled = 0
   local i = 1
   local sampled = {}
   local sampledTwice = 0
   while nSampled < 994 do
      batch = batchSampler(batch)
      mytester:assert(torch.isTypeOf(batch, 'dp.Batch'))
      local inputs = batch:inputs():input()
      local targets = batch:targets():input()
      mytester:assert(inputs:dim() == 2)
      mytester:assert(targets:dim() == 2)
      mytester:assert(inputs:isSameSizeAs(targets))
      local stop = i+contextSize-1
      if stop > slicedData:size(2) then
         stop = slicedData:size(2)
      end
      for i=1,inputs:size(1) do
         for j=1,inputs:size(2) do
            local wordIdx = inputs[{i,j}]
            if sampled[wordIdx] then
               sampledTwice = sampledTwice + 1
            end
            sampled[wordIdx] = true
         end
      end
      local slice = slicedData[{{},{i,stop}}]
      mytester:assertTensorEq(inputs, slice, 0.0000001, "TextSampler full epoch sample "..i.." "..nSampled)
      nSampled = nSampled + inputs:nElement()
      i = i + contextSize
   end
   
   mytester:assert(sampledTwice == 0, sampledTwice..' words sampled twice ')
   mytester:assert(not batchSampler(batch), "iterator not stoping")
   mytester:assert(nSampled == 994, "not all words were sampled "..nSampled.." ~= 994")
end

function dptest.ErrorMinima()
   local report={validator={loss=11},epoch=1}
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
      report.validator.loss = losses[epoch-10]
      m:doneEpoch(report)
   end
   mytester:assert(recv_count == 3, "ErrorMinima recv_count error")
   mytester:assert(minima == 7, "ErrorMinima minima error")
   mytester:assert(minima_epoch == 15, "ErrorMinima epoch error")
end

function dptest.AdaptiveDecay()
   local lr = 20
   local m = dp.AdaptiveDecay{max_wait=1,decay_factor=0.1}
   
   local lr2 = 20
   for epoch=1,10 do
      if epoch % 3 == 0 then
         m:errorMinima(true)
      else
         m:errorMinima(false)
      end
      lr2 = lr2*m.decay
   end
   lr = lr*(0.1^3)
   mytester:assert(lr2 == lr, "AdaptiveDecay learningRate error 1")
   
   for epoch=1,10 do
      m:errorMinima(false)
      lr2 = lr2*m.decay
   end
   mytester:assert(math.abs(lr2 - lr*(0.1^5)) < 0.00001, "AdaptiveDecay learningRate error 3")
end

function dptest.SaveToFile()
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

function dptest.Confusion()
   local nSample = 100
   local inputs = torch.randn(nSample,8,8,1)
   local targets = torch.IntTensor(nSample):random(1,10)
   local targetView = dp.ClassView('b', targets)
   targetView:setClasses({0,1,2,3,4,5,6,7,8,9})
   local inputView = dp.ImageView('bhwc', inputs)
   local ds = dp.DataSet{inputs=inputView, targets=targetView}
   local output = torch.randn(nSample,10)
   local conf = dp.Confusion()
   local sampler = dp.Sampler{batch_size=8}
   local batchSampler = sampler:sampleEpoch(ds)
   local batch = batchSampler()
   local i = 1
   while batch do
      conf:add(batch, output:narrow(1,i,batch:nSample()), {epoch=1})
      i = i + batch:nSample()
      batch = batchSampler(batch)
   end
   mytester:assert(i-1 == nSample)
   mytester:assert(conf:nSample() == nSample)
   mytester:assert(conf._cm.mat:sum() == nSample)
   local report = conf:report()
   local acc = report[conf:name()].accuracy
   local vals, preds = output:max(2)
   preds = preds:select(2,1):int()
   local acc2 = torch.eq(preds, targets):float():mean()
   mytester:assert(acc == acc2)
end

function dptest.Perplexity()
   local nSample = 100
   local inputs = torch.randn(nSample,8,8,1)
   local targets = torch.IntTensor(nSample, 10):random(1,10)
   local targetView = dp.ClassView('bt', targets)
   targetView:setClasses({0,1,2,3,4,5,6,7,8,9})
   local inputView = dp.ImageView('bhwc', inputs)
   local ds = dp.DataSet{inputs=inputView, targets=targetView}
   local output = {}
   for i=1,10 do
      table.insert(output, torch.randn(nSample,10))
   end
   local ppl = dp.Perplexity()
   local sampler = dp.Sampler{batch_size=8}
   local batchSampler = sampler:sampleEpoch(ds)
   local batch = batchSampler()
   local i = 1
   while batch do
      local output_ = {}
      for k=1,10 do
         table.insert(output_, output[k]:narrow(1,i,batch:nSample()))
      end
      ppl:add(batch, output_, {epoch=1})
      i = i + batch:nSample()
      batch = batchSampler(batch)
   end
   mytester:assert(i-1 == nSample)
   mytester:assert(ppl:nSample() == nSample*10)
   local ppl_val = ppl:perplexity()
   local nll = 0
   for k=1,10 do
      for i=1,nSample do
         nll = nll - output[k][i][targets[{i,k}]]
      end
   end
   mytester:assert(math.abs(ppl._nll - nll) < 0.0000001)
   local ppl_val2 = torch.exp(nll/(nSample*10))
   mytester:assert(math.abs(ppl_val - ppl_val2) < 0.0000001)
   -- test after reset
   ppl:reset()
   local batchSampler = sampler:sampleEpoch(ds)
   local batch = batchSampler()
   local i = 1
   while batch do
      local output_ = {}
      for k=1,10 do
         table.insert(output_, output[k]:narrow(1,i,batch:nSample()))
      end
      ppl:add(batch, output_, {epoch=1})
      i = i + batch:nSample()
      batch = batchSampler(batch)
   end
   mytester:assert(i-1 == nSample)
   mytester:assert(ppl:nSample() == nSample*10)
   mytester:assert(math.abs(ppl_val - ppl:perplexity()) < 0.0000001)
end

function dptest.TopCrop()
   local fb = dp.TopCrop{n_top={1,3}, n_crop=3,center=1,verbose=false}
   fb._id = dp.ObjectID('topcrop')
   local preds = {{1,0,0,0,0,0},{1,0,0,0,0,0},{1,0,0,0,0,0}, -- 1,1
                   {0,1,0,0,0,0},{0,1,0,0,0,0},{0,1,0,0,0,0}, -- 1,1
                   {0,1,-3,0,0,0},{0,0,1,0,0,0},{0,0,1,0,0,0}, -- 0,1
                   {1,0,0,2,0,0},{1,0,0,2,0,0},{1,0,0,2,0,0}, -- 0,1
                   {0,-1,0,0,2,0},{0,-1,0,0,2,0},{0,-1,0,0,2,0}, -- 0,0
                   {0,0,-1,0,0,1},{0,0,-1,0,0,1},{0,0,-1,0,0,1}} -- 0,0

   preds = torch.FloatTensor(preds)
   local targets = torch.IntTensor{1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3}
   local inputs = torch.randn(18,1,5,5)
   local targetView = dp.ClassView('b', targets)
   local inputView = dp.ImageView('bchw', inputs)
   local batch = dp.Batch{inputs=inputView, targets=targetView}
   fb:add(batch, preds, {epoch=1})
   fb:add(batch, preds, {epoch=1})
   fb:silent()
   fb:doneEpoch({epoch=1})
   local report = fb:report()
   mytester:assert(math.round(report.topcrop.all[1]) == 33, "topcrop all 1 error")
   mytester:assert(math.round(report.topcrop.all[3]) == 50, "topcrop all 3 error")
   mytester:assert(math.round(report.topcrop.center[1]) == 33, "topcrop center 1 error")
   mytester:assert(math.round(report.topcrop.center[3]) == 50, "topcrop center 3 error")
end

function dptest.HyperLog()
   local hlog = dp.HyperLog()
   local reports = {}
   for i=1,3 do
      local report = {
         lr = 0.0001,
         epoch = i,
         feedback = {
            confusion = {
               accuracy = 3*i
            }
         }
      }
      table.insert(reports, report)
      hlog:doneEpoch(report)
      hlog:errorMinima(i==2, {minima=function() return 9, 2 end})
   end
   local accs = hlog:getResultByEpoch("feedback:confusion:accuracy")
   local acc = hlog:getResultAtMinima("feedback:confusion:accuracy")
   mytester:assertTableEq(accs, {3,6,9}, 0.0000001, "HyperLog getResultByEpoch err")
   mytester:assert(acc == 6, "HyperLog getResultByMinima err")
end

function dp.test(tests)
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester.lenna = image.lena()
   mytester:add(dptest)
   mytester:run(tests)   
   return mytester
end

