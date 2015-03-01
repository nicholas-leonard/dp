require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test ImageNet and ImageClassSet')
cmd:text('Options:')
cmd:option('--dataPath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'path to ImageNet')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--epochSize', 2000, 'number of train examples seen between each epoch')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--testAsync', false, 'test asynchronous mode bar')
cmd:text()
opt = cmd:parse(arg or {})

ds = dp.ImageNet{
   train_path=paths.concat(opt.dataPath, 'ILSVRC2012_img_train'),
   valid_path=paths.concat(opt.dataPath, 'ILSVRC2012_img_val'),
   meta_path=paths.concat(opt.dataPath, 'metadata'),
   verbose = opt.verbose,
   load_all = false
}

validSet = ds:loadValid()
batch = validSet:sample(128)
--[[print(batch:inputs():view(), batch:inputs():input():size())

batch = validSet:sample(128, 'sampleTest')
print(batch:inputs():view(), batch:inputs():input():size(), batch:targets():input())

batch = validSet:sample(4, 'sampleTrain')
print(batch:inputs():view(), batch:inputs():input():size())

batch = validSet:sub(100, 200)
print("sub1", batch:inputs():view(), batch:inputs():input():size())

validSet:sub(batch, 200, 240)
print("sub2", batch:inputs():view(), batch:inputs():input():size())--]]


if opt.testAsync then
   samplerB = dp.Sampler{batch_size=math.floor(opt.batchSize/10), epoch_size=opt.epochSize}
      
   local isum2, tsum2 = 0, 0
   local a = torch.Timer()
   local nBatch2 = 0
   for k=1,2 do
      local batch2, i, n
      local sampler2 = samplerB:sampleEpoch(validSet)
      while true do
         batch2, i, n = sampler2(batch2)
         if not batch2 then
            break
         end
         local sumi2, sumt2 = batch2:inputs():forward():sum(), batch2:targets():forward():sum()
         isum2 = isum2 + sumi2
         tsum2 = tsum2 + sumt2
         nBatch2 = nBatch2 + 1
      end
   end
   print("sync", (a:time().real)/nBatch2)
   
   validSet:multithread(4)
   samplerA = dp.Sampler{batch_size=math.floor(opt.batchSize/10), epoch_size=opt.epochSize}
   samplerA:async()
   
   local a = torch.Timer()
   local nBatch = 0
   local isum, tsum = 0, 0
   for k=1,2 do
      local batch, i, n
      local sampler = samplerA:sampleEpoch(validSet)
      while true do
         batch, i, n = sampler(batch)
         if not batch then
            break
         end
         assert(batch.isBatch)
         local sumi, sumt = batch:inputs():forward():sum(), batch:targets():forward():sum()
         isum = isum + sumi
         tsum = tsum + sumt
         nBatch = nBatch + 1
      end
   end
   print("async", (a:time().real)/nBatch)
   
   print("sums", isum, tsum, isum2, tsum2, "counts", nBatch, nBatch2)
   
   assert(isum == isum2)
   assert(tsum == tsum2)
   assert(nBatch == nBatch2)
end


os.exit()

--ppf = ds:normalizePPF()

trainSet = ds:trainSet() or ds:loadTrain()

batch = trainSet:sample(batch,120,'sampleTrain')

inputView = batch:inputs() 
inputs = inputView:forward('bchw')

targetView = batch:targets()
targets = targetView:forward('b')

savePath = paths.concat(dp.SAVE_DIR, 'imagenettest')
paths.mkdir(savePath)

for i=1,inputs:size(1) do
   image.save(paths.concat(savePath, 'sample'..i..'_'..targets[i]..'.png'), inputs[i])
end

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   local img = image.load(imgpath)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImageB", (a:time().real)/10)

local a = torch.Timer()
float = torch.FloatTensor()
dst = torch.FloatTensor()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   local out = image.load(imgpath)
   float:resize(out:size()):copy(out)
   dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
   image.scale(dst, float)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImageB+scale", (a:time().real)/10)

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   trainSet:loadImage(imgpath) 
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage", (a:time().real)/10)


local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   input = trainSet:loadImage(imgpath) 
   local out = input:toTensor('float','RGB','DHW', true)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage+", (a:time().real)/10)

local a = torch.Timer()
for i=1,10*120 do
   local imgpath = ffi.string(torch.data(trainSet.imagePath[i]))
   input = trainSet:loadImage(imgpath) 
   local out = input:toTensor('float','RGB','DHW', true)
   dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
   image.scale(dst, out)
   if math.fmod(i, 120) == 0 then
      collectgarbage()
   end
end
print("loadImage+scale", (a:time().real)/10)

local a = torch.Timer()
for i=1,120 do
   local dst = trainSet:getImageBuffer(i)
   dst:resize(10, 3, trainSet._sample_size[3], trainSet._sample_size[2])
end
print("getImageBuffer : first pass", (a:time().real)/120)

local a = torch.Timer()
for j=1,10 do
   local inputTable = {}
   local targetTable = {} 
   for i=1,120 do 
      assert(batch)
      local imgpath = ffi.string(torch.data(trainSet.imagePath[j*120+i]))
      input = trainSet:loadImage(imgpath) 
      local out = input:toTensor('float','RGB','DHW', true)
      local dst = trainSet:getImageBuffer(i)
      dst:resize(out:size(1), trainSet._sample_size[3], trainSet._sample_size[2])
      image.scale(dst, out)
      table.insert(inputTable, dst)
      table.insert(targetTable, 1)  
   end
   local inputView = batch and batch:inputs() or dp.ImageView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()
   
   trainSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   
   assert(inputTensor:size(2) == 3)
   inputView:forward('bchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(trainSet._classes)
   batch:setInputs(inputView)
   batch:setTargets(targetView)  
   batch:carry():putObj('nSample', targetTensor:size(1))
   collectgarbage()
end
print("loadImage+scale+tableToTensor", (a:time().real)/10)

a = torch.Timer()
for i=1,10 do
   trainSet:sample(batch,120,'sampleDefault') 
   collectgarbage()
end
print("sampleDefault", (a:time().real)/10)

a = torch.Timer()
for i=1,10 do
   trainSet:sample(batch,120,'sampleTrain') 
   collectgarbage()
end
print("sampleTrain", (a:time().real)/10)

a = torch.Timer()
for i=1,120*10,120 do
   trainSet:sub(batch,i,i+119) 
   collectgarbage()
end
print("sampleTest", (a:time().real)/10)

