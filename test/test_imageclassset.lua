require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test ImageNet and ImageClassSet')
cmd:text('Options:')
cmd:option('--dataPath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'path to ImageNet')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--epochSize', 2000, 'number of train examples seen between each epoch')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--testAsyncSub', false, 'test asynchronous sub')
cmd:option('--testAsyncSample', false, 'test asynchronous sample')
cmd:option('--nThread', 2, 'number of threads')
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
print"dataset loaded"
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


if opt.testAsyncSub then
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
   
   validSet:multithread(opt.nThread)
   print"multithread"
   samplerA = dp.Sampler{batch_size=math.floor(opt.batchSize/10), epoch_size=opt.epochSize}
   samplerA:async()
   
   local a = torch.Timer()
   local nBatch = 0
   local isum, tsum = 0, 0
   for k=1,2 do
      local batch, i, n
      local sampler = samplerA:sampleEpoch(validSet)
      print(k)
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


if opt.testAsyncSample then
   validSet = ds:loadTrain()
   samplerB = dp.RandomSampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
      
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
         nBatch2 = nBatch2 + 1
      end
   end
   print("sync", nBatch2, (a:time().real)/nBatch2)
   
   validSet:multithread(opt.nThread, opt.nThread*2)
   samplerA = dp.RandomSampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
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
         nBatch = nBatch + 1
      end
   end
   print("async", nBatch, (a:time().real)/nBatch)
end

validSet = ds:validSet() or ds:loadValid()

--ppf = ds:normalizePPF()

trainSet = ds:validSet() or ds:loadValid()

batch = trainSet:sample(batch,120,'sampleTrain')


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

