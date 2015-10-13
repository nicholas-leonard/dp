require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test CocoBoxDetect')
cmd:text('Options:')
cmd:option('--imagePath', '/media/Nick/coco/val2014', 'path to Coco images')
cmd:option('--instancePath', '/media/Nick/coco/annotations/instances_val2014.json', 'path to Coco annotations (instancnes)')
cmd:option('--debugPath', '/media/Nick/coco/debug', 'path to debug folder')
cmd:option('--overwrite', false, 'overwrite cache')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--epochSize', 2000, 'number of train examples seen between each epoch')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--testAsyncSub', false, 'test asynchronous sub')
cmd:option('--testAsyncSample', false, 'test asynchronous sample')
cmd:option('--nThread', 2, 'number of threads')
cmd:text()
opt = cmd:parse(arg or {})


validSet = dp.CocoBoxDetect{
   image_path=opt.imagePath, 
   instance_path=opt.instancePath,
   cache_mode = opt.overwrite and 'overwrite' or 'writeonce'
}

--[[
print"dataset loaded"
batch = validSet:sample(32)

input = batch:inputs():input()
for i=1,input:size(1) do
   image.save(paths.concat(opt.debugPath, 'input'..i..'img.png'), input[i]:narrow(1,1,3))
   image.save(paths.concat(opt.debugPath, 'input'..i..'mask.png'), input[i]:narrow(1,4,1))
end

batch = validSet:sample(128)
print(batch:inputs():view(), batch:inputs():input():size(), batch:targets():input())

batch = validSet:index(torch.LongTensor(128):random(1,10000))
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
   print("testAsyncSub batches")
   for k=1,2 do
      local batch2, i, n
      local sampler2 = samplerB:sampleEpoch(validSet)
      while true do
         batch2, i, n = sampler2(batch2)
         if not batch2 then
            break
         end
         local sumi2, sumt2 = batch2:inputs():forward():sum(), batch2:targets():forward()[1]:sum()
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
         local sumi, sumt = batch:inputs():forward():sum(), batch:targets():forward()[1]:sum()
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

os.exit()

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

