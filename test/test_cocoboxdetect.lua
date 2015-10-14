require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test CocoBoxDetect')
cmd:text('Options:')
cmd:option('--dataPath', '/media/nicholas14/Nick/coco')
cmd:option('--debugPath', '/media/nicholas14/Nick/coco/debug', 'path to debug folder')
cmd:option('--overwrite', false, 'overwrite cache')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--epochSize', 2000, 'number of train examples seen between each epoch')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--testAsyncSub', false, 'test asynchronous sub')
cmd:option('--testAsyncSample', false, 'test asynchronous sample')
cmd:option('--testClassMap', false, 'test that class map matches for train and test')
cmd:option('--nThread', 2, 'number of threads')
cmd:text()
opt = cmd:parse(arg or {})


ds = dp.CocoDetect{
   data_path=opt.dataPath, load_all=false,
   cache_mode = opt.overwrite and 'overwrite' or 'writeonce'
}
if opt.testAsyncSample then
   ds:loadTrain()
   dataSet = ds:trainSet()
elseif opt.testClassMap then
   ds:loadTrain()
   ds:loadValid()
else
   ds:loadValid()
   dataSet = ds:validSet()
end
print"dataset loaded"

if opt.testAsyncSub then
   samplerB = dp.Sampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
   
   local isum2, tsum2, bsum2 = 0, 0, 0
   local a = torch.Timer()
   local nBatch2 = 0
   print("testAsyncSub batches")
   for k=1,2 do
      local batch2, i, n
      local sampler2 = samplerB:sampleEpoch(dataSet)
      while true do
         batch2, i, n = sampler2(batch2)
         if not batch2 then
            break
         end
         local sumi2, sumb2, sumt2 = batch2:inputs():forward():sum(), batch2:targets():forward()[1]:sum(), batch2:targets():forward()[2]:sum()
         isum2 = isum2 + sumi2
         bsum2 = bsum2 + sumb2
         tsum2 = tsum2 + sumt2
         nBatch2 = nBatch2 + 1
      end
   end
   print("sync", (a:time().real)/nBatch2)
   
   dataSet:multithread(opt.nThread)
   print"multithread"
   samplerA = dp.Sampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
   samplerA:async()
   
   local a = torch.Timer()
   local nBatch = 0
   local isum, tsum, bsum = 0, 0, 0
   for k=1,2 do
      local batch, i, n
      local sampler = samplerA:sampleEpoch(dataSet)
      print(k)
      while true do
         batch, i, n = sampler(batch)
         if not batch then
            break
         end
         assert(batch.isBatch)
         local sumi, sumb, sumt = batch:inputs():forward():sum(), batch:targets():forward()[1]:sum(), batch:targets():forward()[2]:sum()
         isum = isum + sumi
         bsum = bsum + sumb
         tsum = tsum + sumt
         nBatch = nBatch + 1
      end
   end
   print("async", (a:time().real)/nBatch)
   
   print("sums", isum, isum2, bsum, bsum2, tsum, tsum2, "counts", nBatch, nBatch2)
   
   assert(isum == isum2)
   assert(bsum == bsum2)
   assert(tsum == tsum2)
   assert(nBatch == nBatch2)
elseif opt.testAsyncSample then
   samplerB = dp.RandomSampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
      
   local a = torch.Timer()
   local nBatch2 = 0
   for k=1,2 do
      local batch2, i, n
      local sampler2 = samplerB:sampleEpoch(dataSet)
      while true do
         batch2, i, n = sampler2(batch2)
         if not batch2 then
            break
         end
         nBatch2 = nBatch2 + 1
      end
   end
   print("sync", nBatch2, (a:time().real)/nBatch2)
   
   dataSet:multithread(opt.nThread, opt.nThread*2)
   samplerA = dp.RandomSampler{batch_size=math.floor(opt.batchSize), epoch_size=opt.epochSize}
   samplerA:async()
   
   local a = torch.Timer()
   local nBatch = 0
   for k=1,2 do
      local batch, i, n
      local sampler = samplerA:sampleEpoch(dataSet)
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
elseif opt.testClassMap then
   print(unpack(ds:trainSet():classes()))
   print(unpack(ds:validSet():classes()))
   assert(_.same(ds:trainSet():classes(), ds:validSet():classes()))
   print(unpack(_.map(ds:trainSet().classMap, function(k,v) return v[3] end)))
else
   batch = dataSet:sample(32)

   input = batch:inputs():input()
   for i=1,input:size(1) do
      image.save(paths.concat(opt.debugPath, 'input'..i..'img.png'), input[i]:narrow(1,1,3))
      image.save(paths.concat(opt.debugPath, 'input'..i..'mask.png'), input[i]:narrow(1,4,1))
   end

   batch = dataSet:sample(128)
   print(batch:inputs():view(), batch:inputs():input():size(), batch:targets():input())

   batch = dataSet:index(torch.LongTensor(128):random(1,10000))
   print(batch:inputs():view(), batch:inputs():input():size())

   batch = dataSet:sub(100, 200)
   print("sub1", batch:inputs():view(), batch:inputs():input():size())

   dataSet:sub(batch, 200, 240)
   print("sub2", batch:inputs():view(), batch:inputs():input():size())
end



