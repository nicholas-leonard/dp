require 'dp'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test CocoDetectSet')
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
cmd:option('--small', false, 'test 10 categories')
cmd:option('--testValid', false, 'test on valid set')
cmd:option('--nThread', 2, 'number of threads')
cmd:text()
opt = cmd:parse(arg or {})


ds = dp.CocoDetect{
   data_path=opt.dataPath, load_all=false,
   cache_mode = opt.overwrite and 'overwrite' or 'writeonce',
   category_ids = opt.small and {10,11,12,13,14,15,16,17,18,19,20} or nil
}

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.min(img:size(3), math.max(1, x1)), math.min(img:size(2), math.max(1, y1))
    x2, y2 = math.min(img:size(3), math.max(1, x2)), math.min(img:size(2), math.max(1, y2))

    local max = img:max()

   print(img:size(), x1, y1, x2, y2)
    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

if opt.testAsyncSub then
   ds:loadValid()
   dataSet = ds:validSet()
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
   ds:loadTrain()
   dataSet = ds:trainSet()
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
   ds:loadTrain()
   ds:loadValid()
   print("nClasses", #ds:trainSet():classes())
   print(unpack(ds:trainSet():classes()))
   print(unpack(ds:validSet():classes()))
   assert(_.same(ds:trainSet():classes(), ds:validSet():classes()))
   print(unpack(_.map(ds:trainSet().classMap, function(k,v) return v[3] end)))
elseif opt.testValid then
   ds:loadValid()
   dataSet = ds:validSet()
   batch = dataSet:sub(1, opt.batchSize)

   inputs = batch:inputs():input()
   inputs = inputs:view(opt.batchSize, 10, 3, inputs:size(3), inputs:size(4))
   bboxes, classes, offsets, sizes = unpack(batch:targets():input())
   classNames = dataSet:classes()
   
   for i=1,inputs:size(1) do
      local input_ = inputs[i]
      local bbox = bboxes[i]
      local class = classes[i]
      local offset = offsets[i]
      local size = sizes[i]
      local W,H = size[1], size[2]
      
      for k=1,10 do
         local input = input_[k]
         
         local off = offset[k]
         local xOff, yOff, hflip = off[1], off[2], off[3]
         
         local cns = {}
         local cns2 = {}
         
         for j=1,bbox:size(1) do
            local c = class[j]
            if c == 0 then
               break
            end
            
            if not cns2[c] then
               cns2[c] = true
               table.insert(cns, classNames[c])
            end
            
            local x,y,w,h = unpack(bbox[j]:totable())
            
            x,y = x-xOff, y-yOff
            
            if hflip > 1/2 then
               x = input:size(3)-(x+w)
            end
            
            drawBox(input, {x,y,w,h})
         end
         image.save(paths.concat(opt.debugPath, 'input'..i..'.'..k..table.concat(cns, '-')..'.png'), input)
      end
      
   end
   
   print("done saving")

   batch = dataSet:index(torch.LongTensor(opt.batchSize):random(1,100))
   print(batch:inputs():view(), batch:inputs():input():size())

   batch = dataSet:sub(100, 200)
   print("sub1", batch:inputs():view(), batch:inputs():input():size())

   dataSet:sub(batch, 200, 240)
   print("sub2", batch:inputs():view(), batch:inputs():input():size())
else
   ds:loadTrain()
   dataSet = ds:trainSet()
   batch = dataSet:sample(opt.batchSize)

   inputs = batch:inputs():input()
   bboxes, classes = unpack(batch:targets():input())
   classNames = dataSet:classes()
   
   for i=1,inputs:size(1) do
      local input = inputs[i]
      local bbox = bboxes[i]
      local class = classes[i]
      
      local cns = {}
      local cns2 = {}
      
      for j=1,bbox:size(1) do
         local c = class[j]
         if c == 0 then
            break
         end
         
         if not cns2[c] then
            cns2[c] = true
            table.insert(cns, classNames[c])
         end
         drawBox(input, bbox[j]:totable())
      end
      image.save(paths.concat(opt.debugPath, 'input'..i..table.concat(cns, '-')..'.png'), input)
   end
   
   print("done saving")

   batch = dataSet:sample(opt.batchSize)
   print(batch:inputs():view(), batch:inputs():input():size(), batch:targets():input())

   batch = dataSet:index(torch.LongTensor(opt.batchSize):random(1,10000))
   print(batch:inputs():view(), batch:inputs():input():size())

   batch = dataSet:sub(100, 200)
   print("sub1", batch:inputs():view(), batch:inputs():input():size())

   dataSet:sub(batch, 200, 240)
   print("sub2", batch:inputs():view(), batch:inputs():input():size())
end



