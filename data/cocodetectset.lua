------------------------------------------------------------------------
--[[ CocoDetectSet ]]--
-- Wraps the MS COCO bounding box (bbox) detection dataset.
-- The input is 3 channel image : rgb 
-- The target contains many instance bboxes and classes.
------------------------------------------------------------------------
local CocoDetectSet, parent = torch.class("dp.CocoDetectSet", "dp.DataSet")

CocoDetectSet._input_shape = 'bchw' -- image
CocoDetectSet._output_shape = {'bf', 'b'} -- bbox(x,y,w,h), class

function CocoDetectSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, image_path, instance_path, load_size, sample_size, which_set,  
      verbose, cache_mode, cache_path, evaluate, category_ids
      = xlua.unpack(
      {config},
      'CocoDetectSet', 
      'A DataSet for the MS COCO detection challenge.',
      {arg='image_path', type='string', req=true,
       help='path to directory containing images'},
      {arg='instance_path', type='string', req=true,
       help='path to JSON instance (instance) file'},
      {arg='load_size', type='number', default=256,
       help='a size to load the images to, initially'},
      {arg='sample_size', type='number', default=224,
       help='a consistent sample size to resize the images.'},
      {arg='which_set', type='string', default='train',
       help='"train", "valid" or "test" set'},
      {arg='verbose', type='boolean', default=true,
       help='display verbose messages'},
      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'},
      {arg='cache_path', type='string',
       help='Path to cache. Defaults to [image_path]/[which_set]_cache.t7'},
      {arg='evaluate', type='boolean', default=false,
       help='set this to true for evaluation using the CocoEvaluator'},
      {arg='category_ids', type='table',
       help='use a subset of categoryIds'}
   )
   -- globals
   gm = require 'graphicsmagick'
   
   -- locals
   self:whichSet(which_set)
   self._image_path = image_path
   self._instance_path = instance_path
   self._load_size = load_size
   self._sample_size = sample_size
   self._verbose = verbose
   self._evaluate = evaluate
   self._category_ids = category_ids
   
   self._cache_mode = cache_mode
   self._cache_path = cache_path or paths.concat(self._image_path, which_set..(category_ids and table.concat(category_ids,'-') or '')..'_cache.t7')
   
   -- indexing and caching
   assert(_.find({'writeonce','overwrite','nocache','readonly'},cache_mode), 'invalid cache_mode :'..cache_mode)
   local cacheExists = paths.filep(self._cache_path)
   if cache_mode == 'readonly' or (cache_mode == 'writeonce' and cacheExists) then
      if not cacheExists then
         error"'readonly' cache_mode requires an existing cache, none found"
      end
      self:loadIndex()
   else
      self:buildIndex()
      if cache_mode ~= 'nocache' then
         self:saveIndex()
      end
   end
   
   self._config = config 
end

function CocoDetectSet:buildIndex()
   -- load json instances
   local file = io.open(self._instance_path)
   local blocks = {}
   while true do
      local block = file:read(1000000)
      if not block then break end
      table.insert(blocks, block)
   end
   local data = table.concat(blocks)
   data = torch.json.decode(data)
   collectgarbage()
   
   -- maps 
   self.classMap = {} -- class-id -> {instance-ids, frequency, category-id}
   self.instanceMap = {} -- instance-id -> {class-id, image-id, bbox, instance-idx}
   self.imageMap = {} -- image-id -> {filename, instance-ids, height, width}
   
   for i,img in ipairs(data.images) do
      self.imageMap[img.id] = {img.file_name, {}, img.height, img.width}
   end
   
   -- category-id -> class-id
   self.category = {} 
   -- class-id -> category_name
   local classId = 1
   self._classes = {}
   for i,category in ipairs(data.categories) do
      if not (self._category_ids and not _.contains(self._category_ids, category.id)) then
         self.category[category.id] = classId
         self.classMap[classId] = {{}, 0, category.id}
         self._classes[classId] = category.name 
         classId = classId + 1
      end
   end
   data.categories = nil
   
   dp.vprint(self._verbose, "Building data indexes")
   local nInst = #data.annotations
   for i,inst in ipairs(data.annotations) do
      -- class/category
      local class_id = self.category[inst.category_id]
      if class_id then
         -- frequency
         self.classMap[class_id][2] = self.classMap[class_id][2] + 1
         
         -- add instance to map
         self.instanceMap[inst.id] = {class_id, inst.image_id, inst.bbox}
         
         -- add instance to image
         table.insert(self.imageMap[inst.image_id][2], inst.id)
         
         -- add instance to class
         local class = self.classMap[class_id]
         table.insert(class[1], inst.id)
         
         -- delete from data as we index
         data.annotations[i] = nil
         if i % 100000 == 0 then
            collectgarbage()
         end
         if self._verbose then
            xlua.progress(i, nInst)
         end
      end
   end
   
   -- remove images having no instances
   for imageId, imageData in pairs(self.imageMap) do
      if #imageData[2] == 0 then
         self.imageMap[imageId] = nil
      end
   end
   
   data = nil
   collectgarbage()
   
   self.instanceIds = torch.LongTensor(_.keys(self.instanceMap))
   self.imageIds = torch.LongTensor(_.keys(self.imageMap))
   
   if self._verbose then
      print("nInstance = "..self.instanceIds:size(1))
      print("nImage = "..self.imageIds:size(1))
   end
   
   print("Building instance key index")
   for instanceIdx=1,self.instanceIds:size(1) do
      local instanceId = self.instanceIds[instanceIdx]
      self.instanceMap[instanceId][4] = instanceIdx
   end
   
   -- what is max number of instances in a single image?
   self._max_instance = 0
   for k,img in pairs(self.imageMap) do
      self._max_instance = math.max(self._max_instance, #img[2])
   end
end

function CocoDetectSet:saveIndex()
   local index = {}
   for i,k in ipairs{'imageMap','_classes','category','classMap','instanceMap','instanceIds', 'imageIds','_max_instance'} do
      index[k] = self[k]
   end
   torch.save(self._cache_path, index)
end

function CocoDetectSet:loadIndex()
   local index = torch.load(self._cache_path)
   for k,v in pairs(index) do
      self[k] = v
   end
end

-- evaluation iterates through images
-- training iterates through instances
function CocoDetectSet:nSample()
   return self._evaluate and self.imageIds:size(1) or self.instanceIds:size(1)
end

function CocoDetectSet:loadImage(path)
   -- load image with size hints
   local input = gm.Image():load(path, self._load_size, self._load_size)
   -- resize by imposing the smallest dimension (while keeping aspect ratio)
   local iW, iH = input:size()
   input:size(nil, self._load_size)
   return input
end

-- Get a sample. 
-- For evaluation, idx indexes image. 
-- Input is 10 crops (corners, center, mirror).
-- For training, idx indexes instance.
-- Input is 1 crop around instance.
-- Target includes bboxes (x,y,w,h) and classes.
-- x,y is coordinate of the center of the patch
-- There are multiple target instances (classes + bbox) per image.
function CocoDetectSet:getSample(idx, input, bbox, class, offset, size)
   assert(input and bbox and class and idx)
   class:zero()
   bbox:zero()
   
   local imageId, instanceId, imagePath
   if self._evaluate then
      offset:zero()
      imageId = self.imageIds[idx]      
   else
      instanceId = self.instanceIds[idx]
      imageId = self.instanceMap[instanceId][2]
   end
   
   local imagePath = paths.concat(self._image_path, self.imageMap[imageId][1])
   
   -- load image with size hints
   local gmImg = self:loadImage(imagePath)
   local oW, oH = gmImg:size()
   assert(oW >= self._load_size)
   assert(oH >= self._load_size)
   
   local imgData = self.imageMap[imageId]
   local iH, iW = imgData[3], imgData[4]
   local sW, sH = oW/iW, oH/iH
   
   local img = gmImg:toTensor('float','RGB','DHW', true)
   
   local xOffset, yOffset, hflip = 0, 0, 0
   if self._evaluate then
      assert(offset)
      -- input is 10 crops       
      local iW, iH = gmImg:size()
      local oW, oH = self._sample_size, self._sample_size
      
      -- CocoEvaluator needs the original size in order to merge the 10 outputs
      size[1], size[2] = oW, oH
      
      local dst = input:view(10, 3, oW, oH)

      local w1 = math.ceil((iW-oW)/2)
      local h1 = math.ceil((iH-oH)/2)
      -- center
      image.crop(dst[1], img, w1, h1)
      offset[{1,1}], offset[{1,2}] = w1, h1 
      image.hflip(dst[2], dst[1])
      offset[{2,1}], offset[{2,2}], offset[{2,3}] = w1, h1, 1
      -- top-left
      h1 = 0; w1 = 0;
      image.crop(dst[3], img, w1, h1) 
      offset[{3,1}], offset[{3,2}] = w1, h1 
      dst[4] = image.hflip(dst[3])
      offset[{4,1}], offset[{4,2}], offset[{4,3}] = w1, h1, 1
      -- top-right
      h1 = 0; w1 = iW-oW;
      image.crop(dst[5], img, w1, h1) 
      offset[{5,1}], offset[{5,2}] = w1, h1 
      image.hflip(dst[6], dst[5])
      offset[{6,1}], offset[{6,2}], offset[{6,3}] = w1, h1, 1
      -- bottom-left
      h1 = iH-oH; w1 = 0;
      image.crop(dst[7], img, w1, h1) 
      offset[{7,1}], offset[{7,2}] = w1, h1 
      image.hflip(dst[8], dst[7])
      offset[{8,1}], offset[{8,2}], offset[{8,3}] = w1, h1, 1
      -- bottom-right
      h1 = iH-oH; w1 = iW-oW;
      image.crop(dst[9], img, w1, h1) 
      offset[{9,1}], offset[{9,2}] = w1, h1 
      image.hflip(dst[10], dst[9])
      offset[{10,1}], offset[{10,2}], offset[{10,3}] = w1, h1, 1
   else   
      -- crop around the instance 
      local x,y,w,h = unpack(self.instanceMap[instanceId][3])
      -- rescale between 0 and (size - 1)
      x,y,w,h = x*sW, y*sH, w*sW, h*sH
      -- make safe (+1 is because x,y are zero-indexed)
      local x2 = math.max(1, math.min(oW, torch.round(x+w+1)))
      local y2 = math.max(1, math.min(oH, torch.round(y+h+1)))
      local x1 = math.max(1, math.min(oW, torch.round(x+1)))
      local y1 = math.max(1, math.min(oH, torch.round(y+1)))
      -- random crop coord (of top-left corner) that includes instance
      local xCrop = torch.uniform(x2-self._sample_size+1, x1)
      xCrop = math.max(1, math.min(oW-self._sample_size+1, xCrop)) 
      local yCrop = torch.uniform(y2-self._sample_size+1, y1)
      yCrop = math.max(1, math.min(oH-self._sample_size+1, yCrop))
      -- crop it
      local crop = img:narrow(2,yCrop,self._sample_size):narrow(3,xCrop,self._sample_size)
      
      xOffset, yOffset = xCrop-1, yCrop-1
      hflip = math.random(0,1)
      if hflip > 1/2 then
         image.hflip(input, crop)
      else
         input:copy(crop)
      end
   end
   
   -- targets : bounding box and classes
   for i,instanceId in ipairs(imgData[2]) do
      local instance = self.instanceMap[instanceId]
      local x,y,w,h = unpack(instance[3])
      -- rescale between (0,0) and ((oW-1),(oH-1))
      x,y,w,h = x*sW, y*sH, w*sW, h*sH
      -- center to cropped area (affects training only)
      x,y = x-xOffset, y-yOffset
      -- horizontal flip ?
      if hflip > 1/2 then
         assert(not self._evaluate)
         x = self._sample_size - (x + w)
      end
      -- bounding box
      local bb = bbox[i]
      bb[1], bb[2], bb[3], bb[4] = x, y, w, h
      -- instance class
      class[i] = instance[1]
   end
   
   return input, bbox, class, offset
end

function CocoDetectSet:batch(batch_size)
   return dp.Batch{
      which_set=self._which_set,
      inputs=dp.ImageView('bchw', torch.FloatTensor(batch_size, self._evaluate and 30 or 3, self._sample_size, self._sample_size)),
      targets=dp.ListView{
         dp.SequenceView('bwc', torch.FloatTensor(batch_size, self._max_instance, 4)),
         dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance)),
         self._evaluate and dp.SequenceView('bwc', torch.FloatTensor(batch_size, 10, 3)) or nil,
         self._evaluate and dp.DataView('bf', torch.FloatTensor(batch_size, 2)) or nil,
      }
   }
end

function CocoDetectSet:sub(batch, start, stop)
   if not stop then
      stop = start
      start = batch
      batch = nil
   end
   
   self._sub_index = self._sub_index or torch.LongTensor()
   self._sub_index:range(start, stop)
   
   return self:index(batch, self._sub_index)
end

-- For evaluation, indices index images ;
-- for training, they index instances.
function CocoDetectSet:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

   local batch_size = indices:size(1)
   
   -- target : {bboxes, classes, [offsets]}
   local targets = {
      dp.SequenceView('bwc', torch.FloatTensor(batch_size, self._max_instance, 4)),
      dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance))
   }
   if self._evaluate then
      table.insert(targets, dp.SequenceView('bwc', torch.FloatTensor(batch_size, 10, 3)))
      table.insert(targets, dp.DataView('bf', torch.FloatTensor(batch_size, 2)))
   end
   local targetView = batch and batch:targets() or dp.ListView(targets)
   local bboxes, classes, offsets, sizes = unpack(targetView:input())
   bboxes:resize(batch_size, self._max_instance, 4)
   classes:resize(batch_size, self._max_instance)
   if self._evaluate then
      offsets:resize(batch_size, 10, 3)
      sizes:resize(batch_size, 2)
   end
   
   local inputView = batch and batch:inputs() or dp.ImageView()
   local inputs = inputView:input() or torch.FloatTensor()
   inputs:resize(batch_size, self._evaluate and 30 or 3, self._sample_size, self._sample_size) -- rgb
   inputView:forward('bchw', inputs)
   
   for i=1,batch_size do
      self:getSample(indices[i], inputs[i], bboxes[i], classes[i], self._evaluate and offsets[i], self._evaluate and sizes[i])
   end
   
   if self._evaluate then
      targetView:forward({'bwc','bt', 'bwc', 'bf'}, {bboxes, classes, offsets, sizes})
   else
      targetView:forward({'bwc','bt'}, {bboxes, classes})
   end
   targetView:components()[2]:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  
   return batch
end

-- Uniformly sample a class, then an instance of that class
function CocoDetectSet:sample(batch, nSample)
   assert(not self._evaluate, "sample only works with evaluate=false")
   if not nSample then
      nSample = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}
   
   self._sample_index = self._sample_index or torch.LongTensor()
   self._sample_index:resize(nSample)
   
   for i=1,nSample do
      -- sample class
      local classId = torch.random(1, #self._classes)
      -- sample instance from class
      local classInstances = self.classMap[classId][1]
      local instanceId = classInstances[torch.random(1, #classInstances)]
      -- instanceId to instanceIdx
      self._sample_index[i] = self.instanceMap[instanceId][4]
   end
   return self:index(batch, self._sample_index)
end

function CocoDetectSet:classes()
   return self._classes
end
------------------------ multithreading --------------------------------

function CocoDetectSet:multithread(nThread)
   nThread = nThread or 2
   if not paths.filep(self._cache_path) then
      -- workers will read a serialized index to speed things up
      self:saveIndex()
   end
   
   local mainSeed = os.time()
   local config = self._config
   config.cache_mode = 'readonly'
   config.verbose = self._verbose
   
   local threads = require "threads"
   threads.Threads.serialization('threads.sharedserialize')
   self._threads = threads.Threads(
      nThread,
      function()
         require 'dp'
      end,
      function(idx)
         tid = idx
         local seed = mainSeed + idx
         math.randomseed(seed)
         torch.manualSeed(seed)
         if config.verbose then
            print(string.format('Starting worker thread with id: %d seed: %d', tid, seed))
         end
      
         dataset = dp.CocoDetectSet(config)
         tbatch = dataset:batch(1)
      end
   )
   
   self._send_batches = dp.Queue() -- batches sent from main to threads
   self._recv_batches = dp.Queue() -- batches received in main from threads
   self._buffer_batches = dp.Queue() -- buffered batches
   
   -- public variables
   self.nThread = nThread
   self.isAsync = true
end

function CocoDetectSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

-- send request to worker : put request into queue
function CocoDetectSet:subAsyncPut(batch, start, stop, callback)   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(stop-start+1)
   end
   local input = batch:inputs():input()
   local bbox, class, offset, size = unpack(batch:targets():input())
   assert(input and class and bbox)
   
   self._send_batches:put(batch)
   
   for i=1,1000 do
      if self._threads:acceptsjob() then
         break
      else
         sys.sleep(0.01)
      end
      if i==1000 then
         error"infinit loop"
      end
   end
   
   local evaluate = self._evaluate
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         tbatch:inputs():forward('bchw', input)
         if evaluate then
            tbatch:targets():forward({'bwc', 'bt', 'bwc', 'bf'}, {bbox, class, offset, size})
         else
            tbatch:targets():forward({'bwc', 'bt'}, {bbox, class})
         end
         
         dataset:sub(tbatch, start, stop)
         
         return input, bbox, class, offset, size
      end,
      -- the endcallback (runs in the main thread)
      function(input, bbox, class, offset, size)
         local batch = self._send_batches:get()
         batch:inputs():forward('bchw', input)
         if evaluate then
            batch:targets():forward({'bwc', 'bt', 'bwc', 'bf'}, {bbox, class, offset, size})
         else
            batch:targets():forward({'bwc', 'bt'}, {bbox, class})
         end
         
         callback(batch)
         
         batch:targets():components()[2]:setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

function CocoDetectSet:sampleAsyncPut(batch, nSample, funcName, callback)
   assert(not funcName)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async CocoDetectSet (for now)'
   end  
   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
   end
   local input = batch:inputs():input()
   local bbox, class = unpack(batch:targets():input())
   assert(input and class and bbox)
   
   self._send_batches:put(batch)
   
   for i=1,1000 do
      if self._threads:acceptsjob() then
         break
      else
         print"sleep2"
         sys.sleep(0.01)
      end
      if i==1000 then
         error"infinite loop"
      end
   end
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         tbatch:inputs():forward('bchw', input)
         tbatch:targets():forward({'bwc', 'bt'}, {bbox, class})
         
         dataset:sample(tbatch, nSample)
         return input, bbox, class
      end,
      -- the endcallback (runs in the main thread)
      function(input, bbox, class)
         local batch = self._send_batches:get()
         batch:inputs():forward('bchw', input)
         batch:targets():forward({'bwc', 'bt'}, {bbox, class})

         callback(batch)

         batch:targets():components()[2]:setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

-- recv results from worker : get results from queue
function CocoDetectSet:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   local i = 0
   while self._recv_batches:empty() do
      self._threads:dojob()
      if self._recv_batches:empty() then
         print("sleep1", i)
         sys.sleep(0.01)
      else
         break
      end
      i = i + 1
      if i == 100 then 
         error"infinite loop" 
      end
   end
   
   return self._recv_batches:get()
end

