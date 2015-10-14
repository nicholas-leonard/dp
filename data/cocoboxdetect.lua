------------------------------------------------------------------------
--[[ CocoBoxDetect ]]--
-- Wraps the MS COCO bounding box (bbox) detection dataset.
-- The input is 4 channel image : rgb + mask. The mask channel will 
-- contain all the previously detected (known) instance bboxes.
-- The target is the remaining (unknown) instance bboxes and classes.
-- For training, the instances are randomly split between known/unkown.
-- For evaluation, all instances are unknown. The CocoEvaluator will 
-- be responsible for iterating the model, updating the next input mask
-- with the previous predicted bboxes.
-- Note : some images have no instances
------------------------------------------------------------------------
local CocoBoxDetect, parent = torch.class("dp.CocoBoxDetect", "dp.DataSet")

CocoBoxDetect._input_shape = 'bchw' -- image
CocoBoxDetect._output_shape = {'bf', 'b'} -- bbox(x,y,w,h), class

function CocoBoxDetect:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, image_path, instance_path, input_size, which_set,  
      verbose, cache_mode, cache_path, evaluate = xlua.unpack(
      {config},
      'CocoBoxDetect', 
      'A DataSet for the MS COCO detection challenge.',
      {arg='image_path', type='string', req=true,
       help='path to directory containing images'},
      {arg='instance_path', type='string', req=true,
       help='path to JSON instance (instance) file'},
      {arg='input_size', type='number', default=256,
       help='size (height=width) of the image. Padding will be added '..
       'around non-square images, which will be centered in the input'},
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
       help='Path to cache. Defaults to [image_path]/[which_set]_[input_size]_cache.t7'},
      {arg='evaluate', type='boolean', default=false,
       help='set this to true for evaluation using the CocoEvaluator'}
   )
   -- globals
   gm = require 'graphicsmagick'
   
   -- locals
   self:whichSet(which_set)
   self._image_path = image_path
   self._instance_path = instance_path
   self._input_size = input_size
   self._verbose = verbose
   self._evaluate = evaluate
   
   self._cache_mode = cache_mode
   self._cache_path = cache_path or paths.concat(self._image_path, which_set..'_'..self._input_size..'_cache.t7')
   
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

function CocoBoxDetect:buildIndex()
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
   local nClass = 0
   self._classes = _.map(data.categories, 
      function(class_id,category) 
         self.category[category.id] = class_id
         self.classMap[class_id] = {{}, 0, category.id}
         nClass = nClass + 1
         return category.name 
      end)
   assert(nClass == #self._classes)
   data.categories = nil
   
   dp.vprint(self._verbose, "Building data indexes")
   local nInst = #data.annotations
   for i,inst in ipairs(data.annotations) do
      -- class/category
      local class_id = self.category[inst.category_id]
      assert(class_id)
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

function CocoBoxDetect:saveIndex()
   local index = {}
   for i,k in ipairs{'imageMap','_classes','category','classMap','instanceMap','instanceIds', 'imageIds','_max_instance'} do
      index[k] = self[k]
   end
   torch.save(self._cache_path, index)
end

function CocoBoxDetect:loadIndex()
   local index = torch.load(self._cache_path)
   for k,v in pairs(index) do
      self[k] = v
   end
end

-- evaluation iterates through images
-- training iterates through instances
function CocoBoxDetect:nSample()
   return self._evaluate and self.imageIds:size(1) or self.instanceIds:size(1)
end

-- Get a sample. 
-- Input is concatenation of mask (known instance + padding) with image.
-- Target includes unknown bboxes (x,y,w,h) and classes.
-- For evaluation, idx indexes image, 
-- and no instances are known, all are unknown.
-- For training, idx indexes instance, 
-- and the number of known instances is sampled uniformly.
-- The known instances are included in the input mask.
-- The unkown instances are included as targets.
-- There are multiple target instances (classes + bbox).
function CocoBoxDetect:getSample(input, bbox, class, idx)
   assert(input and bbox and class and idx)
   class:zero()
   bbox:zero()
   local imageId, instanceId, imagePath
   if self._evaluate then
      imageId = self.imageIds[idx]
      nKnown = 0
   else
      local instanceId = self.instanceIds[idx]
      imageId = self.instanceMap[instanceId][2]
      nKnown = torch.random(0,#self.imageMap[imageId][2]-1)
   end
   
   local imagePath = paths.concat(self._image_path, self.imageMap[imageId][1])
   
   -- load image with size hints
   local gmImg = gm.Image():load(imagePath, self._input_size, self._input_size)
   -- resize by imposing the largest dimension (while keeping aspect ratio)
   local iW, iH = gmImg:size()
   if iW/iH < 1 then
      gmImg:size(self._input_size)
   else
      gmImg:size(self._input_size)
   end
   local oW, oH = gmImg:size()
   assert(oW <= self._input_size)
   assert(oH <= self._input_size)
   
   local img = gmImg:toTensor('float','RGB','DHW', true)
   
   -- insert image
   local resImg = input:narrow(1,1,3):zero()
   local padW = torch.round((self._input_size - oW)/2)
   local padH = torch.round((self._input_size - oH)/2)
   resImg:narrow(2,padH+1,oH):narrow(3,padW+1,oW):copy(img)
      
   -- mask padding
   local resMask = input:narrow(1,4,1):fill(1)
   resMask:narrow(2,padH+1,oH):narrow(3,padW+1,oW):zero()
   
   -- get known instances (previously found, masked in input)
   -- and unknown instances (possible target classes and bboxes)   
   local unknown, known
   if self._evaluate then
      unknown = _.clone(self.imageMap[imageId][2])
      known = {} 
   else
      local instanceIds = _.shuffle(self.imageMap[imageId][2])
      instanceIds = _.pull(instanceIds, instanceId) -- remove main instance Id
      unknown =  _.first(instanceIds, #instanceIds - nKnown)
      known = _.last(instanceIds, #instanceIds - #unknown) or {}
      table.insert(unknown, instanceId) -- put back main instance Id
   end
   
   -- mask all known instance bounding box
   local imgData = self.imageMap[imageId]
   local iH, iW = imgData[3], imgData[4]
   local sW, sH = oW/iW, oH/iH
   for i,knownId in ipairs(known) do
      local x,y,w,h = unpack(self.instanceMap[knownId][3])
      -- rescale between 1 and (self._input_size - 1)
      x,y,w,h = x*sW, y*sH, w*sW, h*sH
      -- translate 
      x, y = x + padW, y + padH
      -- make safe (+1 is because x,y are zero-indexed)
      local x2 = math.max(1, math.min(self._input_size, torch.round(x+w+1)))
      local y2 = math.max(1, math.min(self._input_size, torch.round(y+h+1)))
      local x1 = math.max(1, math.min(self._input_size, torch.round(x+1)))
      local y1 = math.max(1, math.min(self._input_size, torch.round(y+1)))
      resMask:narrow(2,y1,y2-y1+1):narrow(3,x1,x2-x1+1):fill(1)
   end
   
   -- include unknown instance bounding box and classes as targets 
   for i,unknownId in ipairs(unknown) do
      local unknownInstance = self.instanceMap[unknownId]
      local x,y,w,h = unpack(unknownInstance[3])
      -- rescale
      x,y,w,h = x*sW, y*sH, w*sW, h*sH
      -- translate
      x, y = x + padW, y + padH
      -- top left x,y, bottom right x,y, instead of x, y, w, h
      local bb = bbox[i]
      bb[1], bb[2], bb[3], bb[4] = x, y, x+w, y+h
      -- instance class
      class[i] = unknownInstance[1]
   end
   
   -- rescale : -1 to 1 (0,0 is center)
   if #unknown > 0 then
      bbox:narrow(1,1,#unknown):div((self._input_size-1)/2):add(-1)
   end
   
   return input, bbox, class
end

function CocoBoxDetect:batch(batch_size)
   return dp.Batch{
      which_set=self._which_set,
      inputs=dp.ImageView('bchw', torch.FloatTensor(batch_size, 4, self._input_size, self._input_size)),
      targets=dp.ListView{
         dp.SequenceView('bwc', torch.FloatTensor(batch_size, self._max_instance, 4)),
         dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance))
      }
   }
end

function CocoBoxDetect:sub(batch, start, stop)
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
-- The number of unknown instances is sampled uniformly
function CocoBoxDetect:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

   local batch_size = indices:size(1)
   
   -- target : {bboxes, classes}
   local targetView = batch and batch:targets() or dp.ListView{
      dp.SequenceView('bwc', torch.FloatTensor(batch_size, self._max_instance, 4)),
      dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance))
   }
   local bboxes, classes = unpack(targetView:input())
   bboxes:resize(batch_size, self._max_instance, 4)
   classes:resize(batch_size, self._max_instance)
   
   -- input : an rgb image concatenated with a mask
   local inputView = batch and batch:inputs() or dp.ImageView()
   local inputs = inputView:input() or torch.FloatTensor()
   inputs:resize(batch_size, 4, self._input_size, self._input_size) -- rgb + mask
   inputView:forward('bchw', inputs)
   
   for i=1,batch_size do
      local idx = indices[i]
      self:getSample(inputs[i], bboxes[i], classes[i], idx)
   end
   
   targetView:forward({'bwc','bt'}, {bboxes, classes})
   targetView:components()[2]:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  
   return batch
end

-- Uniformly sample a class, then an instance of that class,
-- then the number of unknown instances (1->numInstance(instance.img))
-- This keeps the class distribution somewhat balanced.
-- "Somewhat" since the sampled instance will not necessarily be focused upon.
function CocoBoxDetect:sample(batch, nSample)
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

function CocoBoxDetect:classes()
   return self._classes
end
------------------------ multithreading --------------------------------

function CocoBoxDetect:multithread(nThread)
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
      
         dataset = dp.CocoBoxDetect(config)
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

function CocoBoxDetect:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

-- send request to worker : put request into queue
function CocoBoxDetect:subAsyncPut(batch, start, stop, callback)   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(stop-start+1)
   end
   local input = batch:inputs():input()
   local bbox, class = unpack(batch:targets():input())
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
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         tbatch:inputs():forward('bchw', input)
         tbatch:targets():forward({'bwc', 'bt'}, {bbox, class})
         
         dataset:sub(tbatch, start, stop)
         
         return input, target
      end,
      -- the endcallback (runs in the main thread)
      function(input, target)
         local batch = self._send_batches:get()
         batch:inputs():forward('bchw', input)
         batch:targets():forward({'bwc', 'bt'}, {bbox, class})
         
         callback(batch)
         
         batch:targets():components()[2]:setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

function CocoBoxDetect:sampleAsyncPut(batch, nSample, funcName, callback)
   assert(not funcName)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async CocoBoxDetect (for now)'
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
         return input, target
      end,
      -- the endcallback (runs in the main thread)
      function(input, target)
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
function CocoBoxDetect:asyncGet()
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

