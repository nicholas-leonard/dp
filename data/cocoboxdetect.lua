------------------------------------------------------------------------
--[[ CocoBoxDetect ]]--
-- Wraps the MS COCO bounding box detection dataset.
-- The input is 4 channel image : rgb + mask. The mask channel will 
-- contain all the previously detected (known) instnace bboxes.
-- The target is the remaining (unknown) instance bboxes and classes.
-- For training, the instances are randomly split between known/unkown.
-- For evaluation, all instances are unknown. The CocoEvaluator will 
-- be responsible for iterating the model, updating the next input mask
-- with the previous predicted bboxes.
------------------------------------------------------------------------
local CocoBoxDetect, parent = torch.class("dp.CocoBoxDetect", "dp.DataSet")

CocoBoxDetect._input_shape = 'bchw' -- image
CocoBoxDetect._output_shape = {'bf', 'b'} -- bbox(x,y,w,h), class

function CocoBoxDetect:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, image_path, instance_path, input_size, which_set,  
      verbose, cache_mode, cache_path, self._evaluate = xlua.unpack(
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
       help='Path to cache. Defaults to [image_path]/[which_set]_cache.th7'},
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
   
   self._cache_mode = cache_mode
   self._cache_path = cache_path or paths.concat(self._image_path, which_set..'_cache.th7')
   
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
      local block = file:read(10000000)
      if not block then break end
      table.insert(blocks, block)
   end
   local data = table.concat(blocks)
   self.data = torch.json.decode(data)
   
   -- maps 
   self.classMap = {} -- class-id -> {instance-ids, frequency, category-id}
   self.instanceMap = {} -- instance-id -> {class-id, image-id, bbox}
   self.imageMap = {} -- image-id -> {filename, instance-ids}
   
   self._n_sample = #self.data.images
   
   for i,img in ipairs(self.data.images) do
      self.imageMap[img.id] = {img.file_name, {}}
   end
   
   self._classes = {} -- class-id -> category_name
   self.category = {} -- category-id -> class-id
   
   local classIdx = 1
   for i,inst in ipairs(self.data.annotations) do
      -- bounding box
      local bbox = self.bbox[tensorIdx]
      
      -- class/category
      local class_id = self.category[inst.category_id]
      if not class_id then
         self.category[inst.category_id] = classIdx
         self._classes[classIdx] = self.data.categories[inst.category_id].name
         self.classMap[classIdx] = {{}, 0, inst.category_id}
         
         classIdx = classIdx + 1
         class_id = classIdx
      end
      -- frequency
      self.classMap[class_id][2] = self.classMap[class_id][2] + 1
      
      -- add instance to map
      self.instanceMap[inst.id] = {class_id, inst.image_id, bbox}
      
      -- add instance to image
      table.insert(self.imageMap[inst.image_id][2], instance-ids)
      
      -- add instance to class
      local class = self.classMap[class_id]
      table.insert(class[1], inst.id)
   end
   
   -- what is max number of instances in a single image?
   self._max_instance = 0
   for k,img in pairs(self.imageMap) do
      self._max_instance = math.max(self._max_instance, #img[2])
   end
end

function CocoBoxDetect:saveIndex()
   local index = {}
   for i,k in ipairs{'imageMap','_classes','category','classMap','instanceMap','_n_sample','_max_instance'} do
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

function CocoBoxDetect:batch(batch_size)
   return dp.Batch{
      which_set=self._which_set,
      inputs=dp.ImageView('bchw', torch.FloatTensor(batch_size, unpack(self._sample_size))),
      targets=dp.ListView{
         dp.SequenceView('bwc', torch.IntTensor(batch_size, self._max_instance, 4)),
         dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance))
      }
   }
end

function CocoBoxDetect:nSample()
   return self._n_sample
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

function CocoBoxDetect:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

   local batch_size = indices:size(1)
   
   -- target : {bboxes, classes, known}
   local targetView = batch and batch:targets() or dp.ListView{
      dp.SequenceView('bwc', torch.IntTensor(batch_size, self._max_instance, 4)),
      dp.ClassView('bt', torch.IntTensor(batch_size, self._max_instance))
   }
   local bboxes, classes, knowns = unpack(targetView:input())
   bboxes:resize(batch_size, self._max_instance, 4)
   classes:resize(batch_size, self._max_instance)
   
   
   -- input : an rgb image concatenated with a mask
   local inputView = batch and batch:inputs() or dp.ImageView()
   local inputs = inputView:input() or torch.FloatTensor()
   inputs:resizeAs(stop-start+1,4,self._input_size, self._input_size) -- rgb + mask
   inputView:forward('bchw', inputTensor)
   
   for i=1,batch_size do
      local idx = indices[i]
      self:getSample(inputs[idx], bboxes[idx], classes[idx], idx)
   end
   
   targetView:forward({'bwc','bt'}, {bboxes, classes})
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  
   return batch
end

function CocoBoxDetect:getSample(input, bbox, class, known, idx)
   local instanceId = self.instance[idx]
   local imageId = self.instanceMap[instanceId][2]
   local imagePath = paths.concat(self._image_path, self.imageMap[imageId][1])
   
   -- https://github.com/clementfarabet/graphicsmagick#gmimage
   -- load image with size hints
   local input = gm.Image():load(imagePath, self._input_size, self._input_size)
   -- resize by imposing the largest dimension (while keeping aspect ratio)
   local iW, iH = input:size()
   if iW/iH < 1 then
      input:size(nil, self._input_size)
   else
      input:size(self._input_size, nil)
   end
   local oW, oH = input:size()
   assert(oW <= self._input_size)
   assert(oH <= self._input_size)
   
   local img = input:toTensor('float','RGB','DHW', true)
   
   -- insert image
   local resImg = res:narrow(1,1,3):zero()
   local padW = torch.round((self._input_size - oW)/2)
   local padH = torch.round((self._input_size - oH)/2)
   resImg:narrow(2,padH,oH):narrow(2,padW,oW):copy(img)
   
   -- 
   for 
   
   -- insert bbox mask
   local resMask = res:narrow(1,4,1):fill(1)
   resMask:narrow(2,padH,oH):narrow(2,padW,oW):zero()
   
   
   return input
end


function CocoBoxDetect:getImageBuffer(i)
   self._imgBuffers[i] = self._imgBuffers[i] or torch.FloatTensor()
   return self._imgBuffers[i]
end

-- Uniformly sample a class, then an instance of that class,
-- then the number of unknown instances (1->numInstance(img of instance))
-- This keeps the class distribution somewhat balanced.
-- "Somewhat" since the sampled instance will not necessarily be focused upon.
function CocoBoxDetect:sample(batch, nSample, sampleFunc)
   if (not batch) or (not sampleFunc) then 
      if torch.type(batch) == 'number' then
         sampleFunc = nSample
         nSample = batch
         batch = nil
      end
      batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}   
   end
   
   sampleFunc = sampleFunc or self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end
  
   nSample = nSample or 1
   local inputTable = {}
   local targetTable = {}   
   for i=1,nSample do
      -- sample class
      local class = torch.random(1, #self._classes)
      -- sample image from class
      local index = torch.random(1, self.classListSample[class]:nElement())
      local imgPath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, imgPath)
      table.insert(inputTable, dst)
      table.insert(targetTable, class)  
   end
   
   local inputView = batch and batch:inputs() or dp.ImageView()
   local targetView = batch and batch:targets() or dp.ClassView()
   local inputTensor = inputView:input() or torch.FloatTensor()
   local targetTensor = targetView:input() or torch.IntTensor()
   
   self:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   
   assert(inputTensor:size(2) == 3)
   inputView:forward('bchw', inputTensor)
   targetView:forward('b', targetTensor)
   targetView:setClasses(self._classes)
   batch:inputs(inputView)
   batch:targets(targetView)  
   
   collectgarbage()
   return batch
end

-- by default, just load the image and return it
function CocoBoxDetect:sampleDefault(dst, path)
   if not path then
      path = dst
      dst = torch.FloatTensor()
   end
   if not dst then
      dst = torch.FloatTensor()
   end
   -- if load_size[1] == 1, converts to greyscale (y in YUV)
   local input = self:loadImage(path)
   local out = input:toTensor('float','RGB','DHW', true)
   dst:resize(out:size(1), self._sample_size[3], self._sample_size[2])
   image.scale(dst, out)
   return dst
end

-- function to load the image, jitter it appropriately (random crops etc.)
function CocoBoxDetect:sampleTrain(dst, path)
   if not path then
      path = dst
      dst = torch.FloatTensor()
   end
   
   local input = self:loadImage(path)
   local iW, iH = input:size()
   -- do random crop
   local oW = self._sample_size[3]
   local oH = self._sample_size[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = input:crop(oW, oH, w1, h1)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then 
      out:flop()
   end
   out = out:toTensor('float','RGB','DHW', true)
   dst:resizeAs(out):copy(out)
   return dst
end

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
-- Works with the TopCrop feedback
function CocoBoxDetect:sampleTest(dst, path)
   if not path then
      path = dst
      dst = torch.FloatTensor()
   end
   
   local input = self:loadImage(path)
   iW, iH = input:size()
   
   local oH = self._sample_size[2]
   local oW = self._sample_size[3];
   dst:resize(10, 3, oW, oH)
   
   local im = input:toTensor('float','RGB','DHW', true)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   -- center
   image.crop(dst[1], im, w1, h1) 
   image.hflip(dst[2], dst[1])
   -- top-left
   h1 = 0; w1 = 0;
   image.crop(dst[3], im, w1, h1) 
   dst[4] = image.hflip(dst[3])
   -- top-right
   h1 = 0; w1 = iW-oW;
   image.crop(dst[5], im, w1, h1) 
   image.hflip(dst[6], dst[5])
   -- bottom-left
   h1 = iH-oH; w1 = 0;
   image.crop(dst[7], im, w1, h1) 
   image.hflip(dst[8], dst[7])
   -- bottom-right
   h1 = iH-oH; w1 = iW-oW;
   image.crop(dst[9], im, w1, h1) 
   image.hflip(dst[10], dst[9])
   return dst
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
         opt = options -- pass to all donkeys via upvalue
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
   local target = batch:targets():input()
   assert(batch:inputs():input() and batch:targets():input())
   
   self._send_batches:put(batch)
   
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         tbatch:inputs():forward('bchw', input)
         tbatch:targets():forward('b', target)
         
         dataset:sub(tbatch, start, stop)
         
         return input, target
      end,
      -- the endcallback (runs in the main thread)
      function(input, target)
         local batch = self._send_batches:get()
         batch:inputs():forward('bchw', input)
         batch:targets():forward('b', target)
         
         callback(batch)
         
         batch:targets():setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

function CocoBoxDetect:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async CocoBoxDetect (for now)'
   end  
   
   if not batch then
      batch = (not self._buffer_batches:empty()) and self._buffer_batches:get() or self:batch(nSample)
   end
   local input = batch:inputs():input()
   local target = batch:targets():input()
   assert(input and target)
   
   -- transfer the storage pointer over to a thread
   local inputPointer = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
   local targetPointer = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
   input:cdata().storage = nil
   target:cdata().storage = nil
   
   self._send_batches:put(batch)
   
   assert(self._threads:acceptsjob())
   self._threads:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         -- set the transfered storage
         torch.setFloatStorage(input, inputPointer)
         torch.setIntStorage(target, targetPointer)
         tbatch:inputs():forward('bchw', input)
         tbatch:targets():forward('b', target)
         
         dataset:sample(tbatch, nSample, sampleFunc)
         
         -- transfer it back to the main thread
         local istg = tonumber(ffi.cast('intptr_t', torch.pointer(input:storage())))
         local tstg = tonumber(ffi.cast('intptr_t', torch.pointer(target:storage())))
         input:cdata().storage = nil
         target:cdata().storage = nil
         return input, target, istg, tstg
      end,
      -- the endcallback (runs in the main thread)
      function(input, target, istg, tstg)
         local batch = self._send_batches:get()
         torch.setFloatStorage(input, istg)
         torch.setIntStorage(target, tstg)
         batch:inputs():forward('bchw', input)
         batch:targets():forward('b', target)
         
         callback(batch)
         
         batch:targets():setClasses(self._classes)
         self._recv_batches:put(batch)
      end
   )
end

-- recv results from worker : get results from queue
function CocoBoxDetect:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   if self._recv_batches:empty() then
      self._threads:dojob()
   end
   
   return self._recv_batches:get()
end



function CocoBoxDetect.test()
   local ds = dp.CocoBoxDetect{
      image_path='/media/nicholas14/Nick/coco/val2014',
      instance_path='/media/nicholas14/Nick/coco/annotations/instances_val2014.json',
   }
   return ds
end
