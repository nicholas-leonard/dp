------------------------------------------------------------------------
--[[ ImageClassSet ]]--
-- A DataSet for image classification in a flat folder structure :
-- [data_path]/[class]/[imagename].JPEG  (folder-name is class-name)
-- Optimized for extremely large datasets (14 million images+).
-- Tested only on Linux (as it uses command-line linux utilities to 
-- scale up to 14 million+ images)
-- Images on disk can have different height, width and number of channels.
------------------------------------------------------------------------
local ImageClassSet, parent = torch.class("dp.ImageClassSet", "dp.DataSet")

ImageClassSet._input_shape = 'bchw'
ImageClassSet._output_shape = 'b'

function ImageClassSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data_path, load_size, sample_size, sample_func, which_set,  
      verbose, sort_func, cache_mode, cache_path = xlua.unpack(
      {config},
      'ImageClassSet', 
      'A DataSet for images in a flat folder structure',
      {arg='data_path', type='table | string', req=true,
       help='one or many paths of directories with images'},
      {arg='load_size', type='table', req=true,
       help='a size to load the images to, initially'},
      {arg='sample_size', type='table',
       help='a consistent sample size to resize the images. '..
       'Defaults to load_size'},
      {arg='sample_func', type='string | function', default='sampleDefault',
       help='function f(self, dst, path) used to create a sample(s) from '..
       'an image path. Stores them in dst. Strings "sampleDefault", '..
       '"sampleTrain" or "sampleTest" can also be provided as they '..
       'refer to existing functions'},
      {arg='which_set', type='string', default='train',
       help='"train", "valid" or "test" set'},
      {arg='verbose', type='boolean', default=true,
       help='display verbose messages'},
      {arg='sort_func', type='function', 
       help='comparison operator used for sorting class dir to get idx.'
       ..' Defaults to < operator'},
      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'},
      {arg='cache_path', type='string', 
       help='Path to cache. Defaults to [data_path[1]]/cache.th7'}
   )
   -- globals
   gm = require 'graphicsmagick'
   
   -- locals
   self:whichSet(which_set)
   self._load_size = load_size
   assert(self._load_size[1] == 3, "ImageClassSet doesn't yet support greyscaling : load_size")
   self._sample_size = sample_size or self._load_size
   assert(self._sample_size[1] == 3, "ImageClassSet doesn't yet support greyscaling : sample_size")
   self._verbose = verbose   
   self._data_path = type(data_path) == 'string' and {data_path} or data_path
   self._sample_func = sample_func
   self._sort_func = sort_func
   self._cache_mode = cache_mode
   self._cache_path = cache_path or paths.concat(self._data_path[1], 'cache.th7')
   
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
   
   -- buffers
   self._imgBuffers = {}
   
   -- required for multi-threading
   self._config = config 
end

function ImageClassSet:saveIndex()
   local index = {}
   for i,k in ipairs{'_classes','_classIndices','imagePath','imageClass','classList','classListSample'} do
      index[k] = self[k]
   end
   torch.save(self._cache_path, index)
end

function ImageClassSet:loadIndex()
   local index = torch.load(self._cache_path)
   for k,v in pairs(index) do
      self[k] = v
   end
   self._n_sample = self.imagePath:size(1)
end

function ImageClassSet:buildIndex()
   -- loop over each paths folder, get list of unique class names, 
   -- also store the directory paths per class
   local classes = {}
   local classList = {}
   for k,path in ipairs(self._data_path) do
      for class in lfs.dir(path) do
         local dirpath = paths.concat(path, class)
         if class:sub(1,1) ~= '.' and paths.dirp(dirpath) and not classes[class] then
            table.insert(classList, class)
            classes[class] = true
         end
      end
   end
   
   -- sort classes for indexing consistency
   _.sort(classList, self._sort_func)
   
   local classPaths = {}
   for i, class in ipairs(classList) do
      classes[class] = i
      classPaths[i] = {}
   end
   
   for k,path in ipairs(self._data_path) do
      for class in lfs.dir(path) do
         local dirpath = paths.concat(path, class)
         if class:sub(1,1) ~= '.' and paths.dirp(dirpath) then
            local idx = classes[class]
            table.insert(classPaths[idx], dirpath)
         end
      end
   end
   
   self._classes = classList
   
   if self._verbose then
      print("found " .. #self._classes .. " classes")
   end
   
   self._classIndices = classes
   
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   
   ---------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self._classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   
   if self._verbose then
      print('running "find" on each class directory, and concatenate all' 
         .. ' those filenames into a single file containing all image paths for a given class')
   end
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self._classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();
   
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self._classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions 
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   if self._verbose then
      print('now combine all the files to a single large file')
   end
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self._classes
   for i=1,#self._classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   ---------------------------------------------------------------------
   if self._verbose then
      print('loading concatenated list of sample paths to self.imagePath')
   end
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '" 
                                                  .. combinedFindList .. "' |" 
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '" 
                                           .. combinedFindList .. "' |" 
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self._verbose and count % 10000 == 0 then 
         xlua.progress(count, length) 
      end
      count = count + 1
   end
   if self._verbose then 
      xlua.progress(length, length) 
   end
      
   self._n_sample = self.imagePath:size(1)
   ---------------------------------------------------------------------
   if self._verbose then
      print('Updating classList and imageClass appropriately')
   end
   self.imageClass:resize(self._n_sample)
   local runningIndex = 0
   for i=1,#self._classes do
      if self.verbose then xlua.progress(i, #(self._classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '" 
                                              .. classFindFiles[i] .. "' |" 
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   ----------------------------------------------------------------------
   -- clean up temporary files
   if self._verbose then
      print('Cleaning up temporary files')
   end
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
end

function ImageClassSet:batch(batch_size)
   return dp.Batch{
      which_set=self._which_set,
      inputs=dp.ImageView('bchw', torch.FloatTensor(batch_size, unpack(self._sample_size))),
      targets=dp.ClassView('b', torch.IntTensor(batch_size))
   }
end

-- nSample(), nSample(class)
function ImageClassSet:nSample(class, list)
   list = list or self.classList
   if not class then
      return self._n_sample
   elseif type(class) == 'string' then
      return list[self._classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

function ImageClassSet:sub(batch, start, stop)
   if not stop then
      stop = start
      start = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}
   
   local sampleFunc = self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end
   
   local inputTable = {}
   local targetTable = {}
   local i = 1
   for idx=start,stop do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[idx]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, imgpath)
      table.insert(inputTable, dst)
      table.insert(targetTable, self.imageClass[idx])     
      i = i + 1
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

   return batch
end

function ImageClassSet:index(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   batch = batch or dp.Batch{which_set=self:whichSet(), epoch_size=self:nSample()}

   local sampleFunc = self._sample_func
   if torch.type(sampleFunc) == 'string' then
      sampleFunc = self[sampleFunc]
   end

   local inputTable = {}
   local targetTable = {}
   for i = 1, indices:size(1) do
      idx = indices[i]
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[idx]))
      local dst = self:getImageBuffer(i)
      dst = sampleFunc(self, dst, imgpath)
      table.insert(inputTable, dst)
      table.insert(targetTable, self.imageClass[idx])
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
   return batch
end

-- converts a table of samples (and corresponding labels) to tensors
function ImageClassSet:tableToTensor(inputTable, targetTable, inputTensor, targetTensor)
   inputTensor = inputTensor or torch.FloatTensor()
   targetTensor = targetTensor or torch.IntTensor()
   local n = #targetTable

   local samplesPerDraw = inputTable[1]:dim() == 3 and 1 or inputTable[1]:size(1)
   inputTensor:resize(n, samplesPerDraw, unpack(self._sample_size))
   targetTensor:resize(n, samplesPerDraw)
   
   for i=1,n do
      inputTensor[i]:copy(inputTable[i])
      targetTensor[i]:fill(targetTable[i])
   end
   
   inputTensor:resize(n*samplesPerDraw, unpack(self._sample_size))
   targetTensor:resize(n*samplesPerDraw)
   
   return inputTensor, targetTensor
end

function ImageClassSet:loadImage(path)
   -- https://github.com/clementfarabet/graphicsmagick#gmimage
   local lW, lH = self._load_size[3], self._load_size[2]
   -- load image with size hints
   local input = gm.Image():load(path, self._load_size[3], self._load_size[2])
   -- resize by imposing the smallest dimension (while keeping aspect ratio)
   local iW, iH = input:size()
   if iW/iH < lW/lH then
      input:size(nil, lW)
   else
      input:size(nil, lH)
   end
   return input
end

function ImageClassSet:getImageBuffer(i)
   self._imgBuffers[i] = self._imgBuffers[i] or torch.FloatTensor()
   return self._imgBuffers[i]
end

-- Sample a class uniformly, and then uniformly samples example from class.
-- This keeps the class distribution balanced.
-- sampleFunc is a function that generates one or many samples
-- from one image. e.g. sampleDefault, sampleTrain, sampleTest.
function ImageClassSet:sample(batch, nSample, sampleFunc)
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
function ImageClassSet:sampleDefault(dst, path)
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
function ImageClassSet:sampleTrain(dst, path)
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
function ImageClassSet:sampleTest(dst, path)
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

function ImageClassSet:classes()
   return self._classes
end
------------------------ multithreading --------------------------------

function ImageClassSet:multithread(nThread)
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
         
         dataset = dp.ImageClassSet(config)
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

function ImageClassSet:synchronize()
   self._threads:synchronize()
   while not self._recv_batches:empty() do
     self._buffer_batches:put(self._recv_batches:get())
   end
end

-- send request to worker : put request into queue
function ImageClassSet:subAsyncPut(batch, start, stop, callback)   
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

function ImageClassSet:sampleAsyncPut(batch, nSample, sampleFunc, callback)
   self._iter_mode = self._iter_mode or 'sample'
   if (self._iter_mode ~= 'sample') then
      error'can only use one Sampler per async ImageClassSet (for now)'
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
function ImageClassSet:asyncGet()
   -- necessary because Threads:addjob sometimes calls dojob...
   if self._recv_batches:empty() then
      self._threads:dojob()
   end
   
   return self._recv_batches:get()
end
