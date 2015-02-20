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

function ImageClassSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data_path, load_size, sample_size, sample_func, which_set,  
      carry, verbose = xlua.unpack(
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
      {arg='carry', type='dp.Carry',
       help='An object store that is carried (passed) around the '..
       'network during a propagation.'},
      {arg='verbose', type='boolean', default=true,
       help='display verbose messages'}
   )
   -- globals :
   ffi = require 'ffi'
   gm = require 'graphicsmagick'

   self:setWhichSet(which_set)
   self._load_size = load_size
   assert(self._load_size[1] == 3, "ImageClassSet doesn't yet support greyscaling : load_size")
   self._sample_size = sample_size or self._load_size
   assert(self._sample_size[1] == 3, "ImageClassSet doesn't yet support greyscaling : sample_size")
   self._carry = carry or dp.Carry()
   self._verbose = verbose   
   self._data_path = type(data_path) == 'string' and {data_path} or data_path
   self._sample_func = sample_func
   
   -- find class names
   self._classes = {}
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
   
   local classPaths = {}
   -- sort classes for indexing consistency
   for i, class in ipairs(_.sort(classList)) do
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
   
   -- buffers
   self._imgBuffers = {}
end

function ImageClassSet:batch(batch_size)
   return self:sub(1,batch_size)
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
   batch:setInputs(inputView)
   batch:setTargets(targetView)  
   batch:carry():putObj('nSample', targetTensor:size(1))

   return batch
end

function ImageClassSet:index(batch, indices)
   error"notImplemented"
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
      local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
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
   batch:setInputs(inputView)
   batch:setTargets(targetView)  
   batch:carry():putObj('nSample', targetTensor:size(1))
   
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
   iW, iH = input:size()
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

-- Create a test data loader (testLoader),
-- which can iterate over the test set and returns an image's
-- 10 crops (center + 4 corners) and their hflips]]--
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
   h1 = 1; w1 = 1;
   image.crop(dst[3], im, w1, h1) 
   dst[4] = image.hflip(dst[3])
   -- top-right
   h1 = 1; w1 = iW-oW;
   image.crop(dst[5], im, w1, h1) 
   image.hflip(dst[6], dst[5])
   -- bottom-left
   h1 = iH-oH; w1 = 1;
   image.crop(dst[7], im, w1, h1) 
   image.hflip(dst[8], dst[7])
   -- bottom-right
   h1 = iH-oH; w1 = iW-oW;
   image.crop(dst[9], im, w1, h1) 
   image.hflip(dst[10], dst[9])
   return dst
end
