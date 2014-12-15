
local ffi = require 'ffi'
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
local gm = require 'graphicsmagick'
------------------------------------------------------------------------
--[[ ImageNet ]]--
-- http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php
-- Wraps the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)
-- classification dataset (commonly known as ImageNet). The dataset
-- hasn't changed from 2012-2014.

-- A dataset class for images in a flat folder structure :
-- [pathtodata]/[class]/[imagename].JPEG  (folder-name is class-name)
-- Optimized for extremely large datasets (14 million images+).
-- Tested only on Linux (as it uses command-line linux utilities to 
-- scale up to 14 million+ images)
------------------------------------------------------------------------
local ImageNet, DataSource = torch.class("dp.ImageNet", "dp.DataSource")

ImageNet._name = 'ImageNet'
ImageNet._image_axes = 'bhwc'
ImageNet._structured_url = 'http://www.image-net.org/api/xml/structure_released.xml'

function ImageNet:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess
   self._args, self._load_size, self._sample_size, self._sampling_mode, 
      self._data_path, self._train_dir, self._valid_dir, self._test_dir, 
      self._verbose, self._sample_hook_train, self._sample_hook_test,
      self._download_url, load_all, input_preprocess, 
      target_preprocess
      = xlua.unpack(
      {config},
      'ImageNet',
      'ILSVRC2012-14 image classification dataset',
      {arg='load_size', type='table',
       help='a size to load the images to, initially'},
      {arg='sample_size', type='table',
       help='a consistent sample size to resize the images'},
      {arg='sampling_mode',type='string', default = 'balanced',
       help='Sampling mode: random | balanced '},
      {arg='data_path', type='table | string', default=dp.DATA_DIR,
       help='one or many paths of directories with images'},
      {arg='train_dir', type='string', default='ILSVRC2012_img_train',
       help='name of train_dir'},
      {arg='valid_dir', type='string', default='ILSVRC2012_img_val',
       help='name of valid_dir'},
      {arg='test_dir', type='string', default='ILSVRC2012_img_test',
       help='name of test_dir'},
      {arg='verbose', type='boolean', default = false,
       help='Verbose mode during initialization'},
      {arg='sample_hook_train', type='function',
       help='applied to sample during training(ex: for lighting jitter). '
       .. 'It takes the image path as input'},
      {arg='sample_hook_test', type='function', 
       help='applied to sample during testing'},
      {arg='download_url', type='string',
       default='http://yaroslavvb.com/upload/notMNIST/',
       help='URL from which to download dataset if not found on disk.'},
      {arg='load_all', type='boolean', 
       help='Load all datasets : train, valid, test.', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'}  
   }

   self._load_size = self.load_size or self._sample_size
   self._data_path = torch.type(self._data_path) == 'string' and {self._data_path) or self._data_path
   
   -- find class names
   self.classes = {}
   -- loop over each paths folder, get list of unique class names, 
   -- also store the directory paths per class
   -- for each class, 
   local classPaths = {}
   for k,path in ipairs(self.paths) do
      local dirs = dir.getdirectories(path);
      for k,dirpath in ipairs(dirs) do
         local class = paths.basename(dirpath)
         local idx = tablex.find(self.classes, class)
         if not idx then
            table.insert(self.classes, class)
            idx = #self.classes
            classPaths[idx] = {}
         end
         if not tablex.find(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath);
         end
      end
   end
   
   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end
   
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   
   print('running "find" on each class directory, and concatenate all' 
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();
   
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
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
   
   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
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
      if self.verbose and count % 10000 == 0 then 
         xlua.progress(count, length) 
      end; 
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
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

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
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
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of ' 
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         local count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

function ImageNet:loadStructure()
   local path = DataSource.getDataPath{
      name=self._name, url=self._structured_url, 
      decompress_file='structure_released.xml', 
      data_dir=self._data_path
   }
   -- sudo luarocks install xml
   local xml = require 'xml'
   local structure = xml.loadpath(path)
   return structure
end

-- size(), size(class)
function ImageNet:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- size(), size(class)
function ImageNet:sizeTrain(class)
   if self.split == 0 then
      return 0;
   end
   if class then
      return self:size(class, self.classListTrain)
   else
      return self.numSamples - self.testIndicesSize
   end
end

-- size(), size(class)
function ImageNet:sizeTest(class)
   if self.split == 100 then
      return 0
   end
   if class then
      return self:size(class, self.classListTest)
   else
      return self.testIndicesSize
   end
end

-- by default, just load the image and return it
function ImageNet:defaultSampleHook(imgpath)
   local out = gm.Image()
   out:load(imgpath, self.loadSize[3], self.loadSize[2])
   :size(self.sampleSize[3], self.sampleSize[2])
   out = out:toTensor('float','RGB','DHW')
   return out
end

-- getByClass
function ImageNet:getByClass(class)
   local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
   local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
   return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   local samplesPerDraw
   if dataTable[1]:dim() == 3 then samplesPerDraw = 1
   else samplesPerDraw = dataTable[1]:size(1) end
   if quantity == 1 and samplesPerDraw == 1 then
      data = dataTable[1]
      scalarLabels = scalarTable[1]
      labels = torch.LongTensor(#(self.classes)):fill(-1)
      labels[scalarLabels] = 1
   else
      data = torch.Tensor(quantity * samplesPerDraw, 
                          self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
      scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
      labels = torch.LongTensor(quantity * samplesPerDraw, #(self.classes)):fill(-1)
      for i=1,#dataTable do
         data[{{i, i+samplesPerDraw-1}}]:copy(dataTable[i])
         scalarLabels[{{i, i+samplesPerDraw-1}}]:fill(scalarTable[i])
	       labels[{{i, i+samplesPerDraw-1},{scalarTable[i]}}]:fill(1)
      end
   end   
   return data, scalarLabels, labels
end

-- sampler, samples from the training set.
function ImageNet:sample(quantity)
   if self.split == 0 then 
      error('No training mode when split is set to 0') 
   end
   quantity = quantity or 1
   local dataTable = {}
   local scalarTable = {}   
   for i=1,quantity do
      local class = torch.random(1, #self.classes)
      local out = self:getByClass(class)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)      
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels      
end

function ImageNet:get(i1, i2)
   local indices, quantity
   if type(i1) == 'number' then
      if type(i2) == 'number' then -- range of indices
         indices = torch.range(i1, i2); 
         quantity = i2 - i1 + 1;
      else -- single index 
         indices = {i1}; quantity = 1 
      end 
   elseif type(i1) == 'table' then
      indices = i1; quantity = #i1;         -- table
   elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
      indices = i1; quantity = (#i1)[1];    -- tensor
   else
      error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))      
   end
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
      out = self:sampleHookTest(imgpath)
      table.insert(dataTable, out)
      table.insert(scalarTable, self.imageClass[indices[i]])      
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels
end

function ImageNet:test(quantity)
   if self.split == 100 then
      error('No test mode when you are not splitting the data')
   end
   local i = 1
   local n = self.testIndicesSize
   local qty = quantity or 1
   return function ()
      if i+qty-1 <= n then 
         local data, scalarLabelss, labels = self:get(i, i+qty-1)
         i = i + qty
         return data, scalarLabelss, labels
      end
   end
end
