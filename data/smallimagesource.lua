------------------------------------------------------------------------
--[[ SmallImageSource ]]--
-- Generic DataSource for loading small image classification datasets. 
-- Images are expected to be stored on disk as :
-- data_path/name/[classname]/[imagefilename]
-- We recommend using it for small datasets of small images.
-- For large datasets, use ImageSource instead.
------------------------------------------------------------------------
local SmallImageSource, DataSource = torch.class("dp.SmallImageSource", "dp.DataSource")

function SmallImageSource:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all
   self._args, self._name, self._image_size, self._valid_ratio, 
         self._classes, 
         self._train_dir, self._test_dir, self._data_path, 
         self._cache_mode, self._cache_path,
         self._scale, self._binarize, self._download_url, load_all
      = xlua.unpack(
      {config},
      'SmallImageSource', 
      'Generic DataSource for loading small image classification datasets. ',
      {arg='name', type='string', req=true, 
       help='name of this dataset. Also name of directory at data_path'},
      {arg='image_size', type='table', req=true,
       help='size (c,h,w) of images stored in tensor e.g. {3,28,32}'},
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='classes', type='table', 
       help='name of each class directory on disk'},
      {arg='train_dir', type='string', default='train', 
       help='name of train directory (includes validation set)'},
      {arg='test_dir', type='string', default='test', 
       help='name of test directory'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'},
      {arg='cache_path', type='string', 
       help='path to cache directory (defaults to data_path).'},
      {arg='scale', type='table', 
       help='bounds to scale the values between', default={0,1}},
      {arg='binarize', type='boolean', 
       help='binarize the inputs (0s and 1s)', default=false},
      {arg='download_url', type='string',
       help='URL from which to download dataset if not found on disk.'},
      {arg='load_all', type='boolean', 
       help='Load all datasets : train, valid, test.', default=true}
   )
   
   self._cache_path = self._cache_path or self._data_path
   
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   
   assert(#self._image_size == 3, "Expecting {nColor, height, width}")
   self._image_axes = 'bchw'
   
   if load_all then
      self:loadTrainValid()
      self:loadTest()
   end
   
   DataSource.__init(self, {
      train_set=self:trainSet(), valid_set=self:validSet(),
      test_set=self:testSet(),
      input_preprocess=config.input_preprocess,
      target_preprocess=config.target_preprocess
   })
end

function SmallImageSource:loadTrainValid()
   local inputs, targets, classes = self:loadData(self._train_dir)
   self._classes = classes
   
   -- train
   local start = 1
   local size = math.floor(inputs:size(1)*(1-self._valid_ratio))

   self:trainSet(
      self:createDataSet(
         inputs:narrow(1, start, size), 
         targets:narrow(1, start, size), 
         'train'
      )
   )
   
   -- valid
   start = size + 1
   size = inputs:size(1) - start
   self:validSet(
      self:createDataSet(
         inputs:narrow(1, start, size), 
         targets:narrow(1, start, size), 
         'valid'
      )
   )
   return self:trainSet(), self:validSet()
end

function SmallImageSource:loadTest()
   if self._test_dir == '' then
      return
   end
   local inputs, targets, classes = self:loadData(self._test_dir)
   self._classes = classes
   self:testSet(self:createDataSet(inputs, targets, 'test'))
   return self:testSet()
end

--Creates an SmallImageSource Dataset out of data and which_set
function SmallImageSource:createDataSet(inputs, targets, which_set)
   if self._binarize then
      DataSource.binarize(inputs, 128)
   end
   
   if self._scale and not self._binarize then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end   
   
   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward('bchw', inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'b')
   
   return ds
end

function SmallImageSource:loadData(set_dir, download_url)
   -- use cache?
   local cacheFile = self._name..'_'..set_dir
   cacheFile = cacheFile .. table.concat(self._image_size,'x')
   cacheFile = cacheFile ..'_cache.t7'
   
   local cachePath = paths.concat(self._cache_path, cacheFile)
   if paths.filep(cachePath) then
      if not _.contains({'nocache','overwrite'}, self._cache_mode)  then
         return table.unpack(torch.load(cachePath))
      end
   elseif self._cache_mode == 'readonly' then
      error("SmallImageSource: No cache at "..cachePath)
   end
   
   local data_path = DataSource.getDataPath{
      name=self._name, url=download_url or self._download_url,
      decompress_file=set_dir, data_dir=self._data_path
   }
   
   if (not self._classes) or _.isEmpty(self._classes) then
      -- extrapolate classes from directories
      self._classes = {}
      for class in paths.iterdirs(data_path) do
         table.insert(self._classes, class)
      end
      _.sort(self._classes) -- make indexing consistent
   end
   
   -- count images
   local n_example = 0
   local classfiles= {}
   for classidx, class in ipairs(self._classes) do
      local classpath = paths.concat(data_path, class)
      local files = paths.indexdir(classpath)
      assert(files:size() > 0, "class dir is empty : "..classpath)
      n_example = n_example + files:size()
      table.insert(classfiles, files)
   end
   assert(n_example > 0, "no examples found for at data_path :"..data_path)
   
   -- allocate tensors
   local inputs = torch.FloatTensor(n_example, unpack(self._image_size)):zero()
   local targets = torch.Tensor(n_example):fill(1)
   local shuffle = torch.randperm(n_example) -- useless for test set
   
   -- load images
   local example_idx = 1
   local buffer
   for classidx, class in ipairs(self._classes) do
      local files = classfiles[classidx]
      
      for i=1,files:size() do
         local success, img = pcall(function()
            return image.load(files:filename(i))
         end)
      
         if success then
            assert(img:size(1) == self._image_size[1], "Inconsistent number of channels/colors")
            
            if img:size(2) ~= self._image_size[2] or img:size(3) ~= self._image_size[3] then
               -- rescale the image
               buffer = buffer or img.new()
               buffer:resize(table.unpack(self._image_size))
               image.scale(buffer, img)
               img = buffer
            end
            
            local ds_idx = shuffle[example_idx]
            inputs[ds_idx]:copy(img)
            targets[ds_idx] = classidx
         end
         
         example_idx = example_idx + 1
         collectgarbage()
      end
   end
   
   if self._cache_mode ~= 'nochache' then
      torch.save(cachePath, {inputs, targets, self._classes})
   end
  
   return inputs, targets, self._classes
end

