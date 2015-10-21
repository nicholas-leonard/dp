------------------------------------------------------------------------
--[[ CocoDetect ]]--
-- MS COCO box detection DataSource
-- Encapsulates CocoBoxDetect DataSets.
-- Note: install torch with Lua instead of LuaJIT to make this work...
------------------------------------------------------------------------
local CocoDetect, DataSource = torch.class("dp.CocoDetect", "dp.DataSource")
CocoDetect._name = 'CocoDetect'
       
function CocoDetect:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   if config.input_preprocess or config.output_preprocess then
      error("CocoDetect doesnt support Preprocesses. "..
            "Use Sampler ppf arg instead (for online preprocessing)")
   end
   local load_all
   self._args, self._data_path, self._input_size, 
      self._verbose, load_all, self._cache_mode
      = xlua.unpack(
      {config},
      'CocoDetect',
      'MS Coco Bounding Box Detection dataset',
      {arg='data_path', type='string', default=paths.concat(dp.DATA_DIR, 'coco'),
       help='path to dir containing the following directories :'..
       'train2014, val2014 and annotations. The latter contains the'..
       'instances_[train,val]2014.json files'},
      {arg='input_size', type='number', default=256 ,
       help='size (height=width) of the image. Padding will be added '..
       'around non-square images, which will be centered in the input'},
      {arg='verbose', type='boolean', default=true,
       help='Verbose mode during initialization'},
      {arg='load_all', type='boolean', default=true,
       help='Load all datasets : train and valid'},
      {arg='cache_mode', type='string', default='writeonce',
       help='writeonce : read from cache if exists, else write to cache. '..
       'overwrite : write to cache, regardless if exists. '..
       'nocache : dont read or write from cache. '..
       'readonly : only read from cache, fail otherwise.'}
   )
   
   self._image_size = {4, self._input_size, self._input_size}
   
   if load_all then
      self:loadTrain()
      self:loadValid()
   end
end

function CocoDetect:loadTrain()
   local dataset = dp.CocoBoxDetect{
      image_path=paths.concat(self._data_path, 'train2014'),
      instance_path=paths.concat(self._data_path, 'annotations/instances_train2014.json'), 
      input_size=self._input_size, which_set='train', 
      verbose=self._verbose, cache_mode=self._cache_mode,
      evaluate=false
   }
   self:trainSet(dataset)
   return dataset
end

function CocoDetect:loadValid()
   local dataset = dp.CocoBoxDetect{
      image_path=paths.concat(self._data_path, 'val2014'),
      instance_path=paths.concat(self._data_path, 'annotations/instances_val2014.json'), 
      input_size=self._input_size, which_set='valid', 
      verbose=self._verbose, cache_mode=self._cache_mode,
	  evaluate=true
   }
   self:validSet(dataset)
   return dataset
end

function CocoDetect:classes()
	local ds = self:trainSet() or self:validSet()
	return ds:classes()
end

-- Returns normalize preprocessing function (PPF)
-- Estimate the per-channel mean/std on training set and caches results
CocoDetect.normalizePPF = dp.ImageNet.normalizePPF

function CocoDetect:multithread(nThread)
   if self._train_set then
      self._train_set:multithread(nThread)
   end
   if self._valid_set then
      self._valid_set:multithread(nThread)
   end
end
