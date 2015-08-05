------------------------------------------------------------------------
--[[ DataSource ]]--
-- Abstract class.
-- Used to generate up to 3 DataSets : train, valid and test.
-- Can also perform preprocessing using Preprocess on all DataSets by
-- fitting only the training set.
------------------------------------------------------------------------
local DataSource = torch.class("dp.DataSource")
DataSource.isDataSource = true

function DataSource:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, train_set, valid_set, test_set, 
         input_preprocess, target_preprocess
      = xlua.unpack(
      {config},
      'DataSource', 
      'Abstract Class ' ..
      'Used to generate up to 3 DataSets : train, valid and test. ' ..
      'Preprocessing can be performed on all ' .. 
      'DataSets by fitting the preprocess (e.g. Standardization) on ' ..
      'only the training set, and reusing the same statistics on ' ..
      'the validation and test sets',
      {arg='train_set', type='dp.DataSet', --req=true,
       help='used for minimizing a Loss by optimizing a Model'},
      {arg='valid_set', type='dp.DataSet',
       help='used for cross-validation and for e.g. early-stopping.'},
      {arg='test_set', type='dp.Dataset',
       help='used to evaluate generalization performance ' ..
      'after training (e.g. to compare different models).'},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'}  
   )
   --datasets
   self:trainSet(train_set)
   self:validSet(valid_set)
   self:testSet(test_set)
   --preprocessing
   self:inputPreprocess(input_preprocess)
   self:targetPreprocess(target_preprocess)
   self:preprocess()
end

function DataSource:getView(which_set, attribute)
   which_set = which_set or 'train'
   attribute = attribute or 'input'
   
   local dataset
   if which_set == 'train' then
      dataset = self:trainSet()
   elseif which_set == 'valid' then
      dataset = self:validSet()
   elseif which_set == 'test' then
      dataset = self:testSet()
   else
      error("expecting 'train', 'valid' or 'test' at arg 1: "..which_set)
   end
   
   local dataview
   if attribute == 'input' or attribute == 'inputs' then
      dataview = dataset:inputs()
   elseif attribute == 'target' or attribute == 'targets' then 
      dataview = dataset:targets()
   else
      error("expecting 'input' or 'target' at arg 2: "..attribute)
   end
   
   return dataview, dataset
end

-- a function to simplify calling :
-- ds:[train,valid,test]Set():[inputs,targets]():forward(view, tensor_type).
-- all attributes are optional.
-- example usage : get('train', 'input', 'bchw', 'float')
function DataSource:get(which_set, attribute, view, type)
   view = view or 'default'
   local dataview, dataset = self:getView(which_set, attribute)
   local tensor_type
   if torch.type(type) == 'string' then
      tensor_type = 'torch.%sTensor'
      if type:sub(-6,-1) == 'Tensor' then
         tensor_type = type
      elseif type == 'float' or type == 'Float' then
         tensor_type = string.format(tensor_type, 'Float')
      elseif type == 'double' or type == 'Double' then
         tensor_type = string.format(tensor_type, 'Double')
      elseif type == 'cuda' or type == 'Cuda' then
         tensor_type = string.format(tensor_type, 'Cuda')
      elseif type == 'int' or type == 'Int' then
         tensor_type = string.format(tensor_type, 'Int')
      elseif type == 'long' or type == 'Long' then
         tensor_type = string.format(tensor_type, 'Long')
      elseif type == 'char' or type == 'Char' then
         tensor_type = string.format(tensor_type, 'Char')
      elseif type == 'byte' or type == 'Byte' then
         tensor_type = string.format(tensor_type, 'Byte')
      elseif type == 'short' or type == 'Short' then
         tensor_type = string.format(tensor_type, 'Short')
      else
         error("expecting tensor type at arg 4 : "..type)
      end
   end
   local tensor = dataview:forward(view, tensor_type)
   return tensor, dataview, dataset
end

-- And we cannot have a get without a set:
function DataSource:set(which_set, attribute, view, tensor)
   assert(view, "expecting view at arg 3")
   local dataview, dataset = self:getView(which_set, attribute)
   dataview:forward(view, tensor)
   return dataview, dataset
end

function DataSource:trainSet(train_set)
   if train_set then
      self._train_set = train_set
   end
   return self._train_set
end

function DataSource:validSet(valid_set)
   if valid_set then
      self._valid_set = valid_set
   end
   return self._valid_set
end

function DataSource:testSet(test_set)
   if test_set then
      self._test_set = test_set
   end
   return self._test_set
end

function DataSource:inputPreprocess(input_preprocess)
   if input_preprocess then
      if torch.type(input_preprocess)  == 'table' then
         input_preprocess = dp.Pipeline(input_preprocess)
      end
      self._input_preprocess = input_preprocess
   end
   return self._input_preprocess
end

function DataSource:targetPreprocess(target_preprocess)
   if target_preprocess then
      if torch.type(target_preprocess) == 'table' then
         target_preprocess = dp.Pipeline(target_preprocess)
      end
      self._target_preprocess = target_preprocess
   end
   return self._target_preprocess
end

--preprocess datasets:
function DataSource:preprocess()
   if not (self:inputPreprocess() or self:targetPreprocess()) then
      return
   end
   train_set = self:trainSet()
   if train_set then
      train_set:preprocess{
         input_preprocess=self:inputPreprocess(), 
         target_preprocess=self:targetPreprocess(),
         can_fit=true}
   end
   
   valid_set = self:validSet()
   if valid_set then
      valid_set:preprocess{
         input_preprocess=self:inputPreprocess(), 
         target_preprocess=self:targetPreprocess(),
         can_fit=false}
   end
   
   test_set = self:testSet()
   if test_set then
      test_set:preprocess{
         input_preprocess=self:inputPreprocess(), 
         target_preprocess=self:targetPreprocess(),
         can_fit=false}
   end
end

-- The following methods access optional static attributes (defined for class)
function DataSource:name()
   return self._name
end

function DataSource:classes(classes)
   if classes then
      self._classes = classes
   end
   return self._classes
end

-- input size
function DataSource:iSize(idx)
   if torch.type(idx) == 'string' then
      local view = string.gsub(self:iAxes(), 'b', '')
      local axis_pos = view:find(idx)
      if not axis_pos then
         if idx == 'f' then
            if self._feature_size then 
               -- legacy
               return self._feature_size
            else
               -- extrapolate feature size
               local set = self:trainSet() or self:validSet() or self:testSet()
               local batch = set:sub(1,2)
               local inputView = batch:inputs()
               local inputs = inputView:forward('bf')
               return inputs:size(2)
            end
         else
            error("Datasource has no axis '"..idx.."'")
         end
      end
      idx = axis_pos
   end
   
   if self._image_size then
      -- legacy 
      return idx and self._image_size[idx] or self._image_size
   else
      -- extrapolate input size
      local set = self:trainSet() or self:validSet() or self:testSet()
      local batch = set:sub(1,2)
      local inputView = batch:inputs()
      assert(torch.isTypeOf(inputView, 'dp.ImageView'), "Expecting dp.ImageView inputs")
      local inputs = inputView:forward(self:imageAxes())
      local size = inputs:size():totable()
      local b_idx = inputView:findAxis('b')
      table.remove(size, b_idx)
      return idx and size[idx] or size
   end
end

function DataSource:iAxes(idx)
   if self._image_axes then -- legacy
      return idx and self._image_axes[idx] or self._image_axes
   else
      local iShape = self:ioShapes()
      return idx and iShape[idx] or iShape
   end
end

function DataSource:ioShapes(input_shape, output_shape)
   if input_shape or output_shape then
      if self:trainSet() then
         self:trainSet():ioShapes(input_shape, output_shape)
      end
      if self:validSet() then
         self:validSet():ioShapes(input_shape, output_shape)
      end
      if self:testSet() then
         self:testSet():ioShapes(input_shape, output_shape)
      end
      return
   end
   local set = self:trainSet() or self:validSet() or self:testSet()
   return set:ioShapes()
end

-- DEPRECATED
function DataSource:imageSize(idx)
   return self:iSize(idx)
end

function DataSource:featureSize()
   return self:iSize('f')
end

function DataSource:imageAxes(idx)
   return self:iAxes(idx)
end
-- END DEPRECATED

-- Download datasource if not found locally.  
-- Returns the path to the resulting data file.
function DataSource.getDataPath(config)
   assert(type(config) == 'table', "getDataPath requires key-value arguments")
   local args, name, url, data_dir, decompress_file
      = xlua.unpack(
         {config},
         'getDataPath', 
         'Check locally and download datasource if not found. ' ..
         'Returns the path to the resulting data file. ' ..
         'Decompress if data_dir/name/decompress_file is not found',
         {arg='name', type='string', default='', 
          help='name of the DataSource (e.g. "mnist", "svhn", etc). ' ..
          'A directory with this name is created within ' ..
          'data_dir to contain the downloaded files. Or is ' ..
          'expected to find the data files in this directory.'},
         {arg='url', type='string', req=true,
          help='URL from which data can be downloaded in case '..
          'it is not found in the path.'},
         {arg='data_dir', type='string', default=dp.DATA_DIR,
          help='path to directory where directory name is expected ' ..
          'to contain the data, or where they will be downloaded.'},
         {arg='decompress_file', type='string', 
          help='When non-nil, decompresses the downloaded data if ' ..
          'data_dir/name/decompress_file is not found. In which ' ..
          'case, returns data_dir/name/decompress_file.'}
   )
   local datasrc_dir = paths.concat(data_dir, name)
   local data_file = paths.basename(url)
   local data_path = paths.concat(datasrc_dir, data_file)

   dp.mkdir(data_dir)
   dp.mkdir(datasrc_dir)
   dp.check_and_download_file(data_path, url)
   
   -- decompress 
   if decompress_file then
      local decompress_path = paths.concat(datasrc_dir, decompress_file)

      if not dp.is_file(decompress_path) then
         dp.do_with_cwd(datasrc_dir,
            function()
               print("decompressing file: ", data_path)
               dp.decompress_file(data_path)
            end)
      end
      return decompress_path
   end
   
   return data_path
end

function DataSource.rescale(data, min, max, dmin, dmax)
   local range = max - min
   local dmin = dmin or data:min()
   local dmax = dmax or data:max()
   local drange = dmax - dmin

   data:add(-dmin)
   data:mul(range)
   data:mul(1/drange)
   data:add(min)
end

function DataSource.binarize(x, threshold)
   x[x:lt(threshold)] = 0;
   x[x:ge(threshold)] = 1;
   return x
end

-- BEGIN DEPRECATED (June 13, 2015)
function DataSource:setTrainSet(train_set)
   self._train_set = train_set
end

function DataSource:setValidSet(valid_set)
   self._valid_set = valid_set
end

function DataSource:setTestSet(test_set)
   self._test_set = test_set
end

function DataSource:setInputPreprocess(input_preprocess)
   if torch.type(input_preprocess)  == 'table' then
      input_preprocess = dp.Pipeline(input_preprocess)
   end
   self._input_preprocess = input_preprocess
end

function DataSource:setTargetPreprocess(target_preprocess)
   if torch.type(target_preprocess) == 'table' then
      target_preprocess = dp.Pipeline(target_preprocess)
   end
   self._target_preprocess = target_preprocess
end
-- END DEPRECATED
