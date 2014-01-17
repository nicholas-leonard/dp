require 'torch'
require 'paths'


--[[ TODO ]]--
-- Only the training set requires sampling?
-- How to preprocess targets (fit only train?)
-- Each preprocess has a unique id, which we can use to cache

------------------------------------------------------------------------
--[[ DataSource ]]--
-- Abstract class.
-- Used to generate up to 3 DataSets : train, valid and test.
-- Can also perform preprocessing using Preprocess on all DataSets by
-- fitting only the training set.
------------------------------------------------------------------------
local DataSource = torch.class("dp.DataSource")

function DataSource:__init(...)
   local args, train_set, valid_set, test_set, 
         input_preprocess, output_preprocess
      = xlua.unpack(
      {... or {}},
      'DataSource', 
      'Abstract Class ' ..
      'Used to generate up to 3 DataSets : train, valid and test. ' ..
      'train_set is used for optimizing (e.g. backpropagating) a ' ..
      'Criteria to a model. ' ..
      'valid_set is used for cross-validation during training ' ..
      '(e.g. for early-stopping). ' ..
      'test_set is used to evaluate generalization performance ' ..
      'after training (e.g. to compare different models).' ..
      'Preprocessing can be performed on all ' .. 
      'DataSets by fitting the preprocess (e.g. Standardization) on ' ..
      'only the training set, and reusing the same statistics on ' ..
      'the validation and test sets',
      {arg='train_set', type='dp.DataSet',
       help='used for optimizing (e.g. backpropagating) a ' ..
      'Criteria to a model. '},
      {arg='valid_set', type='dp.DataSet',
       help='used for cross-validation during training ' ..
      '(e.g. for early-stopping). '},
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
   self:setTrainSet(train_set)
   self:setValidSet(valid_set)
   self:setTestSet(test_set)
   --preprocessing
   self:setInputPreprocess(input_preprocess)
   self:setTargetPreprocess(target_preprocess)
   self:preprocess()
end

function DataSource:write(...)
   error"DataSource Error: Shouldn't serialize DataSource"
end

function DataSource:setTrainSet(train_set)
   self._train_set = train_set
end

function DataSource:setValidSet(valid_set)
   self._valid_set = valid_set
end

function DataSource:setTestSet(test_set)
   self._test_set = test_set
end

function DataSource:trainSet()
   return self._train_set
end

function DataSource:validSet()
   return self._valid_set
end

function DataSource:testSet()
   return self._test_set
end

function DataSource:setInputPreprocess(input_preprocess)
   if not torch.typename(input_preprocess) 
      and type(input_preprocess) == 'table' 
   then
      input_preprocess = dp.Pipeline(input_preprocess)
   end
   self._input_preprocess = input_preprocess
end

function DataSource:setTargetPreprocess(target_preprocess)
   if not torch.typename(target_preprocess) 
      and type(target_preprocess) == 'table' 
   then
      target_preprocess = dp.Pipeline(target_preprocess)
   end
   self._target_preprocess = target_preprocess
end

function DataSource:inputPreprocess()
   return self._input_preprocess
end

function DataSource:targetPreprocess()
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

function DataSource:name()
   return self._name
end

--optional
function DataSource:classes()
   return self._classes
end

--optional
function DataSource:imageSize()
   return self._image_size
end


-- Check locally and download datasource if not found.  
-- Returns the path to the resulting data file.
function DataSource.getDataPath(...)
   local args, name, url, data_dir, decompress_file
      = xlua.unpack(
         {... or {}},
         'getDataPath', 
         'Check locally and download datasource if not found. ' ..
         'Returns the path to the resulting data file. ' ..
         'Decompress if data_dir/name/decompress_file is not found',
         {arg='name', type='string', req=true, 
          help='name of the DataSource (e.g. "mnist", "svhn", etc). ' ..
          'A directory with this name is created within ' ..
          'data_directory to contain the downloaded files. Or is ' ..
          'expected to find the data files in this directory.'},
         {arg='url', type='string', help='URL from which data ' ..
          'can be downloaded in case it is not found in the path.'},
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

   print("checking for file located at: ", data_path)

   check_and_mkdir(data_dir)
   check_and_mkdir(datasrc_dir)
   check_and_download_file(data_path, url)
   
   -- decompress 
   if decompress_file then
      local decompress_path = paths.concat(datasrc_dir, decompress_file)

      if not is_file(decompress_path) then
        do_with_cwd(datasrc_dir,
          function()
              print("decompressing file: ", data_path)
              dp.decompress_file(data_path)
          end)
      end
      return decompress_path
   end
   
   return data_path
end


function DataSource.rescale(data, min, max)
   local range = max - min
   local dmin = data:min()
   local dmax = data:max()
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
