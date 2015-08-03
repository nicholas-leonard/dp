-------------------------------------------------
--[[ Svhn ]]--
-- Ref.: A. http://ufldl.stanford.edu/housenumbers
-- B. http://yann.lecun.com/exdb/publis/psgz/sermanet-icpr-12.ps.gz
-- A color-image set of 10 digits.
-------------------------------------------------
local Svhn, DataSource = torch.class("dp.Svhn", "dp.DataSource")
Svhn.isSvhn = true

Svhn._name = 'svhn'
Svhn._image_size = {3, 32, 32}
Svhn._feature_size = 3*32*32
Svhn._image_axes = 'bchw' 
Svhn._classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

function Svhn:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess

   self.args, self._valid_ratio, self._valid_per_class, self._use_extra,
   self._extra_file, self._train_file, self._test_file,
   self._data_path, self._shuffle, self._scale, self._download_url, 
   load_all, input_preprocess, target_preprocess
      = xlua.unpack(
      {config},
      'Svhn', 
      'Street View House Numbers datasource',
      {arg='valid_ratio', type='number | table',
       help='proportion of training set to use for cross-validation.'..
       'A pair of numbers can be used to specify {train, extra}.'},
      {arg='valid_per_class', type='number | table',
       help='number of images per class to use for cross-validation.'..
       'A pair of numbers can be used to specify {train, extra}. '..
       'Defaults to the recommended {400,200}. '..
       'http://yann.lecun.com/exdb/publis/psgz/sermanet-icpr-12.ps.gz'},
      {arg='use_extra', type='boolean', default=true,
       help='use extra training data. These are a little easier, '..
       'yet greater in number.'},
      {arg='train_file', type='string', default='housenumbers/train_32x32.t7',
       help='name of train_file'},
      {arg='extra_file', type='string', default='housenumbers/extra_32x32.t7',
       help='name of extra (yet easier) patterns used for training'},
      {arg='test_file', type='string', default='housenumbers/test_32x32.t7',
       help='name of test_file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='shuffle', type='boolean',
       help='shuffle different sets', default=true},
      {arg='scale', type='table', default=false,
       help='bounds to scale the values between'},
      {arg='download_url', type='string',
       default='http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz',
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
   )
   if self._valid_per_class and self._valid_ratio then
      error"Either specify valid_per_class or valid_ratio (not both)"
   elseif torch.type(self._valid_per_class) == 'number' then
      self._valid_per_class = {self._valid_per_class, self._valid_per_class}
   elseif torch.type(self._valid_ratio) == 'number' then
      self._valid_ratio = {self._valid_ratio, self._valid_ratio}
   elseif not (self._valid_per_class or self._valid_ratio) then
      self._valid_per_class = {400, 200}
   end
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   if load_all then
      self:loadTrainValid()
      self:loadTest()
   end
   DataSource.__init(self, {train_set=self:trainSet(),
                        valid_set=self:validSet(),
                        test_set=self:testSet(),
                        input_preprocess=input_preprocess,
                        target_preprocess=target_preprocess})
end

function Svhn:_loadTrainValid(file_name, valid_ratio, valid_per_class)
   local doc_path = DataSource.getDataPath{
      name=self._name, url=self._download_url,
      decompress_file=file_name, data_dir=self._data_path
   }
   local data = torch.load(doc_path,'ascii') 
   local inputs = data.X:transpose(3,4):float() -- was ByteTensor
   data.X = nil
   collectgarbage()
   local targets = data.y:view(-1):int() --was DoubleTensor
   data = nil
   collectgarbage()
   
   local tstart, vstop = self._train_start, self._valid_stop
   
   if valid_ratio then
      -- train
      local start = 1
      local size = math.floor(inputs:size(1)*(1-valid_ratio))
      local indices = self._index_tensor:narrow(1, tstart, size)
      
      self._input_tensor:indexCopy(1, indices, inputs:narrow(1, start, size))
      self._target_tensor:indexCopy(1, indices, targets:narrow(1, start, size))
      
      self._train_start = tstart + size
      
      -- valid
      start = size + 1
      size = inputs:size(1) - start
      local vstart = vstop - size + 1
      
      local vinputs = self._input_tensor:narrow(1, vstart, size)
      vinputs:copy(inputs:narrow(1, start, size))
      
      local ttargets = self._target_tensor:narrow(1, vstart, size)
      valid_targets = targets:narrow(1, start, size)
      
      self._valid_stop = vstart - 1
   else
      self._fbuffer = self._fbuffer or torch.FloatTensor()
      self._ibuffer = self._ibuffer or torch.IntTensor()
      local class_indices = {{},{},{},{},{},{},{},{},{},{}}
      local i = 0
      targets:apply(function(class)
         i = i + 1
         table.insert(class_indices[class], i)
      end)
      
      local nValid = valid_per_class * 10
      local nTrain = targets:size(1) - nValid
      local vstart = vstop - nValid + 1
      assert(nTrain > 1, "valid per class is too high")
      
      local indices = self._index_tensor:narrow(1, tstart, nTrain)
      local valid_inputs = self._input_tensor:narrow(1, vstart, nValid)
      local valid_targets = self._target_tensor:narrow(1, vstart, nValid)
      
      local vstart_, tstart_ = 1, 1
      for class, cindices in ipairs(class_indices) do
         local cidx = torch.LongTensor(cindices)
         -- valid
         local vidx = cidx:narrow(1, 1, valid_per_class)
         local vinput = valid_inputs:narrow(1, vstart_, valid_per_class)
         local vtarget = valid_targets:narrow(1, vstart_, valid_per_class)
         vinput:index(inputs, 1, vidx)
         vtarget:index(targets, 1, vidx)
         vstart_ = vstart_ + valid_per_class
         
         -- train
         local tsize = cidx:size(1)-valid_per_class
         local tidx = cidx:narrow(1, valid_per_class+1, tsize)
         self._fbuffer:index(inputs, 1, tidx)
         self._ibuffer:index(targets, 1, tidx)

         local tindices = indices:narrow(1, tstart_, tsize)
         self._input_tensor:indexCopy(1, tindices, self._fbuffer)
         self._target_tensor:indexCopy(1, tindices, self._ibuffer)
         tstart_ = tstart_ + tsize
      end
      assert(vstart_ == valid_targets:size(1) + 1, vstart_ .. " ~= " .. valid_targets:size(1) + 1)
      assert(tstart_ == nTrain + 1, tstart_ .. "~=" .. nTrain + 1)
      self._train_start = tstart + nTrain
      self._valid_stop = vstart - 1
   end
   return train_inputs, train_targets, valid_inputs, valid_targets
end

function Svhn:loadTrainValid()
   local nSample = 73257
   local nValid = self._valid_ratio 
                  and math.floor(nSample*self._valid_ratio[1])
                  or self._valid_per_class[1] * 10
                  
   if self._use_extra then
      nSample = nSample + 531131
      nValid = nValid + (self._valid_ratio
                        and math.floor(531131*self._valid_ratio[2]) 
                        or self._valid_per_class[2] * 10)
   end
   
   self._input_tensor = torch.FloatTensor(nSample, 3, 32, 32):fill(-1)
   self._target_tensor = torch.IntTensor(nSample):fill(-1)
   self._index_tensor = self._shuffle 
                        and torch.randperm(nSample-nValid):long()
                        or torch.range(1, nSample-nValid):long()
  
   self._train_start = 1
   self._valid_stop = nSample
   

   self:_loadTrainValid(
      self._train_file, 
      self._valid_ratio and self._valid_ratio[1], 
      self._valid_per_class and self._valid_per_class[1]
   )
   
   if self._use_extra then
      self:_loadTrainValid(
         self._extra_file, 
         self._valid_ratio and self._valid_ratio[2], 
         self._valid_per_class and self._valid_per_class[2]
      )
   end
   
   assert(self._input_tensor:min() > -1)
   assert(self._target_tensor:min() > -1)
   
   self:trainSet(
      self:createDataSet(
         self._input_tensor:narrow(1, 1, nSample-nValid),
         self._target_tensor:narrow(1, 1, nSample-nValid),
         'train'
      )
   )
   
   -- valid
   self:validSet(
      self:createDataSet(
         self._input_tensor:narrow(1, nSample-nValid+1, nValid),
         self._target_tensor:narrow(1, nSample-nValid+1, nValid),
         'valid'
      )
   )

   return self:trainSet(), self:validSet()
end

function Svhn:loadTest()
   local doc_path = DataSource.getDataPath{
      name=self._name, url=self._download_url,
      decompress_file=self._test_file, data_dir=self._data_path
   }
   local data = torch.load(doc_path,'ascii')
   local inputs = data.X:transpose(3,4):float()
   local targets = data.y:view(-1)
   
   self:testSet(self:createDataSet(inputs, targets, 'test'))
   return self:testSet()
end

--Creates an Svhn Dataset out of data and which_set
function Svhn:createDataSet(inputs, targets, which_set)
   if self._scale then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end

   -- construct inputs and targets dp.Views
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)

   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'b')
   return ds
end

function Svhn:loadData(file_name, download_url)
   local doc_path = DataSource.getDataPath{
      name=self._name, url=download_url,
      decompress_file=file_name, data_dir=self._data_path
   }
   local data = torch.load(doc_path,'ascii')
   return data.X:float(), data.y:view(-1)
end
