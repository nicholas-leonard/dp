------------------------------------------------------------------------
--[[ FacialKeypoints ]]--
-- The task consists in predicting 15 (x,y) coordinate keypoints
-- from black-and-white 96x96 images. Courtesy of Yoshua Bengio's 
-- LISA Lab at Universite de Montreal.
-- The train/valid set have been pre-shuffled in train.th7
-- The test set can only be evaluated on kaggle.
-- https://www.kaggle.com/c/facial-keypoints-detection/data
------------------------------------------------------------------------
local FacialKeypoints, DataSource = torch.class("dp.FacialKeypoints", "dp.DataSource")
FacialKeypoints.isFacialKeypoints = true

FacialKeypoints._name = 'FacialKeypoints'
FacialKeypoints._image_size = {1, 96, 96}
FacialKeypoints._feature_size = 1*96*96
FacialKeypoints._image_axes = 'bchw'
FacialKeypoints._target_axes = 'bwc'

function FacialKeypoints:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, load_all
   args, self._valid_ratio, self._train_file, self._test_file, 
      self._data_path, self._download_url, self._stdv, self._scale, 
      self._shuffle, load_all, input_pp, target_pp = xlua.unpack(
      {config},
      'FacialKeypoints', 
      'https://www.kaggle.com/c/facial-keypoints-detection/data',
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_file', type='string', default='train.th7',
       help='name of training file'},
      {arg='test_file', type='string', default='test.th7',
       help='name of test file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='download_url', type='string',
       default='https://stife076.files.wordpress.com/2014/08/facialkeypoints1.zip',
       help='URL from which to download dataset if not found on disk.'},
      {arg='stdv', type='number', default=0.8, 
       help='standard deviation of the gaussian blur used for targets'},
      {arg='scale', type='table', 
       help='bounds to scale the values between. [Default={0,1}]'},
      {arg='shuffle', type='boolean', 
       help='shuffle train set', default=true},
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
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   self._pixels = torch.range(0,97):float()
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet(),
      input_preprocess=input_pp, target_preprocess=target_pp
   })
end

function FacialKeypoints:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file, self._download_url)
   local start = 1
   local size = math.floor(data:size(1)*(1-self._valid_ratio))
   local train_data = data:narrow(1, start, size)
   self:trainSet(self:createTrainSet(train_data, 'train'))
   return self:trainSet()
end

function FacialKeypoints:loadValid()
   local data = self:loadData(self._train_file, self._download_url)
   if self._valid_ratio == 0 then
      print"Warning : No Valid Set due to valid_ratio == 0"
      return
   end
   local start = math.ceil(data:size(1)*(1-self._valid_ratio))
   local size = data:size(1)-start
   local valid_data = data:narrow(1, start, size)
   self:validSet(self:createTrainSet(valid_data, 'valid'))
   return self:validSet()
end

function FacialKeypoints:loadTest()
   local data = self:loadData(self._test_file, self._download_url)
   
   local inputs = data:narrow(2, 2, 96*96):clone():view(data:size(1),1,96,96)
   local targets = data:select(2, 1):int()
   self._image_ids = data:select(2, 1):clone()
   if self._scale then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end
   
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'b')
   self:testSet(ds)
   return ds
end

function FacialKeypoints:createTrainSet(data, which_set)
   if self._shuffle then
      data = data:index(1, torch.randperm(data:size(1)):long())
   end
   local inputs = data:narrow(2, 31, 96*96):clone():view(data:size(1),1,96,96)
   local targets = self:makeTargets(data:narrow(2, 1, 30))
   
   if self._scale then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end
   
   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.SequenceView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward(self._target_axes, targets)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'bwc')
   return ds
end

function FacialKeypoints:makeTargets(y)
   -- y : (batch_size, num_keypoints*2)
   -- Y : (batch_size, num_keypoints*2, 98)
   Y = torch.FloatTensor(y:size(1), y:size(2), 98):zero()
   local pixels = self._pixels
   local stdv = self._stdv
   local k = 0
   for i=1,y:size(1) do
      local keypoints = y[i]
      local new_keypoints = Y[i]
      for j=1,y:size(2) do
         local kp = keypoints[j]
         if kp ~= -1 then
            local new_kp = new_keypoints[j]
            new_kp:add(pixels, -kp)
            new_kp:cmul(new_kp)
            new_kp:div(2*stdv*stdv)
            new_kp:mul(-1)
            new_kp:exp(new_kp)
            new_kp:div(math.sqrt(2*math.pi)*stdv)
         else
            k = k + 1
         end
      end
   end
   return Y
end

function FacialKeypoints:loadData(file_name, download_url)
   local path = DataSource.getDataPath{
      name=self._name, url=download_url, 
      decompress_file=file_name, 
      data_dir=self._data_path
   }
   return torch.load(path)
end

function FacialKeypoints:loadSubmission(path)
   path = path or DataSource.getDataPath{
      name=self._name, url=self._download_url, 
      decompress_file='submissionFileFormat.csv', 
      data_dir=self._data_path
   }
   require 'csvigo'
   local csv = csvigo.load{path=path,mode='raw'}
   -- fix weird string bug
   for i, row in ipairs(csv) do
      if i ~= 1 then
         row[3] = row[3]:sub(1,#row[3]-1)
      end
   end
   return csv
end

function FacialKeypoints:loadBaseline(path)
   path = path or DataSource.getDataPath{
      name=self._name, url=self._download_url, 
      decompress_file='baseline.th7', 
      data_dir=self._data_path
   }
   return torch.load(path)
end
