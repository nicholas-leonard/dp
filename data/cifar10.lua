-------------------------------------------------
--[[ CIFAR10 ]]--
-- http://www.cs.toronto.edu/~kriz/cifar.html
-- A color-image set of 10 different objects.
-- Small size makes it hard to generalize from train to test set.
-- Regime : overfitting.
-------------------------------------------------

local Cifar10, parent = torch.class("dp.Cifar10", "dp.DataSource")
Cifar10.isCifar10 = true

Cifar10._name = 'cifar10'
Cifar10._image_size = {3, 32, 32}
Cifar10._feature_size = 3*32*32
Cifar10._image_axes = 'bchw'
Cifar10._classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

function Cifar10:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess

   self.args, self._valid_ratio, self._data_folder, self._data_path,
   self._scale, self._download_url, load_all, 
   input_preprocess, target_preprocess
      = xlua.unpack( 
      {config},
      'Cifar10', nil,
      {arg='valid_ratio', type='number', default=1/5,
        help='proportion of training set to use for cross-validation.'},
      {arg='data_folder', type='string', default='cifar-10-batches-t7',
        help='name of test_file'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
        help='path to data repository'},
      {arg='scale', type='table', 
        help='bounds to scale the values between'},
      {arg='download_url', type='string',
        default='http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz',
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
   if (self._scale == nil) then
      self._scale = {0,1}
   end
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadTest()
   end
   parent.__init(self, {train_set=self:trainSet(), 
                        valid_set=self:validSet(),
                        test_set=self:testSet(),
                        input_preprocess=input_preprocess,
                        target_preprocess=target_preprocess})
end

function Cifar10:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._download_url, 'train')
   local size = math.floor(data:size(1)*(1-self._valid_ratio))
   local train_data = data:narrow(1, 1, size)
   self:trainSet(self:createDataSet(train_data, 'train'))
   return self:trainSet()
end

function Cifar10:loadValid()
   data = self:loadData(self._download_url, 'train')
   if self._valid_ratio == 0 then
      print"Warning : No Valid Set due to valid_ratio == 0"
      return
   end
   local start = math.ceil(data:size(1)*(1-self._valid_ratio))
   local size = data:size(1)-start
   local valid_data = data:narrow(1, start, size)
   self:validSet(self:createDataSet(valid_data, 'valid'))
   return self:validSet()
end

function Cifar10:loadTest()
   local test_data = self:loadData(self._download_url, 'test')
   self:testSet(self:createDataSet(test_data, 'test'))
   return self:testSet()
end

function Cifar10:createDataSet(data, which_set)
   local inputs = data:narrow(2, 1, self._feature_size):clone()
   inputs = inputs:type('torch.DoubleTensor')
   inputs:resize(inputs:size(1), unpack(self._image_size))
   if self._scale then
      parent.rescale(inputs, self._scale[1], self._scale[2])
   end
   --inputs:resize(inputs:size(1), unpack(self._image_size))
   local targets = data:select(2, self._feature_size+1):clone()
   -- class 0 will have index 1, class 1 index 2, and so on.
   targets:add(1)
   targets = targets:type('torch.DoubleTensor')
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

--Returns a 50,000 x 3073 tensor, where each image is 32*32*3 = 3072 values in the
--range [0-255], and the 3073th element is the class ID.
function Cifar10:loadData(download_url, which_set)
   local path = dp.DataSource.getDataPath{
      name=self._name, url=download_url, data_dir=self._data_path,
      decompress_file='cifar-10-batches-t7/data_batch_1.t7'
   }

   local dir = paths.dirname(path) --.. '/' .. self._data_folder

   if which_set == 'train' then
      local tensor = torch.ByteTensor(50000, 3073)
      local startIdx = 1
      local train_files = {'data_batch_1.t7',
                         'data_batch_2.t7',
                         'data_batch_3.t7',
                         'data_batch_4.t7',
                         'data_batch_5.t7'}
                 
      for _,f_name in pairs(train_files) do
         local data_path = dir .. '/' .. f_name
         local f = torch.DiskFile(data_path, 'r')
         local t = f:readObject()
         local n_example = t.data:size(2)
         local n_feature = t.data:size(1)
         assert(n_feature == 3072)
         tensor[{{startIdx, startIdx+n_example-1},{1, n_feature}}] = t.data:t()
         tensor[{{startIdx, startIdx+n_example-1},{n_feature+1}}] = t.labels
         startIdx = startIdx + n_example
         f:close()
      end
      assert(startIdx-1 == 50000, 'total number of examples is not equal to 50000')
      return tensor

   elseif which_set == 'test' then
      local tensor = torch.ByteTensor(10000, 3073)
      local test_file = 'test_batch.t7'  
      local data_path = dir .. '/' .. test_file
      local f = torch.DiskFile(data_path, 'r')
      local t = f:readObject()
      local n_example = t.data:size(2)
      local n_feature = t.data:size(1)
      assert(n_feature == 3072)
      assert(n_example == 10000)
      tensor[{{1, n_example},{1, n_feature}}] = t.data:t()
      tensor[{{1, n_example},{n_feature+1}}] = t.labels
      f:close()
      return tensor
   end
       
end

local function cifar10test(num_images)
   local c = dp.Cifar10()
   require 'image'
   local dt = c:trainSet():inputs(1)
   for idx = 1,num_images do
      local img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('cifar10image'..idx..'.png', img)
   end
   dt:feature()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('cifar10feature'..idx..'.png', img)
   end
   c:inputPreprocess(dp.LeCunLCN())
   c:preprocess()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      print(dt:image():select(1,idx):size())

      image.savePNG('cifar10lecun'..idx..'.png', img)
   end
end
