-------------------------------------------------
--[[ CIFAR100 ]]--
-- http://www.cs.toronto.edu/~kriz/cifar.html
-- A color image set of 100 different objects
-- Small size makes it hard to generalize from train to test set
-- Regime : overfitting.
-------------------------------------------------
local Cifar100, parent = torch.class("dp.Cifar100", "dp.DataSource")
Cifar100.isCifar100 = true

Cifar100._name = 'cifar100'
Cifar100._image_size = {3, 32, 32}
Cifar100._feature_size = 3*32*32
Cifar100._image_axes = 'bchw'

function Cifar100:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess

   self.args, self._valid_ratio, self._shuffle, self._train_file, 
   self._test_file, self._data_path, self._scale, self._download_url, 
   load_all, input_preprocess, target_preprocess
   = xlua.unpack( 
   {config},
   'Cifar100', nil,
   {arg='valid_ratio', type='number', default=1/5,
    help='proportion of training set to use for cross-validation.'},
   {arg='shuffle', type='boolean', default=true,
    help='shuffle train set before splitting into train/valid'},
   {arg='train_file', type='string', 
    default='cifar-100-binary/train.bin', help='name of test_file'},
   {arg='test_file', type='string', 
    default='cifar-100-binary/test.bin', help='name of test_file'},
   {arg='data_path', type='string', default=dp.DATA_DIR,
    help='path to data repository'},
   {arg='scale', type='table', 
    help='bounds to scale the values between'},
   {arg='download_url', type='string',
    default='http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz',
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
   self._coarse_classes = self:loadClasses(
      'cifar-100-binary/coarse_label_names.txt'
   )
   assert(#self._coarse_classes == 20)
   self._fine_classes = self:loadClasses(
      'cifar-100-binary/fine_label_names.txt'
   )
   assert(#self._fine_classes == 100)
   self._classes = self._fine_classes
   
   if load_all then
      self:loadTrainValid()
      self:loadTest()
   end
   
   parent.__init(
      self, {
         train_set=self:trainSet(), valid_set=self:validSet(),
         test_set=self:testSet(),
         input_preprocess=input_preprocess,
         target_preprocess=target_preprocess
      }
   )
end

function Cifar100:loadClasses(filename)
   local path = parent.getDataPath{
      name=self._name, url=self._download_url, 
      decompress_file=filename, data_dir=self._data_path
   }
   local f = assert(io.open(path, "rb"))
   local classes = f:read("*all")
   f:close()
   return _.split(classes,'\n')
end

function Cifar100:loadTrainValid()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file, self._download_url)
   if self._shuffle then
      print"shuffling train/valid set"
      data = data:index(1, torch.randperm(data:size(1)):long())
   end
   local size = math.floor(data:size(1)*(1-self._valid_ratio))
   local train_data = data:narrow(1, 1, size)
   self:trainSet(self:createDataSet(train_data, 'train'))
   local start = size + 1
   local size = data:size(1)-start
   local valid_data = data:narrow(1, start, size)
   self:validSet(self:createDataSet(valid_data, 'valid'))
end

function Cifar100:loadTest()
   local test_data = self:loadData(self._test_file, self._download_url)
   if self._shuffle then
      print"shuffling test set"
      test_data = test_data:index(1, torch.randperm(test_data:size(1)):long())
   end
   self:testSet(self:createDataSet(test_data, 'test'))
end

function Cifar100:createDataSet(data, which_set)
   local inputs = data:narrow(2, 3, self._feature_size):clone():double()
   inputs:resize(inputs:size(1), unpack(self._image_size))
   if self._scale then
      parent.rescale(inputs, self._scale[1], self._scale[2])
   end
   
   local coarse_targets = data[{{},1}]:clone()
   local fine_targets = data[{{},2}]:clone()
   
   -- class 0 will have index 1, class 1 index 2, and so on.
   coarse_targets:add(1):resize(coarse_targets:size(1)):float()
   fine_targets:add(1):resize(fine_targets:size(1)):float()
   
   -- construct inputs and targets dp.Views 
   local input_v = dp.ImageView()
   self._coarse_targets = dp.ClassView()
   self._fine_targets = dp.ClassView()
   
   input_v:forward(self._image_axes, inputs)
   self._coarse_targets:forward('b', coarse_targets)
   self._fine_targets:forward('b', fine_targets)
   
   self._coarse_targets:setClasses(self._classes)
   self._fine_targets:setClasses(self._classes)
   -- construct dataset
   local ds = dp.DataSet{inputs=input_v,targets=self._fine_targets,which_set=which_set}
   ds:ioShapes('bchw', 'b')
   return ds
end

--Returns a 10,000 or 50,000 x 3074 tensor, 
--where each image is 32*32*3 = 3072 
--values in the range [0-255], and the 1st and 2nd elements are 
--the coarse and fine class ID.
function Cifar100:loadData(file_name, download_url)
   local path = parent.getDataPath{
      name=self._name, url=download_url, decompress_file=file_name, 
      data_dir=self._data_path
   }
   local f = assert(io.open(path, "rb"))
   local block = 3074
   local bytes = f:read("*all")
   local n_example = #bytes/block
   assert(n_example == 50000 or n_example == 10000)
   local data = torch.ByteTensor(n_example, block)
   data:storage():string(bytes)
   assert(f:close())
   return data
end

local function cifar100test(num_images)
   local c = dp.Cifar100()
   require 'image'
   local dt = c:trainSet():inputs(1)
   for idx = 1,num_images do
      local img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('cifar100image'..idx..'.png', img)
   end
   dt:feature()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      image.savePNG('cifar100feature'..idx..'.png', img)
   end
   c:inputPreprocess(dp.GCN())
   c:preprocess()
   for idx = 1,num_images do
      img = dt:image():select(1,idx):transpose(1,3)
      print(dt:image():select(1,idx):size())
      image.savePNG('cifar100gcn'..idx..'.png', img)
   end
end
