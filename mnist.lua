require 'torch'
require 'image'
require 'paths'
require 'dok'

require 'xlua'

require 'dataset'
require 'dataset/TableDataset'
require 'dataset/whitening'

local Mnist, parent = torch.class("dataset.Mnist", "dataset.TableDataset")

Mnist.name         = 'mnist'
local dimensions   = {1, 28, 28}
local n_dimensions = 1 * 28 * 28
Mnist.classes      = {[0] = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Mnist.url          = 'http://data.neuflow.org/data/mnist-th7.tgz'

local function load_data_file(path)
    local f = torch.DiskFile(path, 'r')
    f:binary()

    local n_examples   = f:readInt()
    local n_dimensions = f:readInt()

    local tensor       = torch.Tensor(n_examples, n_dimensions)
    tensor:storage():copy(f:readFloat(n_examples * n_dimensions))

    return n_examples, n_dimensions, tensor
end

-- Get the raw, unprocessed dataset.
-- Returns a 60,000 x 785 tensor, where each image is 28*28 = 784 values in the
-- range [0-255], and the 785th element is the class ID.
function Mnist:load_data(file_name, dataset_name, download_url)
   local path = dataset.data_path(dataset_name or Mnist.name, 
                                  download_url or Mnist.url, 
                                  file_name, self.data_path)
   n_examples, n_dimensions, data = load_data_file(path)
   return data
end


-- Setup an MNIST dataset instance.
--
--   m = dataset.Mnist{which_set='train'}
--
--   -- scale values between [0,1] (by default they are in the range [0,255])
--   m = dataset.Mnist{which_set='train', scale = {0, 1}}
--
--   -- or normalize (subtract mean and divide by std)
--   m = dataset.Mnist{which_set='train', normalize = true}
--
function Mnist:__init(...)   
   local args
   args, self.which_set, self.valid_ratio, self.train_file, 
         self.test_file, self.data_path, self.pp_scale, self.pp_binarize, 
         self.pp_sort, self.pp_normalize, self.pp_zca_whiten 
      = xlua.unpack(
      {...},
      'Mnist constructor', nil,
      {arg='which_set', type='string', help="'train', 'valid' or 'test'", req=true},
      {arg='valid_ratio', type='number', help='proportion of training set to use for cross-validation', default=1/6},
      {arg='train_file', type='string', help='name of test_file', default='mnist-th7/train.th7'},
      {arg='test_file', type='string', help='name of test_file', default='mnist-th7/test.th7'},
      {arg='data_path', type='string', help='path to data repository', default=dataset.DATA_PATH},
      {arg='scale', type='table', help='bounds to scale the values between', default={0,1}},
      {arg='binarize', type='boolean', help='binarize the inputs (0s and 1s)', default=false},
      {arg='sort', type='boolean', help='', default=false},
      {arg='normalize', type='boolean', help='perform global standardization (-mean/stdev)', default=false},
      {arg='zca_whiten', type='boolean', help='perform zca_whitenining', default=false}
   )

   --Data will contain a tensor where each row is an example, and where
   --the last column contains the label.
   local data
   if self.which_set == 'train'  or self.which_set == 'valid' then
      data = self:load_data(self.train_file)
      local start, size
      if self.which_set == 'train' then
         start = 1
         size = math.floor(data:size(1)*(1-self.valid_ratio))
      else
         start = math.ceil(data:size(1)*(1-self.valid_ratio))
         size = data:size(1)-start
      end
      data = data:narrow(1, start, size)
   elseif self.which_set == 'test' then
      data = self:load_data(self.test_file)
   else 
      assert(false)
   end
   samples = data:narrow(2, 1, n_dimensions):clone()
   samples:resize(samples:size(1), unpack(dimensions))
   labels = data:narrow(2, n_dimensions, 1):clone()
   -- class 0 will have index 1, class 1 index 2, and so on.
   labels:add(1)
   labels:resize(labels:size(1))

   if self.pp_sort then
      samples, labels = dataset.sort_by_class(samples, labels)
   end

   if (#(self.pp_scale) == 2) then
      dataset.scale(samples, self.pp_scale[1], self.pp_scale[2])
   end

   --local d = dataset.TableDataset({data = samples, class = labels}, Mnist)
   parent.__init(self, {data = samples, class = labels}, self)

   if self.pp_binarize then
      if #scale == 2 then
          threshold = (scale[2]+scale[1])/2
      else 
          threshold = 128
      end
      self:binarize(threshold)
   end

   if self.pp_normalize then
      self:normalize_globally()
   end

   if self.pp_zca_whiten then
      self:zca_whiten()
   end

   return self

end

