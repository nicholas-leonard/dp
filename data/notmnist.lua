------------------------------------------------------------------------
--[[ NotMnist ]]--
-- http://yaroslavvb.blogspot.ca/2011/09/notmnist-dataset.html
-- http://yaroslavvb.com/upload/notMNIST/
-- A 500k+ example alternative to MNIST using unicode fonts.
------------------------------------------------------------------------

local NotMnist, DataSource = torch.class("dp.NotMnist", "dp.DataSource")
NotMnist.isNotMnist= true

NotMnist._name = 'notMnist'
NotMnist._image_size = {28, 28, 1}
NotMnist._image_axes = 'bhwc'
NotMnist._feature_size = 28*28*1
NotMnist._classes = {'A','B','C','D','E','F','G','H','I','J'}

function NotMnist:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess
   self._args, self._valid_ratio, self._train_dir, self._test_dir, 
         self._data_path, self._scale, self._binarize, 
         self._download_url, load_all, input_preprocess, 
         target_preprocess
      = xlua.unpack(
      {config},
      'NotMnist', 
      'A 500k+ example alternative to MNIST using unicode fonts.',
      {arg='valid_ratio', type='number', default=1/6,
       help='proportion of training set to use for cross-validation.'},
      {arg='train_dir', type='string', default='notMNIST_large',
       help='name of train_dir'},
      {arg='test_dir', type='string', default='notMNIST_small',
       help='name of test_dir'},
      {arg='data_path', type='string', default=dp.DATA_DIR,
       help='path to data repository'},
      {arg='scale', type='table', 
       help='bounds to scale the values between', default={0,1}},
      {arg='binarize', type='boolean', 
       help='binarize the inputs (0s and 1s)', default=false},
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
   )
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

function NotMnist:loadTrainValid()
   local bad_png = { --these wont load as PNGs, ignore them
      ['A'] = {
         'RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png',
         'Um9tYW5hIEJvbGQucGZi.png', 
         'SG90IE11c3RhcmQgQlROIFBvc3Rlci50dGY=.png'
      },
      ['B'] = {'TmlraXNFRi1TZW1pQm9sZEl0YWxpYy5vdGY=.png'},
      ['D'] = {'VHJhbnNpdCBCb2xkLnR0Zg==.png'}
   }
   local inputs, targets = self:loadData(
      self._train_dir, self._download_url, bad_png
   )
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

function NotMnist:loadTest()
   local bad_png = {
      ['A'] = {'RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png'},
      ['F'] = {'Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png'}
   }
   local inputs, targets = self:loadData(
      self._test_dir, self._download_url, bad_png
   )
   self:testSet(self:createDataSet(inputs, targets, 'test'))
   return self:testSet()
end

--Creates an NotMnist Dataset out of data and which_set
function NotMnist:createDataSet(inputs, targets, which_set)
   if self._binarize then
      DataSource.binarize(inputs, 128)
   end
   if self._scale and not self._binarize then
      DataSource.rescale(inputs, self._scale[1], self._scale[2])
   end   
   -- construct inputs and targets dp.Views 
   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward(self._image_axes, inputs)
   target_v:forward('b', targets)
   target_v:setClasses(self._classes)
   -- construct dataset
   return dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
end

function NotMnist:loadData(file_name, download_url, bad_png)
   local doc_path = DataSource.getDataPath{
      name=self._name, url=download_url..file_name..'.tar.gz', 
      decompress_file=file_name, data_dir=self._data_path
   }
   
   local n_example = 0
   local classes = self._classes
   for classidx, class in ipairs(classes) do
      local classpath = paths.concat(doc_path, class)
      for file in lfs.dir(classpath) do 
         if #file > 2 and not _.contains(bad_png[class] or {}, file) then 
            n_example = n_example + 1 
         else
            --print(class, file)
         end
      end
   end
   
   local inputs = torch.FloatTensor(n_example, unpack(self._image_size))
   local targets = torch.Tensor(n_example)
   local shuffle = torch.randperm(n_example) --useless for test set

   local example_idx = 1
   for classidx, class in ipairs(classes) do
      local classpath = paths.concat(doc_path, class)
      for file in lfs.dir(classpath) do
         if #file > 2 and not _.contains(bad_png[class] or {}, file) then 
            local filename = paths.concat(classpath, file)
            local ds_idx = shuffle[example_idx]
            inputs[{ds_idx,{},{},{}}] = image.loadPNG(filename):float()
            targets[ds_idx] = classidx
            example_idx = example_idx + 1
         end
      end
   end
  
   return inputs, targets
end


