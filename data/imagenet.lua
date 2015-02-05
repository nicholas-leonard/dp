------------------------------------------------------------------------
--[[ ImageNet ]]--
-- http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php
-- Wraps the Large Scale Visual Recognition Challenge 2014 (ILSVRC2014)
-- classification dataset (commonly known as ImageNet). The dataset
-- hasn't changed from 2012-2014.
-- The validation set labels are in the devkit at 
-- data/ILSVRC2014_clsloc_validation_ground_truth.txt
------------------------------------------------------------------------
local ImageNet, DataSource = torch.class("dp.ImageNet", "dp.DataSource")

ImageNet._name = 'ImageNet'
ImageNet._image_axes = 'bchw'
ImageNet._structured_url = 'http://www.image-net.org/api/xml/structure_released.xml'
ImageNet._devkit_url = 'http://image-net.org/image/ilsvrc2014/ILSVRC2014_devkit.tgz'
ImageNet._img_train_url = 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_train.tar'
ImageNet._img_valid_url = 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_val.tar'
ImageNet._img_test_url = 'http://www.image-net.org/challenges/LSVRC/2012/nonpub/ILSVRC2012_img_test.tar'

function ImageNet:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local load_all, input_preprocess, target_preprocess
   self._args, self._load_size, self._sample_size, self._data_path, 
      self._train_dir, self._valid_dir, self._test_dir, self._devkit_dir
      self._verbose, self._sample_hook_train, self._sample_hook_test,
      self._download_url, load_all, input_preprocess, 
      target_preprocess
      = xlua.unpack(
      {config},
      'ImageNet',
      'ILSVRC2012-14 image classification dataset',
      {arg='load_size', type='table',
       help='a size to load the images to, initially'},
      {arg='sample_size', type='table',
       help='a consistent sample size to resize the images'},
      {arg='data_path', type='table | string', default=dp.DATA_DIR,
       help='one or many paths of directories with images'},
      {arg='train_dir', type='string', default='ILSVRC2012_img_train',
       help='name of train_dir'},
      {arg='valid_dir', type='string', default='ILSVRC2012_img_val',
       help='name of valid_dir'},
      {arg='test_dir', type='string', default='ILSVRC2012_img_test',
       help='name of test_dir'},
      {arg='devkit_dir', type='string', default='ILSVRC2014_devkit',
       help='name of devkit dir, which contains validation labels and '
       ..'metadata like mapings of wordnet id to class index'},
      {arg='verbose', type='boolean', default = false,
       help='Verbose mode during initialization'},
      {arg='sample_hook_train', type='function',
       help='applied to sample during training(ex: for lighting jitter). '
       .. 'It takes the image path as input'},
      {arg='sample_hook_test', type='function', 
       help='applied to sample during testing'},
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

   self._load_size = self._load_size or self._sample_size
   self._data_path = torch.type(self._data_path) == 'string' 
      and {self._data_path} or self._data_path

   if load_all then
      self:loadTrain()
      self:loadValid()
   end
end

function ImageNet:download()
   -- call download script
end

function ImageNet:loadTrain()
   
   
   -- there could be multiple trainPaths :
   local trainPaths = _.map(
   
   local dataset = dp.ImageClassSet{
      data_path=self._data_path, load_size=self._load_size,
      which_set='train', sample_size=self._sample_size,
      carry=dp.Carry(), verbose=self._verbose
   }
   self:trainSet(dataset)
   return dataset
end

function ImageNet:loadStructure()
   local path = DataSource.getDataPath{
      name=self._name, url=self._structured_url, 
      decompress_file='structure_released.xml', 
      data_dir=self._data_path
   }
   -- sudo luarocks install xml
   local xml = require 'xml'
   local structure = xml.loadpath(path)
   return structure
end

