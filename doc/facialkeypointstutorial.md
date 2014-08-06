<a name="FacialKeypointsTutorial"/>
# Facial Keypoints Tutorial #
In this tutorial, we demonstrate how the __dp__ library can be used 
to build convolution neural networks and easily extended using Feedback 
objects and the Mediator. To make things more spicy, we consider 
a case study involving its practical application to a 
[Kaggle](https://www.kaggle.com) challenge provided by the University of Montreal: 
[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).
We will attempt to keep this tutorial as complete yet concise as possible.

## Planning and Analysis ##
The first step is to determine how to approach the problem and outline the 
necessary components that will be needed to get the model working. 
It is in this step that one plans the final model(s) and components that
will be required to get your experiments running. In our case, we had 
already implemented a similar model in Pylearn2 such that we had a general 
idea what worked well for this particular problem. 

The problem has each 96x96 black-and-white images associated to 
15 keypoints, each identifies by an (x,y) coordinate. The problem is 
thus a regression where the target is a vector of 15x2=30 values 
bounded between 0 and 96, the size of the image. If you think like me, 
your initial reflex might be to use a simple multi-layer perception 
(or neural network) with a Linear output and a Mean Square Error 
Criterion. Or maybe we can bound the output by using a Sigmoid 
(which bound it between 0 and 1), and then scale the output by a 
constant greater than 96. 

However, these approaches don't work well in practice as they don't 
correctly capture the spatial localities. An alternative 
solution is to model the output space as 30 vectors of size 97, and 
translate each target value to a small (standard deviation of about 1) 
gaussian blur centered at the keypoint coordinate. This increases the 
precision of the new targets as compared to just using a one-hot vector 
(a vector with one 1, the rest being zeros).

The use of a gaussian blur centered on the target, which amounts to 
predicting multinomial probabilities, can be combined with the 
DistKLDivCriterion to train a SoftMax output for each keypoint. However, 
a MultSoftMax Module would need to be implemented to accomodate this 
use case.

As for any new problem, we also need to adapt the Kaggle Facial Keypoints 
Detection dataset to __dp__ by wrapping it in a DataSource. We will 
also require a simple baseline which we can compare our own models to,
and use to test the correctness of our Kaggle submissions 
(test-set predictions). A Feedback object will be required 
for comparing our Mean Square Error on the train and valid set to the 
baseline predictor. And another Feedback will be required for
preparing Kaggle submissions when new minima on the valid set are found 
(where these minima will be evaluated using the above Feedback object).

## Building Components ##
From the above analysis, we can begin to draw a roadmap of components to 
build :
 1. FacialKeypoints : wrapper for the DataSource;
 2. facialkeypointsdetector.lua : launch script with cmd-line options for specifying Model assembly and Experiment hyper-parameters; 
 3. FKDKaggle : a Feedback for creating a Kaggle submission out of predictions;
 4. FacialKeypoints : a Feedback for monitoring performance (and comparing to baseline);
 5. nn.MultiSoftMax : a nn.Module that will allow us to apply a softmax for each keypoint.

### FacialKeypoints ###
The first task of any machine learning endeavor is to prepare the 
dataset for use within the library. In this case, the 
[data](https://www.kaggle.com/c/facial-keypoints-detection/data) was 
provided in CSV-format as `training.csv` and `test.csv` file. 
So we went shopping on GitHub for a free open-source CSV library and 
found Clement Farabet's [csvigo](https://github.com/clementfarabet/lua---csv)
which can be installed through [luarocks](http://www.luarocks.org/).

We loaded the two CSV files into Tensors using 
[th](https://github.com/torch/trepl) (at the time, we didn't think to log this process). 
We shuffled the training set and saved both Tensors into `train.th7` 
and `test.th7` files. 

The dataset wrapper, [FacialKeypoints](), inherits DataSource:
```lua
local FacialKeypoints, DataSource = torch.class("dp.FacialKeypoints", "dp.DataSource")
FacialKeypoints.isFacialKeypoints = true
```
The wrapper has some static attributes like the name (which is also the 
name of the directory where the data will be stored), size of the images 
(for use in [Convolution2D](model.md#dp.convolution2d) Layers), 
collapsed feature size (for use in [Neural](model.md#dp.Neural) Layers),
the image and target axes (or views) as used in [Views](view.md#dp.View).
```lua
FacialKeypoints._name = 'FacialKeypoints'
FacialKeypoints._image_size = {1, 96, 96}
FacialKeypoints._feature_size = 1*96*96
FacialKeypoints._image_axes = 'bchw'
FacialKeypoints._target_axes = 'bwc'
```
Next is the almighty constructor which takes a dictionary of 
keyword arguments. The `valid_ratio` specifies the proportion of 
the `train.th7` Tensor to be allocated for cross-validation (the 
validation set). The `download_url` specifies the location of the 
zipped data (`train.th7`, `test.th7`, `submissionFileFormat.csv`, 
`baseline.th7` and `baseline.csv`). This allows the user to download 
all the data required by this wrapper. The `data_path` specifies where 
this data will be stored. It default to `dp.DATA_DIR` which defaults 
to environment variables `$DEEP_DATA_PATH` or `$TORCH_DATA_PATH/data` 
(which you can specify in your `~/.bashrc file` on Ubuntu). The `stdv` 
specifies the standard deviation of the gaussian blur used for making 
the MultiSoftMax targets. The `scale` table scales the image pixels 
between two numbers (between 0 and 1 in this case):
```lua 
function FacialKeypoints:__init(config) 
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, load_all
   args, self._valid_ratio, self._train_file, self._test_file, 
      self._data_path, self._download_url, self._stdv, self._scale, 
      self._shuffle, load_all = xlua.unpack(
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
       default='http://stife076.files.wordpress.com/2014/08/FacialKeypoints.zip',
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
   self._scale = self._scale or {0,1}
   self._pixels = torch.range(0,97):float()
   if load_all then
      self:loadTrain()
      self:loadValid()
      self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
   })
end
```

```lua
function FacialKeypoints:loadTrain()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = self:loadData(self._train_file, self._download_url)
   local start = 1
   local size = math.floor(data:size(1)*(1-self._valid_ratio))
   local train_data = data:narrow(1, start, size)
   self:setTrainSet(self:createTrainSet(train_data, 'train'))
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
   self:setValidSet(self:createTrainSet(valid_data, 'valid'))
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
   self:setTestSet(dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set})
   return self:testSet()
end

--Creates an Mnist Dataset out of data and which_set
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
   return dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
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
``` 
