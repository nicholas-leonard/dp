<a name="FacialKeypointsTutorial"/>
[]()
# Facial Keypoints Tutorial #

In this tutorial, we demonstrate how the __dp__ library can be used 
to build convolution neural networks and easily extended using Feedback 
objects and the Mediator. To make things more spicy, we consider 
a case study involving its practical application to a 
[Kaggle](https://www.kaggle.com) challenge provided by the University of Montreal: 
[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).

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
your initial reflex might be to use a simple multi-layer perceptron 
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

  1. [FacialKeypoints](#facialkeypoints) : wrapper for the DataSource;
  2. [FKDKaggle](#fkdkaggle) : a Feedback for creating a Kaggle submission out of predictions;
  3. [FacialKeypointFeedback](#facialkeypointfeedback) : a Feedback for monitoring performance (and comparing to baseline);
  4. [MultiSoftMax](#multisoftmax) : a nn.Module that will allow us to apply a softmax for each keypoint;
  5. [facialkeypointsdetector.lua](#facialkeypointsdetector.lua) : main launch script; 

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

The dataset wrapper, [FacialKeypoints](https://github.com/nicholas-leonard/dp/blob/master/data/facialkeypoints.lua), inherits DataSource:
```lua
local FacialKeypoints, DataSource = torch.class("dp.FacialKeypoints", "dp.DataSource")
FacialKeypoints.isFacialKeypoints = true
```
The wrapper has some static attributes like the name (which is also the 
name of the directory where the data will be stored), size of the images 
(useful for initializing [SpatialConvolution](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution) modules), 
collapsed feature size (useful for initializing [Linear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear) modules),
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
All three methods `load[Train,Valid,Test]()` are used to 
respectively load the train, valid and test DataSets. The train and valid 
sets are treated differently than the test set as the former are shuffled
by default and the latter doesn't have any targets per se, as these 
are hidden on Kaggle for the purpose of allowing objective scoring.
All methods are similar, but we will focus on `loadTrain`:
```lua
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
```
All three methods use the `loadData` method internally, which requires 
a `file_name` and `download_url`. They use the `DataSource.getDataPath` 
function to `torch.load` serialized data, as do most of the DataSource 
subclasses. If the file `data_dir/name/FacialKeypoints.zip` cannot be located, 
it is dowloaded from `download_url` and decompressed.:
```lua
function FacialKeypoints:loadData(file_name, download_url)
   local path = DataSource.getDataPath{
      name=self._name, url=download_url, 
      decompress_file=file_name, 
      data_dir=self._data_path
   }
   return torch.load(path)
end
```
The `loadTrain` and `loadValid` methods then narrow the returned 
data using the constructor-specified `valid_ratio`. They then pass their 
chunk of data to the `createTrainSet` method which wraps the inputs 
in an `ImageView` and the targets in a `SequenceView`. Both of these 
are then wrapped in a `DataSet`:
```lua
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
```
It uses a `makeTargets` method to transform a scalar keypoint 
coordinate (for one axis) into a vector of size 98 with a gaussian 
blur centered around the original scalar value. So a Tensor of size 
`(batchSize, nKeypoints*2)` is transformed into another of size 
`(batchSize, nKeypoints*2, 98)`: 
```lua
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
```
Finally, we include a method for loading the `submissionFileFormat.csv` 
which is used to prepare kaggle submissions, and another for loading 
a constant value `baseline.th7`, which contains the mean keypoints from 
the training set:
```lua
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

It is good practice to make all data accessible from such DataSource
classes. Even if some of the data is required to initialize 
other objects. This is the case for example of the FKDKaggle and 
FacialKeypoints Feedbacks, which are initialized with the output of 
this DataSource's `loadSubmission` and `loadBaseline` methods.

### FKDKaggle ###

Feedbacks are a little tricky to get the hang of,
but are very useful for extending an experiment with task-tailored 
I/O functionality. 
The [FKDKaggle](https://github.com/nicholas-leonard/dp/blob/master/feedback/fkdkaggle.lua) is a Feedback class 
used to prepare Kaggle submissions and 
persist them to disk in CSV-format each time a new minima is found on 
the validation set. 
```lua
local FKDKaggle, parent = torch.class("dp.FKDKaggle", "dp.Feedback")
FKDKaggle.isFKDKaggle = true
```
Kaggle requires a CSV submission with two columns: 
RowId, Location. Each such RowId is associated to an 
ImageId and FeatureName in the `submissionFileFormat.csv` file. Furthermore, 
some of the images in the test set only need to predict a subset of all 
keypoints (identified by FeatureName). So we map each FeatureName string
to an index of the output space:
```lua
FKDKaggle._submission_map = {
   ['left_eye_center_x'] = 1, ['left_eye_center_y'] = 2,
   ['right_eye_center_x'] = 3, ['right_eye_center_y'] = 4,
   ['left_eye_inner_corner_x'] = 5, ['left_eye_inner_corner_y'] = 6,
   ['left_eye_outer_corner_x'] = 7, ['left_eye_outer_corner_y'] = 8,
   ['right_eye_inner_corner_x'] = 9, ['right_eye_inner_corner_y'] = 10,
   ['right_eye_outer_corner_x'] = 11, ['right_eye_outer_corner_y'] = 12,
   ['left_eyebrow_inner_end_x'] = 13, ['left_eyebrow_inner_end_y'] = 14,
   ['left_eyebrow_outer_end_x'] = 15, ['left_eyebrow_outer_end_y'] = 16,
   ['right_eyebrow_inner_end_x'] = 17, ['right_eyebrow_inner_end_y'] = 18,
   ['right_eyebrow_outer_end_x'] = 19, ['right_eyebrow_outer_end_y'] = 20,
   ['nose_tip_x'] = 21, ['nose_tip_y'] = 22,
   ['mouth_left_corner_x'] = 23, ['mouth_left_corner_y'] = 24,
   ['mouth_right_corner_x'] = 25, ['mouth_right_corner_y'] = 26,
   ['mouth_center_top_lip_x'] = 27, ['mouth_center_top_lip_y'] = 28,
   ['mouth_center_bottom_lip_x'] = 29, ['mouth_center_bottom_lip_y'] = 30
}
```
The constructor is pretty straightforward. It requires a sample `submission` 
table (see `FacialKeypoints:loadSubmission` above), and a `file_name` 
of the submissions that will be prepared for Kaggle. We also 
prepare some Tensors required for translating outputs to keypoint 
coordinate values. Notice that the `csvigo` package is imported inside 
the constructor, which doesn't make it a dependency of the entire __dp__ 
package:
```lua
function FKDKaggle:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, submission, file_name, save_dir, name = xlua.unpack(
      {config},
      'FKDKaggle', 
      'Used to prepare a Kaggle Submission for the '..
      'Facial Keypoints Detection challenge',
      {arg='submission', type='table', req=true, 
       help='sample submission table'},
      {arg='file_name', type='string', req=true,
       help='name of file to save submission to'},
      {arg='save_dir', type='string', default=dp.SAVE_DIR,
       help='defaults to dp.SAVE_DIR'},
      {arg='name', type='string', default='FKDKaggle',
       help='name identifying Feedback in reports'}
   )
   require 'csvigo'
   config.name = name
   self._save_dir = save_dir
   self._template = submission
   self._submission = {{submission[1][1],submission[1][4]}}
   self._file_name = file_name
   parent.__init(self, config)
   self._pixels = torch.range(0,97):float():view(1,1,98)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._i = 2
   self._path = paths.concat(self._save_dir, self._file_name)
end
```
The next Feedback method is called by the `Propagator` after every forward 
propagation of the Batch. Here we use it to translate the output (a 
SequenceView) to scalar coordinate values. This is done by taking the 
expectation of each coordinate. We use `torch.range` (see previous snipped)
over the coordinate space (the width and height of the image) : `0,1,2,...,97`.
The expected coordinate is taken by summing the element-wise multiplication of 
this range by each keypoint prediction. Remember that each keypoint prediction 
will be passed through a softmax to obtain multinomial probabilities. 
This method also prepares a table of submissions in a format 
that package `csvigo` understands. 
```lua
function FKDKaggle:_add(batch, output, report)
   local target = batch:targets():forward('b')
   local act = output:forward('bwc', 'torch.FloatTensor')
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   for i=1,act:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      local row = self._template[self._i]
      local imageId = tonumber(row[2])
      assert(imageId == target[i])
      while (imageId == target[i]) do
         row = self._template[self._i]
         if not row then
            break
         end
         imageId = tonumber(row[2])
         local keypointName = row[3]
         self._submission[self._i] = {
            row[1], keypoint[self._submission_map[keypointName]]
         }
         self._i = self._i + 1
      end
   end
end

function FKDKaggle:_reset()
   self._i = 2
end
```

When all test set batches and outputs (predictions) have been passed 
to the Feedback, the `submission` contains all the information 
necessary for generating the CSV file. However, this file is only 
prepared when a new minima is discovered by the EarlyStopper. Almost 
all objects part of an Experiment can communicate via the Mediator 
[Singleton](https://en.wikipedia.org/wiki/Singleton_pattern). Listeners 
need only subscribe to a channel and register a callback. In this case, both 
have the same name: `errorMinima`. The EarlySopper notifies the `errorMinima` 
channel (via the Mediator) every epoch, which results in all registered 
callbacks being notified. If a new minima is found, it passes the 
first argument as true, otherwise its false. In our case, this 
means the CSV file is created: 
```lua
function FKDKaggle:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("errorMinima", self, "errorMinima")
end

function FKDKaggle:errorMinima(found_minima)
   if found_minima then
      csvigo.save{path=self._path,data=self._submission,mode='raw'}
   end
end
```

### FacialKeypointFeedback ###

Unlike the previous Feedback which is used for preparing submissions 
base on the model predictions given the test set, this one is used for 
the train and valid set. Its can take an optional baseline Tensor, 
which contains the mean of the 30 keypoints over the train and valid set. 
We built a [simple script](https://github.com/nicholas-leonard/dp/blob/master/examples/fkdbaseline.lua) to prepare this 
baseline and generate `baseline.th7`, which can be obtained via the 
[FacialKeypoints](#FacialKeypoints) DataSource. 

The [FacialKeypointFeedback](feedback.md#dp.FacialKeypointFeedback) 
is initialized with the `baseline` (optional) and the precision (size) 
of the keypoint vectors (in our case, 98). Notice again how we 
initialize a Tensor for each intermediate operation we require. This 
allows us to reuse the same Tensor as opposed to reallocating memory 
for each batch: 
```lua
local FacialKeypointFeedback, parent = torch.class("dp.FacialKeypointFeedback", "dp.Feedback")
FacialKeypointFeedback.isFacialKeypointFeedback = true

function FacialKeypointFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, precision, baseline, name = xlua.unpack(
      {config},
      'FacialKeypointFeedback', 
      'Uses mean square error to measure error w.r.t targets.'..
      'Optionaly compares this to constant (mean) value baseline',
      {arg='precision', type='number', req=true,
       help='precision (an integer) of the keypoint coordinates'},
      {arg='baseline', type='torch.Tensor', default=false,
       help='Constant baseline used for comparison'},
      {arg='name', type='string', default='facialkeypoint',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   if baseline then
      assert(baseline:dim() == 1, "expecting 1D constant-value baseline")
      self._baseline = baseline
      self._baselineSum = torch.Tensor():zero()
   end
   self._precision = precision
   parent.__init(self, config)
   self._pixels = torch.range(0,precision-1):float():view(1,1,precision)
   self._output = torch.FloatTensor()
   self._keypoints = torch.FloatTensor()
   self._targets = torch.FloatTensor()
   self._sum = torch.Tensor():zero()
   self._count = torch.Tensor():zero()
   self._mse = torch.Tensor()
end
```
The `_add` method is very similar to the above in how scalar 
coordinates are obtained from the `output`. The mean square error (MSE)
of both the baseline and the predictions w.r.t. targets is accumulated
using `self._sum` (or `self._baselineSum`) and `self._count`:
```lua
function FacialKeypointFeedback:_add(batch, output, report)
   local target = batch:targets():forward('bwc')
   local act = output:forward('bwc', 'torch.FloatTensor')
   if not self._isSetup then
      self._sum:resize(act:size(2)):zero()
      self._count:resize(act:size(2)):zero()
      if self._baseline then
         self._baselineSum:resize(act:size(2)):zero()
      end
      self._isSetup = true
   end
   local pixels = self._pixels:expandAs(act)
   self._output:cmul(act, pixels)
   self._keypoints:sum(self._output, 3)
   self._output:cmul(target, pixels)
   self._targets:sum(self._output, 3)
   for i=1,self._keypoints:size(1) do
      local keypoint = self._keypoints[i]:select(2,1)
      local target = self._targets[i]:select(2,1)
      for j=1,self._keypoints:size(2) do
         local t = target[j]
         if t > 0.00001 then
            local err = keypoint[j] - t
            self._sum[j] = self._sum[j] + (err*err) --sum square error
            self._count[j] = self._count[j] + 1
            if (not self._baselineMse) and self._baseline then
               local err = self._baseline[j] - t
               self._baselineSum[j] = self._baselineSum[j] + (err*err)
            end
         end
      end
   end
end
```
After each pass over train, valid and test sets (or epoch), a `doneEpoch` 
notificaiton is sent to Subscribers, including this Feedback:
```lua
function FacialKeypointFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end
```
The `doneEpoch` method is the registered callback. It prints a message 
comparing the MSE of the model predictions to that of the baseline:
```lua
function FacialKeypointFeedback:doneEpoch(report)
   if self._n_sample > 0 then
      local msg = self._id:toString().." MSE = "..self:meanSquareError()
      if self._baselineMse then
         msg = msg.." vs "..self._baselineMse
      end
      print(msg)
   end
end

function FacialKeypointFeedback:meanSquareError()
   if (not self._baselineMse) and self._baseline then
      self._baselineMse = torch.cdiv(self._baselineSum, self._count):mean()
   end
   return self._mse:cdiv(self._sum, self._count):mean()
end
```
Every epoch, the MSE statistics are also reset:
```lua
function FacialKeypointFeedback:_reset()
   self._sum:zero()
   self._count:zero()
end
```
Finally, a report is prepared every epoch. It includes field `mse`, which
contains the MSE of predictions. This will be useful later when we 
will need to early-stop on minima found on the validation set:
```lua
function FacialKeypointFeedback:report()
   return { 
      [self:name()] = {
         mse = self._n_sample > 0 and self:meanSquareError() or 0
      },
      n_sample = self._n_sample
   }
end
```

### MultiSoftMax ###
Unlike the other extensions, which affect the __dp__ package, this is 
a [nn](https://github.com/torch/nn/blob/master/README.md) 
[Module](https://github.com/torch/nn/blob/master/doc/module.md#module) 
that we have chose to include via Clement Farabet's 
experimental [nnx](https://github.com/clementfarabet/lua---nnx) package.

The purpose of the MultiSoftMax Module is to take a 2D or 3D input and 
performa a softmax over the row represented by the last dimension. In 
our case, the input is a Tensor of size: `(batchSize, nKeypoints*2, 98)`.
The classic [SoftMax](https://github.com/torch/nn/blob/master/doc/transfer.md#softmax)
Module only takes 1D or 2D inputs, but has C and Cuda implementations, 
which we will reuse here:
```lua
local MultiSoftMax, parent = torch.class('nn.MultiSoftMax', 'nn.Module')

function MultiSoftMax.__init(self)
   parent.__init(self)
   self._input = torch.Tensor()
   self._output = torch.Tensor()
   self._gradInput = torch.Tensor()
   self._gradOutput = torch.Tensor()
end
```
The function `input.nn.SoftMax_updateOutput` is of the same Tensor-type
as the `input`, such that the [C-version](https://github.com/torch/nn/blob/master/generic/SoftMax.c)
is used for `torch.FloatTensors` and `torch.DoubleTensors`, while the 
[CUDA-version](https://github.com/torch/cunn/blob/master/SoftMax.cu) is 
called for the `torch.CudaTensors`. If we reshape the 3D `inputs` to 
2D by collapsing the first two dimensions, we can reuse the same functions
used in SoftMax:  
```lua
function MultiSoftMax:updateOutput(input)
   if input:dim() == 2 then
      return input.nn.SoftMax_updateOutput(self, input)
   end
   if input:dim() ~= 3 then
      error"Only supports 2D or 3D inputs"
   end
   self._input:view(input, input:size(1)*input:size(2), input:size(3))
   local output = self.output
   self.output = self._output
   input.nn.SoftMax_updateOutput(self, self._input)
   output:viewAs(self.output, input)
   self.output = output
   return self.output
end

function MultiSoftMax:updateGradInput(input, gradOutput)
   if input:dim() == 2 then
      return input.nn.SoftMax_updateGradInput(self, input, gradOutput)
   end
   self._gradOutput:view(gradOutput, input:size(1)*input:size(2), input:size(3))
   local gradInput = self.gradInput
   self.gradInput = self._gradInput
   local output = self.output
   self.output = self._output
   input.nn.SoftMax_updateGradInput(self, self._input, self._gradOutput)
   self.gradInput = gradInput:viewAs(self.gradInput, input)
   self.output = output
   return self.gradInput
end
```
We also included some unit tests before submitting our 
[MultiSoftMax](https://github.com/clementfarabet/lua---nnx/blob/master/MultiSoftMax.lua) via a
[Pull Request](https://github.com/clementfarabet/lua---nnx/pull/14) to the nnx package:
```
function nnxtest.MultiSoftMax()
   local inputSize = 7 
   local nSoftmax = 5
   local batchSize = 3
   
   local input = torch.randn(batchSize, nSoftmax, inputSize)
   local gradOutput = torch.randn(batchSize, nSoftmax, inputSize)
   local msm = nn.MultiSoftMax()
   
   local output = msm:forward(input)
   local gradInput = msm:backward(input, gradOutput)
   mytester:assert(output:isSameSizeAs(input))
   mytester:assert(gradOutput:isSameSizeAs(gradInput))
   
   local sm = nn.SoftMax()
   local input2 = input:view(batchSize*nSoftmax, inputSize)
   local output2 = sm:forward(input2)
   local gradInput2 = sm:backward(input2, gradOutput:view(batchSize*nSoftmax, inputSize))
   
   mytester:assertTensorEq(output, output2, 0.000001)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)
end
```

### facialkeypointsdetector.lua ###

The final component is the [script](https://github.com/nicholas-leonard/dp/blob/master/examples/facialkeypointdetector.lua) 
that provides different cmd-line options for specifying Model assembly 
and Experiment hyper-parameters.

## Running Experiments ##

The last step is to use the components assembled in the launch script 
to run some experiments, and maybe even optimize hyper-parameters:
```bash
th examples/facialkeypointdetector.lua --learningRate 0.1 --maxOutNorm 2 --cuda --useDevice 1 --batchSize 16 --channelSize '{96,96}' --kernelSize '{7,7}' --poolSize '{2,2}' --poolStride '{2,2}' --hiddenSize 3000 --maxEpoch 1000 --maxTries 100 --accUpdate --normalInit --activation ReLU --submissionFile cnn1.csv
```
