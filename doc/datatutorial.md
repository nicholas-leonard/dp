# Data Tutorial #

Most of the theory related to deep learning seems to focus on the models and training algorithms.
In this tutorial, we will explore how to prepare the data. We will show you how to build 
DataSources for image classification and language modeling. 

## Object Structure ##

Because it tries to be general enough for different use cases, 
the data interface for the dp package can seem a little complicated. 
We can divide the data interface into 5 levels of abstractions. 
From abstract to concrete these are:
 
 1. dp.DataSource : encapsulates 1 to 3 DataSets labeled *train*, *valid* and *test*. An example would be dp.PennTreeBank or dp.Mnist.
 2. dp.DataSet : encapsulates 1 or 2 DataViews accessed through the `inputs` and `targets` methods.
 3. dp.DataView : encapsulates a Tensor or a table thereof. Can reshape and transpose the Tensor from a stored to a desired `shape`.
 4. torch.Tensor : provides a view to a Storage. Where the storage is just an array of contiguous memory, the Tensor can be strided, multi-dimensional. 
 5. torch.Storage : encapsulates a pointer to an array of data (int, long, char, float, etc.). A storage can be shared by multiple Tensors, each having a different view of the data.

## Image Classification ##

In this section, we will build an image classification DataSource for face detection.
Ultimately, we want to use the dataset to train a classifier to discriminate betweene two classes of images : background or face. 
The dataset is provided as a compressed zip file made available by the University of Purdue at : 
[https://engineering.purdue.edu/elab/files/face-dataset.zip](https://engineering.purdue.edu/elab/files/face-dataset.zip).
Once downloaded and uncompressed, we find a directory or PNG images for each class : *bg/* and *face/*.
The images 

### The Easy Way ###

If all we want to do is quickly prepare the data for our personal use, 
then we can create a function that will return a DataSource.

First,  we need to fill an `input` and a `target` Tensor with the data.

```lua
require 'dp'
function facedetection(dataPath)
   local bgPaths = paths.indexdir(paths.concat(dataPath, 'bg'))
   facePaths = pathds.indexDir(paths.concat(dataPath, 'face'))
   
   for i = 
   trData = torch.load('train_data.t7','ascii')
   trLabel = torch.load('train_label.t7','ascii')


   local input_v, target_v = dp.ImageView(), dp.ClassView()
   input_v:forward('bchw', trData)

   targets = trLabel
   targets:add(1)
   targets = targets:type('torch.DoubleTensor')

   --print(trData[1])
   --print(trLabel[1])
   --print(trLabel)
   target_v:forward('bt',trLabel)
   target_v:setClasses({0,1,2,3,4})

   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bchw', 'b')
end
```

### Planning ###

We want to build a DataSource that will auto-magically download the data 
(i.e. the *face-dataset.zip* file) from the Web if not found locally (on the user's hard disk), and uncompress it.
Since we don't have a means of hosting our own version of the dataset on the Web, we will use Purdue's URI.
This implies that our DataSource will need to build `input` and `target` Tensors from the PNG files,
and divide the data into a training and validation set. We omit the test set since it isn't provided.

### Execution ###

The
