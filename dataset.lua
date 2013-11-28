require 'torch'


--[[
TODO
Create SIST, MIST, SIMT, and MIMT DataSets
Iterators (shuffled, etc)
Eliminate non-standard dependencies
Support multi-input, and target preprocessing
Primary targets are the first ones (so what??)
]]--

------------------------------------------------------------------------
--[[ DataSet ]]--
-- Contains inputs and optional targets. Used for training or testing
-- (evaluating) a model. Inputs/targets are tables of DataTensors, which
-- allows for multi-input / multi-target DataSets. This class can be 
-- inherited to create specific DataSets like MNIST, SVHN, Newsgroups20,
-- CIFAR-100, etc.
------------------------------------------------------------------------

local DataSet = torch.class("dp.DataSet")

DataSet.isDataSet = true

function DataSet:__init(...)
   local args, which_set, inputs, targets, axes, sizes, classes
      = xlua.unpack(
      {...},
      'DataSet', 
      [[Constructs a DataSet.
      ### Unsupervised Learning
      If the DataSet is for unsupervised learning, only inputs need to 
      be provided.
      ### Multi-input/target DataSets
      Inputs and targets should be provided as instances of 
      dp.DataTensor to support conversions to other axes formats. 
      Inputs and targets may also be provided as tables of 
      dp.DataTensors. This is useful for multi-task learning, or 
      learning from hints, in the case of multi-targets. In the case 
      of multi-inputs, images can be combined with tags to provided 
      for richer inputs, etc. 
      Multi-inputs/targets are ready to be used with ParallelTable and 
      ConcatTable nn.Modules.
      ### Automatic dp.DataTensor construction
      If the provided inputs or targets are torch.Tensors, an attempt is 
      made to convert them to dp.DataTensor using the optionally 
      provided axes and sizes (inputs), classes (outputs) ]],
      {arg='which_set', type='string', req=true,
       help=[['train', 'valid' or 'test' set]]},
      {arg='inputs', type='dp.DataTensor | torch.Tensor | table', 
       help=[[Inputs of the DataSet taking the form of torch.Tensor with
            2 dimensions, or more if topological is true. Alternatively,
            inputs may take the form of a table of such torch.Tensors.
            The first dimension of the torch.Tensor(s) should be of size
            number of examples.
            ]], req=true},
      {arg='targets', type='dp.DataTensor | torch.Tensor | table', 
       help=[[Targets of the DataSet taking the form of torch.Tensor
            with 1-2 dimensions. Alternatively, targets may take the 
            form of a table of such torch.Tensors. The first dimension 
            of the torch.Tensor(s) should be of size number of examples.
            ]], default=nil},
      {arg='axes', type='table', 
       help=[[Optional. Used when supplied inputs is a torch.Tensor, in
            order to convert it to dp.DataTensor. In which case, it is 
            a table defining the order and nature of each dimension
            of a tensor. Two common examples would be the archtypical 
            MLP input : {'b', 'f'}, or a common image representation : 
            {'b', 'h', 'w', 'c'}. 
            Possible axis symbols are :
            1. Standard Axes:
              'b' : Batch/Example
              'f' : Feature
              't' : Class
            2. Image Axes
              'c' : Color/Channel
              'h' : Height
              'w' : Width
              'd' : Dept
            Defaults to the dp.DataTensor default.]]},
      {arg='sizes', type='table | torch.LongTensor', 
       help=[[Optional. Used when supplied inputs is a torch.Tensor, in
            order to convert it to dp.DataTensor. In which case, it is
            a table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions in axes. This should be supplied 
            if the dimensions of the data is different from the number
            of elements in the axes table, in which case it will be used
            to : data:resize(sizes). 
            Defaults to the dp.DataTensor default.]]}
   )
   self:setWhichSet(which_set)
   self:setInputs(inputs, axes, size)  
   self:setTargets(targets, classes)
   --self:shuffle()
end

function DataSet:setWhichSet(which_set)
   self._which_set = which_set
end

function DataSet:whichSet()
   return self.which_set
end

function DataSet:isTrain()
   return (self.which_set == 'train')
end

function DataSet:setInputs(inputs)
   self._inputs = inputs
end

function DataSet:appendInput(input)
   table.insert(self._inputs, input)
end

function DataSet:appendTarget(target)
   table.insert(self._targets, target)
end

function DataSet:extendTargets(targets)
   error("NotImplementedError")
end

function DataSet:write(...)
   error"DataSet Error: Shouldn't serialize DataSet"
end

--TODO: accept list of axes and sizes
function DataSet:setInputs(inputs, axes, sizes)
   if type(inputs) ~= 'table' or inputs.isDataTensor then
      if torch.Tensor.isInstance(inputs) then
         --convert torch.Tensor to dp.DataTensor
         inputs = dp.DataTensor{data=inputs, axes=axes, sizes=sizes}
      end
      assert(inputs.isDataTensor,
         "Error : invalid inputs. Expecting type dp.DataTensor")
      -- encapsulate inputs in a table
      inputs = {inputs}
   else
      dp.DataTensor.assertInstances(inputs)
   end
   self._inputs = inputs
end

function DataSet:setTargets(targets, classes)
   if type(targets) ~= 'table' or targets.isDataTensor then
      if torch.Tensor.isInstance(targets) then
         --convert torch.Tensor to dp.DataTensor
         if classes then
            targets = dp.ClassTensor{data=targets, classes=classes}
         else
            targets = dp.DataTensor{data=targets}
         end
      end
      assert(targets.isDataTensor,
         "Error : invalid targets. Expecting type dp.DataTensor")
      -- encapsulate inputs in a table
      targets = {targets}
   else
      dp.DataTensor.assertInstances(targets)
   end
   self._targets = targets
end


-- Returns the number of samples in the DataSet.
function DataSet:nSample()
   --TODO what to do when DataTensors are of different size?
   return self._inputs[1]:nSample()
end

--Returns set of input dp.DataTensors
function DataSet:inputs()
   return self._inputs
end

--Returns set of target dp.DataTensors
function DataSet:targets()
   return self._targets
end

--TODO : allow for examples with different weights (probabilities)
--Returns set of probabilities torch.Tensor
function DataSet:probabilities()
   error"NotImplementedError"
   return self._probabilities
end

--Preprocesses are applied to DataTensors, which means that 
--DataTensor:image(), :expandedAxes(), etc. can be used.
function DataSet:preprocess(...)
   local args, input_preprocess, target_preprocess, can_fit
      = xlua.unpack(
         {... or {}},
         'DataSet:preprocess',
         'Preprocesses the DataSet.',
         {arg='input_preprocess', type='dp.Preprocess', 
          help='Preprocess applied to the input DataTensor(s) of ' .. 
          'the DataSet'},
         {arg='target_preprocess', type='dp.Preprocess',
          help='Preprocess applied to the target DataTensor(s) of ' ..
          'the DataSet'},
         {arg='can_fit', type='boolean',
          help='Allows measuring of statistics on the DataTensor(s) ' .. 
          'of DataSet to initialize the preprocess. Should normally ' .. 
          'only be done on the training set. Default is to fit the ' ..
          'training set.'}
   )
   assert(input_preprocess or target_preprocess, 
      "Error: no preprocess (neither input or target) provided)")
   if can_fit == nil then
      can_fit = self:isTrain()
   end
   --TODO support multi-input/target preprocessing
   if input_preprocess and input_preprocess.isPreprocess then
      input_preprocess:apply(self._inputs[1], can_fit)
   end
   if target_preprocess and target_preprocess.isPreprocess then
      target_preprocess:apply(self._targets[1], can_fit)
   end
end


