------------------------------------------------------------------------
--[[ DataSet ]]--
-- Contains inputs and optional targets. Used for training or testing
-- (evaluating) a model. Inputs/targets are tables of DataTensors, which
-- allows for multi-input / multi-target DataSets.

-- Unsupervised Learning
-- If the DataSet is for unsupervised learning, only inputs need to 
-- be provided.

-- Multi-input/target DataSets
-- Inputs and targets should be provided as instances of 
-- dp.DataTensor to support conversions to other axes formats. 
-- Inputs and targets may also be provided as tables of 
-- dp.DataTensors. This is useful for multi-task learning, or 
-- learning from hints, in the case of multi-targets. In the case 
-- of multi-inputs, images can be combined with tags to provided 
-- for richer inputs, etc. 
-- Multi-inputs/targets are ready to be used with ParallelTable and 
-- ConcatTable nn.Modules.

-- Automatic dp.DataTensor construction
-- If the provided inputs or targets are torch.Tensors, an attempt is 
-- made to convert them to dp.DataTensor using the optionally 
-- provided axes and sizes (inputs), classes (outputs)
------------------------------------------------------------------------
local DataSet, parent = torch.class("dp.DataSet", "dp.BaseSet")
DataSet.isDataSet = true

function DataSet:__init(config)
   local args, which_set, inputs, targets
      = xlua.unpack(
      {config or {}},
      'DataSet', nil,
      {arg='which_set', type='string', req=true,
       help='"train", "valid" or "test" set'},
      {arg='inputs', type='dp.DataTensor | table of dp.DataTensors', 
       help='Sample inputs to a model. These can be DataTensors or '..
       'a table of DataTensors (in which case these are converted '..
       'to a CompositeTensor', req=true},
      {arg='targets', type='dp.DataTensor | table of dp.DataTensors', 
       help='Sample targets to a model. These can be DataTensors or '..
       'a table of DataTensors (in which case these are converted '..
       'to a CompositeTensor. The indices of examples must be '..
       'in both inputs and targets must be aligned.'}
   )
   config.which_set = which_set
   config.inputs = inputs
   config.targets = targets
   parent.__init(self, config)
end

function DataSet:write(...)
   error"DataSet Error: Shouldn't serialize DataSet"
end

--TODO : allow for examples with different weights (probabilities)
--Returns set of probabilities torch.Tensor
function DataSet:probabilities()
   error"NotImplementedError"
   return self._probabilities
end
