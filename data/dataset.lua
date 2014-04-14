------------------------------------------------------------------------
--[[ DataSet ]]--
-- BaseSet
-- Not Serializable
-- Contains inputs and optional targets. Used for training or
-- evaluating a model. Inputs and targets are tables of BaseTensors.

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

function DataSet:write(...)
   error"DataSet Error: Shouldn't serialize DataSet"
end

--TODO : allow for examples with different weights (probabilities)
--Returns set of probabilities torch.Tensor
function DataSet:probabilities()
   error"NotImplementedError"
   return self._probabilities
end

-- builds an empty batch (factory method)
function DataSet:emptyBatch()
   dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=self:inputs():emptyClone(),
      targets=self:targets() and self:targets():emptyClone()
   }
end
