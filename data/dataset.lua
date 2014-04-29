------------------------------------------------------------------------
--[[ DataSet ]]--
-- BaseSet subclass
-- Not Serializable
-- Contains inputs and optional targets. Used for training or
-- evaluating a model. Inputs and targets are tables of BaseTensors.

-- Unsupervised Learning :
-- If the DataSet is for unsupervised learning, only inputs need to 
-- be provided.

-- Multiple inputs and outputs :
-- Inputs and targets should be provided as instances of 
-- dp.BaseTensor to support conversions to other axes formats. 
-- Inputs and targets may also be provided as dp.CompositeTensors. 
-- Allowing for multiple targets is useful for multi-task learning
-- or learning from hints. In the case of multiple inputs, 
-- images can be combined with tags to provided for richer inputs, etc. 
-- Multiple inputs and targets are ready to be used with dp.Parallel.
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

-- builds a batch (factory method)
-- doesn't copy the inputs or targets (so don't modify them)
function DataSet:batch(batch_size)
   return dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=self:inputs():sub(1, batch_size),
      targets=self:targets() and self:targets():sub(1, batch_size)
   }
end

function DataSet:sub(start, stop)
   return dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=self:inputs():sub(start, stop),
      targets=self:targets() and self:targets():sub(start, stop)
   }    
end

function DataSet:index(batch, indices)
   if (not batch) or (not indices) then 
      indices = indices or batch
      return dp.Batch{
         which_set=self:whichSet(), epoch_size=self:nSample(),
         inputs=self:inputs():index(indices),
         targets=self:targets() and self:targets():index(indices)
      }
   end
   assert(batch.isBatch, "Expecting dp.Batch at arg 1")
   self:inputs():index(batch:inputs(), indices)
   if self:targets() then
      self:targets():index(batch:targets(), indices)
   end
   return batch
end
