------------------------------------------------------------------------
--[[ DataSet ]]--
-- BaseSet subclass
-- Not Serializable
-- Contains inputs and optional targets. Used for training or
-- evaluating a model. Inputs and targets are tables of Views.

-- Unsupervised Learning :
-- If the DataSet is for unsupervised learning, only inputs need to 
-- be provided.

-- Multiple inputs and outputs :
-- Inputs and targets should be provided as instances of 
-- dp.View to support conversions to other axes formats. 
-- Inputs and targets may also be provided as dp.ListTensors. 
-- Allowing for multiple targets is useful for multi-task learning
-- or learning from hints. In the case of multiple inputs, 
-- images can be combined with tags to provided for richer inputs, etc. 
------------------------------------------------------------------------
local DataSet, parent = torch.class("dp.DataSet", "dp.BaseSet")
DataSet.isDataSet = true

-- builds a batch (factory method)
-- reuses the inputs and targets (so don't modify them)
function DataSet:batch(batch_size)
   return self:sub(1, batch_size)
end

function DataSet:sub(batch, start, stop)
   if (not batch) or (not stop) then 
      if batch then
         stop = start
         start = batch
      end
      return dp.Batch{
         which_set=self:whichSet(), epoch_size=self:nSample(),
         inputs=self:inputs():sub(start, stop),
         targets=self:targets() and self:targets():sub(start, stop),
         carry=self:carry() and self:carry():sub(start, stop)
      }    
   end
   assert(batch.isBatch, "Expecting dp.Batch at arg 1")
   self:inputs():sub(batch:inputs(), start, stop)
   if self:targets() then
      self:targets():sub(batch:targets(), start, stop)
   end
   self:carry():sub(batch:carry(), start, stop)
   return batch  
end

function DataSet:index(batch, indices)
   if (not batch) or (not indices) then 
      indices = indices or batch
      return dp.Batch{
         which_set=self:whichSet(), epoch_size=self:nSample(),
         inputs=self:inputs():index(indices),
         targets=self:targets() and self:targets():index(indices),
         carry=self:carry() and self:carry():index(indices)
      }
   end
   assert(batch.isBatch, "Expecting dp.Batch at arg 1")
   self:inputs():index(batch:inputs(), indices)
   if self:targets() then
      self:targets():index(batch:targets(), indices)
   end
   self:carry():index(batch:carry(), indices)
   return batch
end
