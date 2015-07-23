------------------------------------------------------------------------
--[[ TextSampler ]]--
-- Used for training recurrent modules on TextSets
------------------------------------------------------------------------
local TextSampler, parent = torch.class("dp.TextSampler", "dp.Sampler")

function TextSampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   assert(torch.isTypeOf(dataset, 'dp.TextSet'), "Expecting TextSet dataset")
   local nSample = dataset:textSize()
   local sliceSize = math.floor(nSample/self._batch_size)
   local nIndex = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   local nSampled = 0
   
   local batch_indices = torch.LongTensor()
   local contextSize = dataset:contextSize()
   
   self._start = self._start or 1
   -- build iterator
   return function(batch)
      if self._start >= sliceSize then
         self._start = 1
         if not self._epoch_size then
            return
         end
      end
      if nSampled >= epochSize then
         return
      end
      local stop = math.min(math.min(self._start+contextSize-1,sliceSize), nIndex+contextSize-dataset:offset())
      batch = batch or dataset:batch(self._batch_size)
      batch_indices:range(0,self._batch_size-1):mul(sliceSize):add(self._start)
      
      local contextSize_ = stop-self._start+1
      if batch_indices:max() > nIndex+contextSize-contextSize_ then
         -- remove the last row
         batch_indices = batch_indices:narrow(1,1,batch_indices:size(1)-1)
      end
      
      dataset:contextSize(contextSize_)
      dataset:index(batch, batch_indices)
      dataset:contextSize(contextSize)
      
      batch:setup{
         batch_iter=stop, batch_size=self._batch_size,
         n_sample=batch_indices:size(1)
      }
      batch = self._ppf(batch)
      
      nSampled = nSampled + batch_indices:size(1)*contextSize_
      
      self._start = math.max(math.min(self._start + contextSize, nIndex), stop+1)
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end
