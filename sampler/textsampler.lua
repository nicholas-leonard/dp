------------------------------------------------------------------------
--[[ TextSampler ]]--
-- Used for training recurrent modules on TextSets
------------------------------------------------------------------------
local TextSampler, parent = torch.class("dp.TextSampler", "dp.Sampler")

function TextSampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   assert(torch.isTypeOf(dataset, 'dp.TextSet'), "Expecting TextSet dataset")
   local nSample = dataset:data():nElement()
   local sliceSize = math.floor(nSample/self._batch_size)
   nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   local nSampled = 0
   local nStep = 0
   
   -- shuffle before each epoch
   local batch_indices = torch.range(1,self._batch_size):long()
   batch_indices:add(-1):mul(sliceSize):add(1)
   local contextSize = dataset:contextSize()
   
   -- build iterator
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      nStep = math.min(sliceSize, nStep + contextSize)
      batch = batch or dataset:batch(self._batch_size)
      
      if batch_indices:max() > nSample then
         -- try removing the last row :
         batch_indices = batch_indices:narrow(1,1,batch_indices:size(1)-1)
         if batch_indices:max() > nSample then
            -- if that doesn't work (because it worked for the prev batch)
            -- then reset the indices to their starting position
            batch_indices = torch.range(1,self._batch_size):long()
            batch_indices:add(-1):mul(sliceSize):add(1)
         end
      end

      -- inputs and targets
      local cs = nStep % contextSize
      cs = (cs == 0) and contextSize or cs
      dataset:contextSize(cs)
      dataset:index(batch, batch_indices)
      dataset:contextSize(contextSize)
      -- metadata
      batch:setup{
         batch_iter=nStep, batch_size=self._batch_size,
         n_sample=batch_indices:size(1)
      }
      batch = self._ppf(batch)
      nSampled = nSampled + (batch_indices:size(1)*cs)
      -- move cursors to next context
      batch_indices:add(contextSize)
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end
