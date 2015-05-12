------------------------------------------------------------------------
--[[ RandomSampler ]]--
-- DataSet iterator
-- Randomly samples batches from a dataset.
------------------------------------------------------------------------
local RandomSampler, parent = torch.class("dp.RandomSampler", "dp.Sampler")

--Returns an iterator over samples for one epoch
function RandomSampler:sampleEpoch(dataset)
   dataset = dp.RandomSampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampled = 0
   local stop
   
   -- build iterator
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      batch = batch or dataset:sample(self._batch_size)
      -- inputs and targets
      dataset:sample(batch, self._batch_size)
      -- metadata
      batch:setup{
         batch_iter=nSampled, batch_size=self._batch_size,
         n_sample=self._batch_size
      }
      batch = self._ppf(batch)
      nSampled = nSampled + self._batch_size
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

-- used with datasets that support asynchronous iterators like ImageClassSet
function RandomSampler:sampleEpochAsync(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampledPut = 0
   local nSampledGet = 0
     
   -- build iterator
   local sampleBatch = function(batch, putOnly)
      if nSampledGet >= epochSize then
         return
      end
      
      if nSampledPut < epochSize then
         -- up values
         local uvbatchsize = self._batch_size
         local uvstart = self._start
         dataset:sampleAsyncPut(batch, self._batch_size, nil,
            function(batch) 
               local indices = batch:indices() or torch.Tensor()
               -- metadata
               batch:setup{batch_iter=uvstop, batch_size=batch:nSample()}
               batch = self._ppf(batch)
            end)
         
         nSampledPut = nSampledPut + self._batch_size
         self._start = self._start + self._batch_size
         if self._start >= nSample then
            self._start = 1
         end
      end
      
      if not putOnly then
         batch = dataset:asyncGet()
         nSampledGet = nSampledGet + self._batch_size
         self:collectgarbage() 
         return batch, math.min(nSampledGet, epochSize), epochSize
      end
   end
   
   assert(dataset.isAsync, "expecting asynchronous dataset")
   -- empty the async queue
   dataset:synchronize()
   -- fill task queue with some batch requests
   for tidx=1,dataset.nThread do
      sampleBatch(nil, true)
   end
   
   return sampleBatch
end
