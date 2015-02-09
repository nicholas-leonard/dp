------------------------------------------------------------------------
--[[ RandomSampler ]]--
-- DataSet iterator
-- Randomly samples batches from a dataset.
------------------------------------------------------------------------
local RandomSampler, parent = torch.class("dp.RandomSampler", "dp.Sampler")

function RandomSampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, sample_func = xlua.unpack(
      {config},
      'RandomSampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='sample_func', type='function',
       help='function to sample batches'}
   )
   self._sample_func = sample_func
   parent.__init(self, config)
end

--Returns an iterator over samples for one epoch
--Default is to iterate sequentially over all examples
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
      batch = batch or dataset:batch(self._batch_size)
      -- inputs and targets
      dataset:sample(batch, self._batch_size)
      local indices = batch:indices() or torch.Tensor()
      -- metadata
      batch:setup{
         batch_iter=nSampled, batch_size=self._batch_size,
         n_sample=sself._batch_size
      }
      nSampled = nSampled + self._batch_size
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
      collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end
