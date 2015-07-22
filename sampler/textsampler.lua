------------------------------------------------------------------------
--[[ TextSampler ]]--
-- Used for training recurrent modules
------------------------------------------------------------------------
local TextSampler, parent = torch.class("dp.TextSampler", "dp.Sampler")

function TextSampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, context_size = xlua.unpack(
      {config},
      'Sampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='context_size', type='number', req=true,
       help='Size of each sequence (context)'},
   )
   self._context_size = context_size
   parent.__init(self, config)
end

function TextSampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local sliceSize = math.floor(nSample/self._batch_size)
   nSample = sliceSize*self._batch_size
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampled = 0
   -- shuffle before each epoch
   local batch_indices = torch.range(1,self._batch_size):long()
   batch_indices:add(-1):mul(sliceSize):add(1)
   -- build iterator
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      batch = batch or dataset:batch(self._batch_size)
      -- inputs and targets
      dataset:index(batch, batch_indices)
      -- metadata
      batch:setup{
         batch_iter=stop, batch_size=self._batch_size,
         n_sample=self._batch_size
      }
      batch = self._ppf(batch)
      nSampled = nSampled + (self._batch_size*self._context_size)
      -- move cursors to next context
      batch_indices:add(self._context_size)
      if batch_indices:max() >= nSample then
         batch_indices = torch.range(1,self._batch_size):long()
         batch_indices:add(-1):mul(sliceSize):add(1)
      end
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end
