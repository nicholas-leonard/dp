------------------------------------------------------------------------
--[[ ShuffleSampler ]]--
-- Iterates over examples in a dataset by shuffling the example 
-- indices before each epoch.
------------------------------------------------------------------------
local ShuffleSampler, parent = torch.class("dp.ShuffleSampler", "dp.Sampler")

function ShuffleSampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, batch_size, random_seed = xlua.unpack(
      {config},
      'ShuffleSampler', 
      'Samples batches from a shuffled set of examples in dataset. '..
      'Iteration ends after all examples have been sampled once (for one epoch). '..
      'Examples are shuffled at the start of the iteration. ',
      {arg='batch_size', type='number', default=128,
       help='Number of examples per sampled batches'},
      {arg='random_seed', type='number', default=777,
       help='Used to initialize the shuffle generator.'}
   )
   self:randomSeed(random_seed)
   config.batch_size = batch_size
   parent.__init(self, config)
end

function ShuffleSampler:setup(config)
   config = config or {}
   local args, random_seed, overwrite = xlua.unpack(
      {config},
      'ShuffleSampler:setup', nil,
      {arg='random_seed', type='number',
       help='Used to initialize the shuffle generator.' ..
       'Not yet supported'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'}
   )
   config.overwrite = overwrite
   parent.setup(self, config)
   if random_seed and ((not self._random_seed) or overwrite) then
      self:randomSeed(random_seed)
   end
end

function ShuffleSampler:randomSeed(seed)
   if seed then
      self._random_seed = seed
      torch.manualSeed(seed)
   end
   return self._random_seed
end
   
function ShuffleSampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampled = 0
   -- shuffle before each epoch
   local dataset_indices = torch.randperm(nSample):long()
   -- build iterator
   return function(batch)
      if nSampled >= epochSize then
         return
      end
      stop = math.min(self._start+self._batch_size-1,nSample)
      batch = batch or dataset:batch(stop-self._start+1)
      local batch_indices = dataset_indices:sub(self._start,stop)
      -- inputs and targets
      dataset:index(batch, batch_indices)
      local indices = batch:indices() or torch.Tensor()
      -- metadata
      batch:setup{
         batch_iter=stop, batch_size=self._batch_size,
         n_sample=stop-self._start+1, 
         indices=indices:range(self._start,stop)
      }
      batch = self._ppf(batch)
      nSampled = nSampled + stop - self._start + 1
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
         dataset_indices = torch.randperm(nSample):long()
      end
      self:collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end
