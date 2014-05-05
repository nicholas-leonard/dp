--[[TODO]]--
-- Use Random_seed
-- Dataset could be called with sub, index to generate Batch
------------------------------------------------------------------------
--[[ Sampler ]]--
-- DataSet iterator
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------
local Sampler = torch.class("dp.Sampler")
Sampler.isSampler = true

function Sampler:__init(...)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, batch_size = xlua.unpack(
      {... or {}},
      'Sampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'}
   )
   self:setBatchSize(batch_size)
end

function Sampler:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, batch_size, overwrite, mediator = xlua.unpack(
      {config},
      'Sampler:setup', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'},
      {arg='mediator', type='dp.Mediator'}
   )
   if batch_size and (not self._batch_size or overwrite) then
      self:setBatchSize(batch_size)
   end
   self._mediator = mediator
end

function Sampler:setBatchSize(batch_size)
   self._batch_size = batch_size
end

function Sampler:batchSize()
   return self._batch_size
end

function Sampler:report()
   return {batch_size = self._batch_size}
end

--static function. Checks dataset type or gets dataset from datasource
function Sampler.toDataset(dataset)
   if dataset.isDataSource and not self._warning then
      print"Sampler Warning: assuming dataset is DataSource:trainSet()"
      dataset = dataset:trainSet()
      self._warning = true
   end
   assert(dataset.isDataSet, "Error : unsupported dataset type.")
   return dataset
end

--Returns an iterator over samples for one epoch
--Default is to iterate sequentially over all examples
function Sampler:sampleEpoch(dataset, batch)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local start = 1
   local stop
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+self._batch_size-1,nSample)
         -- inputs and targets
         local batch = dataset:sub(start, stop)
         -- metadata
         batch:setup{
            batch_iter=stop, batch_size=self._batch_size,
            n_sample=stop-start+1, grad_type=self._sample_type, 
            indices=torch.range(start,stop)
         }
         start = start + self._batch_size
         if start >= nSample then
            return
         end
         --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
         collectgarbage() 
         return batch
      end
   return epochSamples
end

------------------------------------------------------------------------
--[[ ShuffleSampler ]]--
-- Samples from a multinomial distribution where each examples has a 
-- probability of being samples.
------------------------------------------------------------------------
local ShuffleSampler, parent = torch.class("dp.ShuffleSampler", "dp.Sampler")

function ShuffleSampler:_init(config)
   local args, batch_size, random_seed = xlua.unpack(
      {config or {}},
      'ShuffleSampler', 
      'Samples batches from a shuffled set of examples in dataset. '..
      'Iteration ends after all examples have been sampled once (for one epoch). '..
      'Examples are shuffled at the start of the iteration. ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='random_seed', type='number', req=true,
       help='Used to initialize the shuffle generator.' ..
       'Not yet supported'}
   )
   self:setRandomSeed(random_seed)
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
      self:setRandomSeed(random_seed)
   end
end

function ShuffleSampler:setRandomSeed(random_seed)
   self._random_seed = random_seed
end

function ShuffleSampler:randomSeed()
   return self._random_seed
end
   
function ShuffleSampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local start = 1
   local stop
   -- shuffle before each epoch
   local dataset_indices = torch.randperm(nSample)
   -- build iterator
   local epochSamples = 
      function(batch)
         batch = batch or dataset:batch(self._batch_size)
         stop = math.min(start+self._batch_size-1,nSample)
         local batch_indices = dataset_indices:sub(start,stop):long()
         -- inputs and targets
         dataset:index(batch, batch_indices)
         -- metadata
         batch:setup{
            batch_iter=stop, batch_size=self._batch_size,
            n_sample=stop-start+1, grad_type=self._sample_type, 
            indices=torch.range(start,stop)
         }
         start = start + self._batch_size
         if start >= nSample then
            return
         end
         collectgarbage() 
         return batch
      end
   return epochSamples
end
