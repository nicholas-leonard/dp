require 'torch'
require 'utils'

--[[TODO]]--
--Random_seed


------------------------------------------------------------------------
--[[ Sampler ]]--
-- Base Class
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------


local Sampler = torch.class("dp.Sampler")

function Sampler:__init(...)
   local args, batch_size = xlua.unpack(
      {... or {}},
      'Sampler', 
      'Abstract class. ' ..
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'}
   )
   self:setBatchSize(batch_size)
end

function Sampler:setup(...)
   local args, dataset, batch_size, xlua.unpack(
      {... or {}},
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
   if batch_size and (not self.batchSize() or overwrite) then
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

--static function. Checks dataset type or gets dataset from datasource
function Sampler.toDataset(dataset)
   if dataset.isDataSource and not self._warning then
      print"Sampler Warning: assuming dataset is DataSource:trainSet()"
      dataset = dataset:trainSet()
      self._warning = true
   end
   assert(dataset.isDataSet, "Error : unsupported dataset type.")
end

--Returns an iterator over samples for one epoch
--Default is to iterate sequentially over all examples
function Sampler:sampleEpoch(data)
   local dataset = Sample.toDataset(data)
   local nSample = dataset:nSample()
   local start = 1
   local stop
   local dataset_inputs = dataset:inputs()
   local dataset_targets = dataset:targets()
   local batch_inputs, batch_targets, batch_function
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+batch_size,nSample)
         batch_function = 
            function(datatensor) 
               return datatensor:sub(start, stop)
            end
         batch_inputs = _.map(dataset_inputs, batch_function)
         batch_targets = _.map(dataset_targets, batch_function)
         start = start + batch_size
         if start >= nSample then
            return
         end
         return inputs, targets
      end
   return epochSamples
end


------------------------------------------------------------------------
--[[ ShuffleSampler ]]--
-- Samples from a multinomial distribution where each examples has a 
-- probability of being samples.
------------------------------------------------------------------------

local ShuffleSampler 
   = torch.class("dp.ShuffleSampler", "dp.Sampler")

function ShuffleSampler:_init(...)
   local args, batch_size, random_seed = xlua.unpack(
      {... or {}},
      'ShuffleSampler', 
      'Samples batches from a shuffled set of examples in dataset. '..
      'Iteration ends after all examples have been sampled once (for one epoch). '..
      'Examples are shuffled at the start of the iteration. ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='random_seed', type='number',
       help='Used to initialize the shuffle generator.' ..
       'Not yet supported'}
   )
   Sampler.__init(self, ...)
end

function ShuffleSampler:setRandomSeed(random_seed)
   self._random_seed = random_seed
end

function ShuffleSampler:randomSeed()
   return self._random_seed
end
   
function ShuffleSampler:sampleEpoch(dataset)
   local dataset = Sample.toDataset(dataset)
   local nSample = dataset:nSample()
   local start = 1
   local stop
   local dataset_inputs = dataset:inputs()
   local dataset_targets = dataset:targets()
   local batch_inputs, batch_targets, batch_indices, batch_function
   -- shuffle before each epoch
   local dataset_indices = torch.randperm(nSample)
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+batch_size,nSample)
         batch_indices = dataset_indices:sub(start,stop):long()
         batch_function = 
            function(datatensor) 
               return datatensor:index(indices)
            end
         batch_inputs = _.map(dataset_inputs, batch_function)
         batch_targets = _.map(dataset_targets, batch_function)
         start = start + batch_size
         if start >= nSample then
            return
         end
         return inputs, targets
      end
   return epochSamples
end

------------------------------------------------------------------------
--[[ FocusSampler ]]--
-- Samples from a multinomial distribution where each examples has a 
-- sampling probability proportional to its error.
-- TODO : is also an Observer or is also a Strategy
-- if an observer, gets report and subject(criteria or batch) from 
-- doneBatch mediator channel. If Strategy, is called with measureError
-- on inputs, targets by propagator, or Cost (criteria Adapter).
------------------------------------------------------------------------

local FocusSampler = torch.class("dp.FocusSampler", "dp.Sampler")
FocusSampler.isObserver = true

function FocusSampler:__init(...)
   error("Error Not Implemented")
   local args, batch_size, random_seed, prob_dist 
      = xlua.unpack(
         {... or {}},
         'FocusSampler', 
         'Samples batches from a shuffled set of examples in DataSet. '..
         'Iteration ends after all examples have been sampled once (for one epoch). '..
         'Examples are shuffled at the start of the iteration. ',
         {arg='batch_size', type='number', default='512',
          help='Number of examples per sampled batches'},
         {arg='random_seed', type='number',
          help='Used to initialize the shuffle generator.' ..
          'Not yet supported'},
         {arg='prob_dist', type='torch.Tensor', 
          help='initial vector of probabilities : one per example.' ..
          'defaults to a uniform probability distribution'},
         {arg='uniform_prob', type='number', default=0.1,
          help='probability of sampling from a uniform distribution'}
   )
end

function FocusSampler:doneEpoch(report)
   batch_errors = self._mediator:optimizer():criterion().batchOutputs
   self:updateMultinomial(batch_errors)
end

function FocusSampler:updateMultinomial(batch_errors)
   
end

function FocusSampler:sampleEpoch()
   
   self._last_indices = last_indices
end


