require 'torch'
require 'utils'

--[[TODO]]--
--Random_seed

------------------------------------------------------------------------
--[[ Batch ]]--
-- A batch of examples sampled from a dataset.
------------------------------------------------------------------------

local Batch = torch.class("dp.Batch")

function Batch:__init(...)
   local args, inputs, targets
      = xlua.unpack(
      'Batch', nil,
      {arg='inputs', type='torch.Tensor', req=true,
       help='batch of inputs'},
      {arg='targets', type='torch.Tensor',
       help='batch of targets'}
   )
   self:setInputs(inputs)
   self:setTargets(targets)
end

function Batch:setInputs(inputs)
   self._inputs = inputs
end

function Batch:inputs()
   return self._inputs
end

function Batch:setTargets(targets)
   self._targets = targets
end

function Batch:targets()
   return self._targets
end

------------------------------------------------------------------------
--[[ Sampler ]]--
-- Base Class
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------


local Sampler = torch.class("dp.Sampler")

function Sampler:__init(...)
   local args, dataset, batch_size = xlua.unpack(
      {... or {}},
      'Sampler', 
      'Abstract class. ' ..
      'Samples batches from a set of examples in dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='dataset', type='dp.DataSet | dp.DataSource',
       help='dataset from which to sample batches of examples'},
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'}
   )
   if dataset then
      self:setDataset(dataset)
   end
   self:setBatchSize(batch_size)
end

function Sampler:setup(...)
   local args, dataset, batch_size, experiment = xlua.unpack(
      {... or {}},
      'Sampler:setup', 
      'Samples batches from a set of examples in dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='dataset', type='dp.DataSet | dp.DataSource',
       help='dataset from which to sample batches of examples'},
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'},
      {arg='experiment', type='dp.Experiment',
       help='Acts as a Mediator (design pattern). ' ..
       'Provides access to the experiment.'}
   )
   if dataset and (not self.dataset() or overwrite) then
      self:setDataset(dataset)
   end
   if batch_size and (not self.batchSize() or overwrite) then
      self:setBatchSize(batch_size)
   end
   if experiment then
      self.setExperiment(experiment)
   end
end

function Sampler:setDataset(dataset)
   if dataset.isDataSource then
      print"Sampler Warning: assuming dataset is DataSource:trainSet()"
      dataset = dataset:trainSet()
   end
   assert(dataset.isDataSet, "Error : unsupported dataset type.")
   self._dataset = dataset
end

function Sampler:dataset()
   return self._dataset
end

function Sampler:setBatchSize(batch_size)
   self._batch_size = batch_size
end

function Sampler:batchSize()
   return self._batch_size
end

function Sampler:setExperiment(experiment)
   self._experiment = experiment
end

function Sampler:experiment()
   return self._experiment
end

--Returns an iterator over samples for one epoch
--Default is to iterate sequentially over all examples
function Sampler:sampleEpoch(dataset)
   local dataset = dataset or self._dataset
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
   local args, dataset, batch_size, random_seed = xlua.unpack(
      {... or {}},
      'ShuffleSampler', 
      'Samples batches from a shuffled set of examples in dataset. '..
      'Iteration ends after all examples have been sampled once (for one epoch). '..
      'Examples are shuffled at the start of the iteration. ',
      {arg='dataset', type='dp.DataSet', req=true,
       help='dataset from which to sample batches of examples'},
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='random_seed', type='number',
       help='Used to initialize the shuffle generator.' ..
       'Not yet supported'}
   )
   parent.__init(self, args)
end

function ShuffleSampler:setRandomSeed(random_seed)
   self._random_seed = random_seed
end

function ShuffleSampler:randomSeed()
   return self._random_seed
end
   
function ShuffleSampler:sampleEpoch(dataset)
   local dataset = dataset or self._dataset
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
------------------------------------------------------------------------

local FocusSampler = torch.class("dp.FocusSampler", "dp.Sampler")
   
FocusSampler.isEpochObserver = true

function FocusSampler:__init(...)
   error("Error Not Implemented")
   local args, dataset, batch_size, random_seed, prob_dist 
      = xlua.unpack(
         {... or {}},
         'FocusSampler', 
         'Samples batches from a shuffled set of examples in DataSet. '..
         'Iteration ends after all examples have been sampled once (for one epoch). '..
         'Examples are shuffled at the start of the iteration. ',
         {arg='dataset', type='dp.DataSet',
          help='dataset from which to sample batches of examples'},
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

function FocusSampler:onEpoch(mediator)
   batch_errors = mediator:optimizer():criterion().batchOutputs
   self:updateMultinomial(batch_errors)
end

function FocusSampler:updateMultinomial(batch_errors)
   
end

function FocusSampler:sampleEpoch()
   
   self._last_indices = last_indices
end


