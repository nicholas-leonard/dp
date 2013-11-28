--[[TODO]]--
-- Use Random_seed
-- Support multi-input/target datatensors
-- Batch inherits DataSet


------------------------------------------------------------------------
--[[ Sampler ]]--
-- Base Class
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------


local Sampler = torch.class("dp.Sampler")

function Sampler:__init(...)
   local args, batch_size, data_view = xlua.unpack(
      {... or {}},
      'Sampler', 
      'Abstract class. ' ..
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='data_view', type='string | table',
       help='Used to determine which view of the input DataTensors' ..
       'to use. This is automatically setup using Model in ' ..
       'Propagator, but can be manually entered for cases where ' ..
       'it cannot be determined by the model. For example, '..
       'when the first model is a transfer function like Tanh.' }
   )
   self:setBatchSize(batch_size)
   if data_view then self:setDataView(data_view) end
end

function Sampler:setup(...)
   local args, batch_size, overwrite, mediator, model
      = xlua.unpack(
      {... or {}},
      'Sampler:setup', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='64',
       help='Number of examples per sampled batches'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'},
      {arg='mediator', type='dp.Mediator'},
      {arg='model', type='dp.Model'}
   )
   if batch_size and (not self._batch_size or overwrite) then
      self:setBatchSize(batch_size)
   end
   self._mediator = mediator
   assert(model or self._data_view)
   if model and not self._data_view then
      self:setDataView(model:dataView())
   end
end

function Sampler:setBatchSize(batch_size)
   self._batch_size = batch_size
end

function Sampler:setDataView(data_view)
   if type(data_view) == 'string' then
      data_view = {data_view}
   end
   self._data_view = data_view
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
function Sampler:sampleEpoch(data)
   local dataset = dp.Sampler.toDataset(data)
   local nSample = dataset:nSample()
   local batch_size = self._batch_size
   local data_view = self._data_view
   local start = 1
   local stop
   local dataset_inputs = dataset:inputs()
   local dataset_targets = dataset:targets()
   local batch_inputs, batch_targets, batch_function, batch
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+batch_size-1,nSample)
         -- inputs
         batch_inputs = {}
         for i=1,#dataset_inputs do
            local dv = data_view[i]
            local dt = dataset_inputs[i]
            -- we clone these so that they are not serialized
            -- TODO : model:forwardActivate instead of ref.
            batch_inputs[i] = dt[dv](dt):sub(start, stop):clone()
         end
         -- targets
         batch_function = 
            function(datatensor) 
               return datatensor:default():sub(start, stop):clone()
            end
         batch_targets = _.map(dataset_targets, batch_function)
         --TODO support multi-input/target datasets
         --Dataset should be called with sub, index to generate Batch
         batch = dp.Batch{inputs=batch_inputs[1], targets=batch_targets[1], 
                       batch_iter=stop, epoch_size=nSample, 
                       batch_size=batch_size, n_sample=stop-start,
                       classes=dataset_targets[1]:classes()}
         start = start + batch_size
         if start >= nSample then
            return
         end
         return batch
      end
   return epochSamples
end


------------------------------------------------------------------------
--[[ ShuffleSampler ]]--
-- Samples from a multinomial distribution where each examples has a 
-- probability of being samples.
------------------------------------------------------------------------

local ShuffleSampler, parent
   = torch.class("dp.ShuffleSampler", "dp.Sampler")

function ShuffleSampler:_init(config)
   local args, batch_size, random_seed = xlua.unpack(
      {config or {}},
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
   self:setRandomSeed(random_seed)
   config.batch_size = batch_size
   Sampler.__init(self, config)
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
   local dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local batch_size = self._batch_size
   local start = 1
   local stop
   local data_view = self._data_view
   local dataset_inputs = dataset:inputs()
   local dataset_targets = dataset:targets()
   local batch_inputs, batch_targets, batch_indices, batch_function
   local batch
   -- shuffle before each epoch
   local dataset_indices = torch.randperm(nSample)
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+batch_size-1,nSample)
         batch_indices = dataset_indices:sub(start,stop):long()
         -- inputs
         batch_inputs = {}
         for i=1,#dataset_inputs do
            local dv = data_view[i]
            local dt = dataset_inputs[i]
            batch_inputs[i] = dt[dv](dt):index(1, batch_indices)
         end
         -- targets
         batch_function = 
            function(datatensor) 
               return datatensor:default():index(1, batch_indices)
            end
         batch_targets = _.map(dataset_targets, batch_function)
         --TODO support multi-input/target datasets
         --Dataset should be called with sub, index to generate Batch
         batch = dp.Batch{inputs=batch_inputs[1], targets=batch_targets[1], 
                       batch_iter=stop, epoch_size=nSample, 
                       batch_size=batch_size, n_sample=stop-start,
                       classes=dataset_targets[1]:classes()}
         start = start + batch_size
         if start >= nSample then
            return
         end
         return batch
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


