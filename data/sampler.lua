--[[TODO]]--
-- Use Random_seed
-- Support multi-input/target datatensors
-- Batch inherits DataSet
-- postpone dataview to model.
-- criterion needs to be decorated for providing dataview
-- dont clone, copy into buffer.
-- model determines sample type
-- batch gets classes from targets.
-- Dataset could be called with sub, index to generate Batch

------------------------------------------------------------------------
--[[ Sampler ]]--
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------
local Sampler = torch.class("dp.Sampler")
Sampler.isSampler = true

function Sampler:__init(...)
   local args, batch_size, data_view, sample_type = xlua.unpack(
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
       'when the first model is a transfer function like Tanh.' },
      {arg='sample_type', type='string', default='double',
       help='"cuda" | "float" | "double"'}
   )
   self:setBatchSize(batch_size)
   if data_view then self:setDataView(data_view) end
   self._sample_type = typeString_to_tensorType(sample_type)
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
function Sampler:sampleEpoch(data, batch)
   batch = batch or dp.Batch()
   local dataset = dp.Sampler.toDataset(data)
   local nSample = dataset:nSample()
   local batch_size = self._batch_size
   local start = 1
   local stop
   -- build iterator
   local epochSamples = 
      function()
         stop = math.min(start+batch_size-1,nSample)
         -- inputs
         batch:inputs():copy(dataset:inputs():sub(start, stop))
         -- targets
         batch:targets():copy(dataset:targets():sub(start, stop))
         batch:setup{
            batch_iter=stop, epoch_size=nSample, 
            batch_size=batch_size, n_sample=stop-start+1,
            grad_type=self._sample_type, indices=torch.range(start,stop)
         }
         start = start + batch_size
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
   -- shuffle before each epoch
   local dataset_indices = torch.randperm(nSample)
   -- build iterator
   local epochSamples = 
      function()
         if start >= nSample then
            return
         end
         stop = math.min(start+batch_size-1,nSample)
         local batch_indices = dataset_indices:sub(start,stop):long()
         -- inputs
         local batch_inputs = {}
         for i=1,#dataset_inputs do
            local dv = data_view[i]
            local dt = dataset_inputs[i]
            batch_inputs[i] = dt[dv](dt):index(
                                    1, batch_indices
                                 ):type(self._sample_type)
         end
         -- targets
         local batch_function = 
            function(datatensor) 
               return datatensor:default():index(
                                    1, batch_indices
                                 ) --:type(self._sample_type)
            end
         local batch_targets = _.map(dataset_targets, batch_function)
         --TODO support multi-input/target datasets
         --Dataset should be called with sub, index to generate Batch
         local batch = dp.Batch{
            inputs=batch_inputs[1], targets=batch_targets[1], 
            batch_iter=stop, epoch_size=nSample, batch_size=batch_size, 
            n_sample=stop-start+1, classes=dataset_targets[1]:classes(),
            grad_type=self._sample_type, indices=batch_indices
         }
         start = start + batch_size
         collectgarbage() 
         return batch
      end
   return epochSamples
end
