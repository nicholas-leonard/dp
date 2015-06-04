------------------------------------------------------------------------
--[[ Sampler ]]--
-- DataSet iterator
-- Sequentially samples batches from a dataset.
------------------------------------------------------------------------
local Sampler = torch.class("dp.Sampler")
Sampler.isSampler = true

function Sampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, batch_size, epoch_size, ppf, gc_freq = xlua.unpack(
      {config},
      'Sampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default=128,
       help='Number of examples per sampled batches'},
      {arg='epoch_size', type='number', default=-1,
       help='Number of examples presented per epoch. '..
       'Default is to use then entire dataset per epoch'},
      {arg='ppf', type='function', 
       help='a function that preprocesses a Batch into another Batch'},
      {arg='gc_freq', type='number', default=50,
       help='collectgarbage() every gc_freq batches'}
   )
   self._ppf = ppf or function(batch) return batch end
   self._gc_freq = gc_freq
   self:setBatchSize(batch_size)
   self._epoch_size = epoch_size
   self._gc_n_batch = 0
   if epoch_size > 0 then
      if batch_size > epoch_size then
         error("positive epoch_size should be greater than batch_size", 2)
      end
   else
      self._epoch_size = nil
   end
end

function Sampler:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, batch_size, overwrite, mediator = xlua.unpack(
      {config},
      'Sampler:setup', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default=128,
       help='Number of examples per sampled batches'},
      {arg='overwrite', type='boolean', default=false,
       help='overwrite existing values if not nil.' .. 
       'If nil, initialize whatever the value of overwrite.'},
      {arg='mediator', type='dp.Mediator',
       help='used for communication between objects'}
   )
   if batch_size and (not self._batch_size or overwrite) then
      self:setBatchSize(batch_size)
   end
   self._mediator = mediator
end

function Sampler:setBatchSize(batch_size)
   if torch.type(batch_size) ~= 'number' or batch_size < 1 then
      error("Expecting positive batch_size")
   end
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
   if dataset.isDataSource then
      -- assumes dataset is the DataSource's training set
      dataset = dataset:trainSet()
      self._warning = true
   elseif dataset.isView then
      -- assumes dataset is a set of inputs in training set
      dataset = dp.DataSet{which_set='train', inputs=dataset}
   end
   assert(dataset.isDataSet, "Error : unsupported dataset type.")
   return dataset
end

function Sampler:collectgarbage()
   self._gc_n_batch = self._gc_n_batch + 1
   if self._gc_n_batch >= self._gc_freq then
      --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
      collectgarbage()
      self._gc_n_batch = 0
   end
end

-- Returns an iterator over samples for one epoch
-- Default is to iterate sequentially over all examples
function Sampler:sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
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
      stop = math.min(self._start+self._batch_size-1,nSample)
      batch = batch or dataset:batch(stop-self._start+1)
      -- inputs and targets
      dataset:sub(batch, self._start, stop)
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
      end
      self:collectgarbage()
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

-- used with datasets that support asynchronous iterators like ImageClassSet
function Sampler:sampleEpochAsync(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   local epochSize = self._epoch_size or nSample
   self._start = self._start or 1
   local nSampledPut = 0
   local nSampledGet = 0
   local stop
     
   -- build iterator
   local sampleBatch = function(batch, putOnly)
      if nSampledGet >= epochSize then
         return
      end
      
      if nSampledPut < epochSize then
         stop = math.min(self._start+self._batch_size-1,nSample)
         -- up values
         local uvstop = stop
         local uvbatchsize = self._batch_size
         local uvstart = self._start
         dataset:subAsyncPut(batch, self._start, stop,
            function(batch) 
               -- metadata
               batch:setup{batch_iter=uvstop, batch_size=batch:nSample()}
               batch = self._ppf(batch)
            end)
         
         nSampledPut = nSampledPut + stop - self._start + 1
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

function Sampler:async()
   self.sampleEpoch = self.sampleEpochAsync
end
