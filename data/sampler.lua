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
   local args, batch_size, epoch_size = xlua.unpack(
      {config},
      'Sampler', 
      'Samples batches from a set of examples in a dataset. '..
      'Iteration ends after an epoch (sampler-dependent) ',
      {arg='batch_size', type='number', default='1024',
       help='Number of examples per sampled batches'},
      {arg='epoch_size', type='number', default=-1,
       help='Number of examples presented per epoch. '..
       'Default is to use then entire dataset per epoch'}
   )
   self:setBatchSize(batch_size)
   self._epoch_size = epoch_size
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
      {arg='batch_size', type='number', default='64',
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

--Returns an iterator over samples for one epoch
--Default is to iterate sequentially over all examples
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
      batch = batch or dataset:batch(self._batch_size)
      stop = math.min(self._start+self._batch_size-1,nSample)
      -- inputs and targets
      dataset:sub(batch, self._start, stop)
      local indices = batch:indices() or torch.Tensor()
      -- metadata
      batch:setup{
         batch_iter=stop, batch_size=self._batch_size,
         n_sample=stop-self._start+1, 
         indices=indices:range(self._start,stop)
      }
      nSampled = nSampled + stop - self._start + 1
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
      end
      --http://bitsquid.blogspot.ca/2011/08/fixing-memory-issues-in-lua.html
      collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

------------------------------------------------------------------------
--[[ ShuffleSampler ]]--
-- Iterates over examples in a dataset by shuffling the example 
-- indices before each epoch.
------------------------------------------------------------------------
local ShuffleSampler, parent = torch.class("dp.ShuffleSampler", "dp.Sampler")

function ShuffleSampler:_init(config)
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
      batch = batch or dataset:batch(self._batch_size)
      stop = math.min(self._start+self._batch_size-1,nSample)
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
      nSampled = nSampled + stop - self._start + 1
      self._start = self._start + self._batch_size
      if self._start >= nSample then
         self._start = 1
         dataset_indices = torch.randperm(nSample):long()
      end
      collectgarbage() 
      return batch, math.min(nSampled, epochSize), epochSize
   end
end

------------------------------------------------------------------------
--[[ SentenceSampler ]]--
-- Iterates over parallel sentences of equal size one word at a time.
-- The sentences sizes are iterated through randomly.
-- Used for Recurrent Neural Network Language Models.
-- Note that it epoch_size is the minimum samples per epoch.
------------------------------------------------------------------------
local SentenceSampler, parent = torch.class("dp.SentenceSampler", "dp.Sampler")

function SentenceSampler:__init(config)
   config = config or {}
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, evaluate = xlua.unpack(
      {config},
      'SentenceSampler', 
      'Iterates over parallel sentences of equal size one word at a time. '..
      'The sentences sizes are iterated through randomly. '..
      'Used for Recurrent Neural Network Language Models. '..
      'Note that it epoch_size is the minimum samples per epoch.',
      {arg='evaluate', type='boolean', req=true,
       help='In evaluation mode, publishes to "beginSequence" Mediator '..
       'channel before each new Sequence. This prompts the Recurrent* Models '..
       'to forget the previous sequence of inputs. '..
       'In training mode (evaluate=false), published to "doneSequence" '..
       'channel to advise RecurrentVisitorChain to visit the model after '..
       'the sequence is propagated'}
   )
   self._evaluate = evaluate
   parent.__init(self, config)
end

function SentenceSampler:sampleEpoch(dataset)
   -- starting new epoch implies starting a new sequence
   if self._mediator then
      self._mediator:publish("beginSequence")
   end
   self._co = self._co or coroutine.create(function (batch) 
      self:_sampleEpoch(dataset, batch) 
   end)
   return function (batch)   -- "iterator"
      local code, batch, batchIter, epochSize = coroutine.resume(self._co, batch)
      if batch then
         return batch, batchIter, epochSize
      end
   end
end

function SentenceSampler:_sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   local nSample = dataset:nSample()
   
   local sentenceStartId = dataset:startId()
   local sentenceTable_, corpus = dataset:groupBySize()
   local text = corpus:select(2,2) -- second column is the text
      
   local epochSize = self._epoch_size or nSample
   local nSampled = 0
   local batch
   local function newBatch()
      return batch or dp.Batch{
         which_set=dataset:whichSet(), epoch_size=epochSize,
         inputs=dp.ClassView('b', torch.IntTensor{1}), 
         targets=dp.ClassView('b', torch.IntTensor{1})
         -- carry?
      } 
   end
   
   while true do
   
      local sentenceTable = {}
      for size, s in pairs(sentenceTable_) do
         sentenceTable[size] = {indices=s.indices:resize(s.count), sampleIdx=1}
      end
      local sentenceSizes = _.shuffle(_.keys(sentenceTable))
      local nSizes = #sentenceSizes
   
      while nSizes > 0 do
         
         for i,sentenceSize in pairs(sentenceSizes) do
            
            local s = sentenceTable[sentenceSize]
            local start = s.sampleIdx
            local stop = math.min(start + self._batch_size - 1, s.indices:size(1))
            -- batch of word indices, each at same position in different sentence
            local textIndices = s.indices:narrow(1, start, stop - start + 1)
            self._text_indices = self._text_indices or torch.LongTensor()
            self._text_indices:resize(textIndices:size(1))
            self._text_indices:copy(textIndices)
            textIndices = self._text_indices
            
            if self._mediator and self._evaluate then
               -- tells recurrent models to forget the past sequence
               self._mediator:publish("beginSequence")
            end
            
            for wordOffset=1,sentenceSize do

               if nSampled >= epochSize then
                  batch = coroutine.yield(false) or newBatch()
                  -- starting new epoch implies starting a new sequence
                  if self._mediator then
                     self._mediator:publish("beginSequence")
                  end
                  nSampled = 0
               end
               
               batch = batch or newBatch()
               local input_v = batch:inputs()
               assert(input_v.isClassView)
               local inputs = input_v:input() or torch.IntTensor()
               local target_v = batch:targets()
               assert(target_v.isClassView)
               local targets = target_v:input() or torch.IntTensor()
               
               if wordOffset == 1 then
                  inputs:resize(textIndices:size(1))
                  inputs:fill(sentenceStartId)
               else
                  inputs:copy(self._prev_targets)
               end
               
               targets:index(text, 1, textIndices)
               self._prev_targets = self._prev_targets or targets.new()
               self._prev_targets:resize(targets:size()):copy(targets)
               
               -- metadata
               batch:setup{
                  batch_iter=(nSampled + textIndices:size(1) - 1), 
                  batch_size=self._batch_size,
                  n_sample=textIndices:size(1)
               }
               
               -- re-encapsulate in dp.Views
               input_v:forward('b', inputs)
               input_v:setClasses(dataset:vocabulary())
               
               target_v:forward('b', targets)
               target_v:setClasses(dataset:vocabulary())
               
               nSampled = nSampled + textIndices:size(1)
               
               if self._mediator and (not self._evaluate) 
                     and (wordOffset == sentenceSize or nSampled >= epochSize) then
                  -- tells the RecurrentVisitorChain to update the model
                  -- when next called
                  self._mediator:publish("doneSequence")
               end
               
               coroutine.yield(batch, math.min(nSampled, epochSize), epochSize)
               -- move to next word in each sentence
               textIndices:add(1)
            end
            
            s.sampleIdx = s.sampleIdx + textIndices:size(1)
            if s.sampleIdx > s.indices:size(1) then
               sentenceSizes[i] = nil
               nSizes = nSizes - 1
            end
            
         end
         
         collectgarbage()
          
      end
   end
end
