------------------------------------------------------------------------
--[[ SentenceSampler ]]--
-- Iterates over parallel sentences of equal size one word at a time.
-- The sentences size are iterated through randomly.
-- Used for Recurrent Neural Network Language Models.
-- Note that epoch_size is the minimum number of samples per epoch.
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
      'Publishes to "beginSequence" Mediator channel before each '..
      'new Sequence, which prompts the Recurrent* Models '..
      'to forget the previous sequence of inputs. '..
      'Note that epoch_size only garantees the minimum number of '..
      'samples per epoch. More could be sampled.',
      {arg='evaluate', type='boolean', req=true,
       help='In training mode (evaluate=false), the object publishes '..
       'to "doneSequence" channel to advise RecurrentVisitorChain to '..
       'visit the model after the sequence is propagated'}
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
      if batch and not torch.isTypeOf(batch, 'dp.Batch') then
         error("expecting dp.Batch, got"..torch.type(batch))
      end
      
      local code, batch, batchIter, epochSize = coroutine.resume(self._co, batch)
      
      if code and batch then
         batch = self._ppf(batch)
         return batch, batchIter, epochSize
      elseif not code then
         print(batch, dataset:whichSet())
         error("corountine error")
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
            
            if self._mediator then
               -- tells recurrent models to forget the past sequence
               self._mediator:publish("beginSequence")
            end
            
            for wordOffset=1,sentenceSize do
               
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
               
               if self._mediator and (not self._evaluate) and wordOffset == sentenceSize then
                  -- tells the RecurrentVisitorChain to update the model when next called
                  self._mediator:publish("doneSequence")
               end
               
               coroutine.yield(batch, math.min(nSampled, epochSize), epochSize)
               
               -- move to next word in each sentence
               textIndices:add(1)
            end
            
            if nSampled >= epochSize then
               batch = coroutine.yield(false)
               nSampled = 0
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

