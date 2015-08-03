------------------------------------------------------------------------
--[[ SentenceSampler ]]--
-- Iterates over parallel sentences of equal size one word at a time.
-- The sentences size are iterated through randomly.
-- Used for Recurrent Neural Network Language Models.
-- Note that epoch_size is the minimum number of samples per epoch.
------------------------------------------------------------------------
local SentenceSampler, parent = torch.class("dp.SentenceSampler", "dp.Sampler")

function SentenceSampler:__init(config)
   parent.__init(self, config)
   -- the billion words validation set has a sentence of 820 words???
   -- but the test set is alright
   self._max_size = config.max_size or 999999999999
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
         print(batch, "on dataset :", dataset:whichSet())
         error("corountine error")
      end
   end
end

function SentenceSampler:_sampleEpoch(dataset)
   dataset = dp.Sampler.toDataset(dataset)
   
   local sentenceStartId = dataset:startId()
   local sentenceTable_, corpus = dataset:groupBySize()
   local text = corpus:select(2,2) -- second column is the text
   
   -- remove sentences of size > self._max_size
   sentenceTable_ = _.map(sentenceTable_, 
      function(k,v) 
         if k <= self._max_size then
            return v
         else
            return nil 
         end
      end)
   
   local nSample = 0
   for size, s in pairs(sentenceTable_) do
      nSample = nSample + s.count
   end
      
   local epochSize = self._epoch_size or nSample
   local nSampled = 0
   local batch
   local function newBatch()
      return batch or dp.Batch{
         which_set=dataset:whichSet(), epoch_size=epochSize,
         inputs=dp.ClassView('bt', torch.IntTensor{{1}}), 
         targets=dp.ClassView('bt', torch.IntTensor{{1}})
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
            
            batch = batch or newBatch()
            local input_v = batch:inputs()
            assert(torch.isTypeOf(input_v, 'dp.ClassView'))
            local inputs = input_v:input() or torch.IntTensor()
            inputs:resize(textIndices:size(1), sentenceSize+1)
            local target_v = batch:targets()
            assert(torch.isTypeOf(target_v, 'dp.ClassView'))
            local targets = target_v:input() or torch.IntTensor()
            targets:set(inputs:narrow(2,2,inputs:size(2)-1))
            -- metadata
            batch:setup{
               batch_iter=(nSampled + textIndices:size(1) - 1), 
               batch_size=self._batch_size,
               n_sample=textIndices:size(1)
            }
            
            for wordOffset=1,sentenceSize do
               if wordOffset == 1 then
                  inputs:select(2,1):fill(sentenceStartId)
               end
               
               local target = inputs:select(2,wordOffset+1)
               target:index(text, 1, textIndices)         
               
               -- move to next word in each sentence
               textIndices:add(1)
            end
            
            inputs = inputs:narrow(2, 1, sentenceSize)
            
            -- re-encapsulate in dp.Views
            input_v:forward('bt', inputs)
            input_v:setClasses(dataset:vocabulary())
            target_v:forward('bt', targets)
            target_v:setClasses(dataset:vocabulary())
            
            nSampled = nSampled + textIndices:size(1)
            
            coroutine.yield(batch, math.min(nSampled, epochSize), epochSize)
            
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
         
         self:collectgarbage()
          
      end
   end
end

function SentenceSampler:write(file)
   local state = _.map(self, 
      function(k,v) 
         if k ~= '_co' then 
            return v 
         end
      end)
   file:writeObject(state)
end

function SentenceSampler:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
end
