------------------------------------------------------------------------
--[[ SentenceSet ]]--
-- Inherits DataSet
-- Used for Language Modeling
-- Takes a sequence of words stored as a tensor of word ids,
-- and a tensor holding the start index of the sentence of its 
-- commensurate word id (the one at the same index).
-- Unlike DataSet, for memory efficiency reasons, 
-- this class does not store its data in Views.
-- However, the outputs of batch(), sub(), index() are dp.Batches
-- containing ClassViews of inputs and targets.
-- The returned batch:inputs() are filled according to 
-- https://code.google.com/p/1-billion-word-language-modeling-benchmark/source/browse/trunk/README.perplexity_and_such
------------------------------------------------------------------------
local SentenceSet, parent = torch.class("dp.SentenceSet", "dp.DataSet")
SentenceSet.isSentenceSet = true

SentenceSet._input_shape = 'bt'
SentenceSet._output_shape = 'b'

function SentenceSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, which_set, data, context_size, recurrent, 
      end_id, start_id, words = xlua.unpack(
      {config},
      'SentenceSet', 
      'Stores a sequence of sentences. Each sentence is a sequence '..
      'words. Each word is represented as an integer.',
      {arg='which_set', type='string',
       help='"train", "valid" or "test" set'},
      {arg='data', type='torch.Tensor', 
       help='A torch.tensor with 2 columns. First col is for storing '..
       'start indices of sentences. Second col is for storing the '..
       'sequence of words as shuffled sentences. Sentences are '..
       'only seperated by the sentence_end delimiter.', req=true},
      {arg='context_size', type='number', default=5,
       help='number of previous words to be used to predict the next.'},
      {arg='recurrent', type='number', default=false,
       help='For RNN training, set this to true. In which case, '..
       'outputs a target word for each input word'},
      {arg='end_id', type='number', req=true,
       help='word_id of the sentence end delimiter : "</S>"'},
      {arg='start_id', type='number', req=true,
       help='word_id of the sentence start delimiter : "<S>"'},
      {arg='words', type='table',
       help='A table mapping word_ids to the original word strings'}
   )
   self:whichSet(which_set)
   self._data = data
   assert(data[data:size(1)][2] == end_id ,"data should be terminated with end_id")
   self._context_size = context_size
   self._recurrent = recurrent
   self._start_id = start_id
   self._end_id = end_id
   self._words = words
end

function SentenceSet:startId()
   return self._start_id
end

function SentenceSet:vocabulary()
   return self._words
end

function SentenceSet:nSample()
   return self._data:size(1)
end

function SentenceSet:inputs()
   error"Not Implemented"
end

function SentenceSet:targets()
   error"Not Implemented"
end

-- We assume that preprocessing has already been applied
function SentenceSet:preprocess()
   error"Not Implemented"
end

function SentenceSet:batch(batch_size)
   return self:sub(1, batch_size)
end

-- not recommended for training (use for evaluation)
-- used for NNLMs (not recurrent models)
function SentenceSet:sub(batch, start, stop)
   assert(not self._recurrent, "SentenceSet:sub not supported with self._recurrent")
   local input_v, inputs, target_v, targets
   if (not batch) or (not stop) then 
      if batch then
         stop = start
         start = batch
         batch = nil
      end
      inputs = torch.IntTensor()
      targets = torch.IntTensor()
      input_v = dp.ClassView()
      target_v = dp.ClassView()
  else
      input_v = batch:inputs()
      inputs = input_v:input()
      target_v = batch:targets()
      targets = target_v:input()
   end  
   local data = self._data:sub(start, stop)
   inputs:resize(data:size(1), self._context_size)
   targets:resize(data:size(1))
   local words = self._data:select(2, 2)
   -- fill tensor with sentence end tags : </S>
   inputs:fill(self._end_id)
   for i=1,data:size(1) do
      local sample = data:select(1, i)
      -- add input
      local sample_stop = start+i-2
      local sentence_start = self._context_size
      if sample[1] <= sample_stop then
         local sample_start = math.max(sample[1], sample_stop-self._context_size+1)
         local context = words:sub(sample_start, sample_stop)
         sentence_start = self._context_size-context:size(1)
         inputs:select(1, i):narrow(
            1, sentence_start+1, context:size(1)
         ):copy(context)
      end
      -- add sentence start tag : <S> (after sentence end tags)
      if sentence_start > 0 then
         inputs:select(1,i):narrow(1, sentence_start, 1):fill(self._start_id)
      end
   end   
   -- targets
   targets:copy(data:select(2,2))
   
   -- encapsulate in dp.ClassViews
   input_v:forward('bt', inputs)
   input_v:setClasses(self._words)
   
   target_v:forward('b', targets)
   target_v:setClasses(self._words)
   
   return batch or dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=input_v, targets=target_v
   }   
end

-- used for training NNLM on large datasets
-- gets a random sample
function SentenceSet:sample(batch, batchSize)
   if not batchSize then
      batchSize = batch
      batch = nil
   end
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchSize)
   self._indices:random(self._data:size(1)-(self._context_size+1))
   local batch = self:index(batch, self._indices)
   return batch
end

-- used for training NNLM or RNNs on small datasets
function SentenceSet:index(batch, indices)
   local inputs, targets, input_v, target_v
   if (not batch) or (not indices) then 
      indices = indices or batch
      batch = nil
      inputs = torch.IntTensor()
      targets = torch.IntTensor()
      input_v = dp.ClassView()
      target_v = dp.ClassView()
   else
      input_v = batch:inputs()
      inputs = input_v:input()
      target_v = batch:targets()
      targets = target_v:input()
   end
   if self._recurrent then
      inputs:resize(indices:size(1), self._context_size+1)
   else
      inputs:resize(indices:size(1), self._context_size)
      targets:resize(indices:size(1))
   end
   -- fill tensor with sentence end tags : <S>
   inputs:fill(self._end_id)
   -- indexSelect the data and reuse memory (optimization)
   self.__index_mem = self.__index_mem or torch.IntTensor()
   self.__index_mem:index(self._data, 1, indices)
   local data = self.__index_mem -- contains the batch of targets
   local words = self._data:select(2, 2)
   
   for i=1,data:size(1) do
      local sample = data:select(1, i) -- sentence start index
      local sample_stop = indices[i]-1 -- indices[i] is target index
      local sentence_start = self._context_size
      if sample[1] <= sample_stop then
         local sample_start = math.max(sample[1], sample_stop-self._context_size+1)
         local context = words:sub(sample_start, sample_stop)
         sentence_start = self._context_size-context:size(1)
         inputs:select(1, i):narrow(
            1, self._context_size-context:size(1)+1, context:size(1)
         ):copy(context)
      end
      -- add sentence start tag : <S> (after sentence end tags)
      if sentence_start > 0 then
         inputs:select(1,i):narrow(1, sentence_start, 1):fill(self._start_id)
      end
   end
   
   -- targets
   if self._recurrent then
      inputs:select(2,self._context_size+1):copy(data:select(2,2))
      targets:set(inputs:narrow(2,2,self._context_size))
      target_v:forward('bt', targets)
      inputs = inputs:narrow(2,1,self._context_size)
   else
      targets:copy(data:select(2,2))
      target_v:forward('b', targets)
   end
   
   input_v:forward('bt', inputs)
   input_v:setClasses(self._words)
   target_v:setClasses(self._words)
   
   return batch or dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=input_v, targets=target_v
   }  
end

-- returns sentence start indices organized by sentence size.
-- (used by RecurrentSampler)
function SentenceSet:groupBySize(bufferSize)
   bufferSize = bufferSize or 1000
   if not self._sentences then
      local sentenceCache = {}
      local sentenceStartIdx = self._data[1][1]
      local nTotalWord = self._data:size(1)
      local nWord = 0
      local i = 0
      self._data:select(2,1):apply(
         function(startIdx)
            i = i + 1
            if startIdx ~= sentenceStartIdx or i == nTotalWord then
               if i == nTotalWord then
                  nWord = nWord + 1
               end
               assert(nWord > 1, "empty sentence encountered")
               local s = sentenceCache[nWord]
               if not s then
                  s = {indices=torch.LongTensor(bufferSize), count=0}
                  sentenceCache[nWord] = s
               end
               s.count = s.count + 1
               local nIndex = s.indices:size(1)
               if s.count > nIndex then
                  s.indices:resize(nIndex + bufferSize)
               end
               s.indices[s.count] = sentenceStartIdx
               sentenceStartIdx = startIdx
               nWord = 1
            else
               nWord = nWord + 1
            end
         end
      )
      self._sentences = sentenceCache
   end
   return self._sentences, self._data
end
