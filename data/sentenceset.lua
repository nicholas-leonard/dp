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

function SentenceSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, which_set, data, context_size, end_id, start_id, 
      words = xlua.unpack(
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
      {arg='context_size', type='number', req=true,
       help='number of previous words to be used to predict the next.'},
      {arg='end_id', type='number', req=true,
       help='word_id of the sentence end delimiter : "</S>"'},
      {arg='start_id', type='number', req=true,
       help='word_id of the sentence start delimiter : "<S>"'},
      {arg='words', type='table',
       help='A table mapping word_ids to the original word strings'}
   )
   self:setWhichSet(which_set)
   self._data = data
   self._context_size = context_size
   self._start_id = start_id
   self._end_id = end_id
   self._words = words
end
function SentenceSet:nSample()
   return self._data:size(1)
end

function SentenceSet:setInputs(inputs)
   error"Not Implemented"
end

function SentenceSet:setTargets(targets)
   error"Not Implemented"
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

function SentenceSet:sub(batch, start, stop)
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

function SentenceSet:index(batch, indices)
   local inputs, targets, input_v, target_v
   if (not batch) or (not indices) then 
      indices = indices or batch
      batch = nil
      inputs = torch.IntTensor(indices:size(1), self._context_size)
      targets = torch.IntTensor(indices:size(1))
      input_v = dp.ClassView()
      target_v = dp.ClassView()
   else
      input_v = batch:inputs()
      inputs = input_v:input()
      inputs:resize(indices:size(1), self._context_size)
      target_v = batch:targets()
      targets = target_v:input()
      targets:resize(indices:size(1))
   end
   -- fill tensor with sentence end tags : <S>
   inputs:fill(self._end_id)
   -- indexSelect the data and reuse memory (optimization)
   self.__index_mem = self.__index_mem or torch.IntTensor()
   self.__index_mem:index(self._data, 1, indices)
   local data = self.__index_mem
   local words = self._data:select(2, 2)
   for i=1,data:size(1) do
      local sample = data:select(1, i)
      -- add input
      local sample_stop = indices[i]-1
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
   targets:copy(data:select(2,2))
   
   -- encapsulate in dp.Views
   input_v:forward('bt', inputs)
   input_v:setClasses(self._words)
   
   target_v:forward('b', targets)
   target_v:setClasses(self._words)
   return batch or dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=input_v, targets=target_v
   }  
end
