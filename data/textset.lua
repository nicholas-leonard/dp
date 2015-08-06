------------------------------------------------------------------------
--[[ TextSet ]]--
-- Inherits DataSet
-- Used for Language Modeling
-- Takes a sequence of words stored as a tensor of word ids.
-- Contrary to SentenceSet, this wrapper assumes a continuous stream 
-- of words. If consecutive sentences are completely unrelated, you 
-- might be better off using SentenceSet, unless your model can learn 
-- to forget (like LSTMs). 
-- Like SentenceSet, this class does not store its data in Views.
-- However, the outputs of batch(), sub(), index() are dp.Batches
-- containing ClassViews of inputs and targets.
------------------------------------------------------------------------
local TextSet, parent = torch.class("dp.TextSet", "dp.DataSet")
TextSet.isTextSet = true

function TextSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, which_set, data, context_size, words, recurrent, bidirectional 
      = xlua.unpack(
      {config},
      'TextSet', 
      'Stores a sequence of words where each word is represented as an integer.',
      {arg='which_set', type='string',
       help='"train", "valid" or "test" set'},
      {arg='data', type='torch.IntTensor', 
       help='A 1D torch.tensor of word ids.', req=true},
      {arg='context_size', type='number', default=1,
       help='number of previous words to be used to predict the next.'},
      {arg='words', type='table',
       help='A table mapping word_ids to the original word strings'},
      {arg='recurrent', type='number', default=false,
       help='For RNN training, set this to true. In which case, '..
       'outputs a target word for each input word'},
      {arg='bidirectional', type='boolean', default=false,
       help='For BiDirectionalLM, i.e. Bidirectional RNN/LSTMs, '..
       'set this to true. In which case, target = input'}
   )
   self:whichSet(which_set)
   assert(torch.type(data) == 'torch.IntTensor')
   assert(data:dim() == 1)
   self._data = data
   self._context_size = context_size
   self._words = words
   self._recurrent = recurrent
   self._bidirectional = bidirectional
end

function TextSet:contextSize(context_size)
   if context_size then
      self._context_size = context_size
   end
   return self._context_size
end

function TextSet:vocabulary()
   return self._words
end

function TextSet:nSample()
   return self._data:size(1)-self._context_size + (self._bidirectional and -1 or 0)
end

function TextSet:data()
   return self._data
end

function TextSet:textSize()
   return self._data:nElement()
end

function TextSet:offset()
   return self._bidirectional and 0 or 1
end

function TextSet:batch(batch_size)
   return self:index(torch.range(1,batch_size):type("torch.LongTensor"))
end

-- not recommended for training (use for evaluation)
function TextSet:sub(batch, start, stop)
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
   
   if self._bidirectional then
      assert(self._context_size == 1, "can only use sub with contextSize = 1 for recurrent networks")
      local data = self._data:sub(start, stop)
      inputs:resize(1,data:size(1)):copy(data)
      targets:set(inputs)
      target_v:forward('bt', targets)
   elseif self._recurrent then
      assert(self._context_size == 1, "can only use sub with contextSize = 1 for recurrent networks")
      local data = self._data:sub(start, stop+1)
      inputs:resize(1,data:size(1)):copy(data)
      targets:set(inputs:narrow(2,2,data:size(1)-1))
      target_v:forward('bt', targets)
      inputs = inputs:narrow(2,1,data:size(1)-1)
   else
      local batchSize = stop-start+1
      inputs:resize(batchSize,self._context_size)
      for i=1,batchSize do
         inputs[i]:copy(data:narrow(1,iq,self._context_size))
         
      end
      target_v:forward('b', targets)
   end
   
   -- encapsulate in dp.ClassViews
   input_v:forward('bt', inputs)
   input_v:setClasses(self._words)
   target_v:setClasses(self._words)
   
   return batch or dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=input_v, targets=target_v
   }   
end

-- used for training NNLM on large datasets
-- gets a random sample
function TextSet:sample(batch, batchSize)
   if not batchSize then
      batchSize = batch
      batch = nil
   end
   self._indices = self._indices or torch.LongTensor()
   self._indices:resize(batchSize)
   self._indices:random(self:nSample())
   local batch = self:index(batch, self._indices)
   return batch
end

-- used for training NNLM or RNNs on small datasets
-- indices index the targets in the data.
function TextSet:index(batch, indices)
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
   
   indices:add(self._context_size) -- offset by contextsize
   if self._bidirectional then
      inputs:resize(indices:size(1), self._context_size)
   elseif self._recurrent then
      inputs:resize(indices:size(1), self._context_size+1)
      inputs:select(2,self._context_size+1):index(self._data, 1, indices) -- last targets
   else
      inputs:resize(indices:size(1), self._context_size)
      targets:resize(indices:size(1)):index(self._data, 1, indices) -- only targets
   end
   
   for i = self._context_size,1,-1 do      
      indices:add(-1)
      inputs:select(2,i):index(self._data, 1, indices)
   end
   
   -- targets
   if self._bidirectional then
      targets:set(inputs)
      target_v:forward('bt', targets)
   elseif self._recurrent then
      targets:set(inputs:narrow(2,2,self._context_size))
      target_v:forward('bt', targets)
      inputs = inputs:narrow(2,1,self._context_size)
   else
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
