------------------------------------------------------------------------
--[[ SentenceSet ]]--
-- Takes a sequence of words stored as a tensor of word ids,
-- and a tensor holding the start index of the sentence of its 
-- commensurate word id (the one at the same index).
-- Unlike DataSet, for memory efficiency reasons, 
-- this class does not store its data in BaseTensors.
-- However, the outputs of batch(), sub(), index() are dp.Batches
-- containing WordTensors (inputs) and ClassTensors (targets).
------------------------------------------------------------------------
local SentenceSet, parent = torch.class("dp.SentenceSet", "dp.DataSet")
SentenceSet.isSentenceSet = true

function SentenceSet:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, which_set, data = xlua.unpack(
      {config},
      'SentenceSet', 
      'Stores a sequence of sentences. Each sentence is a sequence '..
      'words. Each word is represented as an integer.',
      {arg='which_set', type='string',
       help='"train", "valid" or "test" set'},
      {arg='data', type='torch.Tensor', 
       help='A torch.tensor with 2 columns. First is for storing '..
       'start indices of sentences. Sencond is for storing the '..
       'sequence of words as shuffled sentences.', req=true},
   )
   self:setWhichSet(which_set)
   self._data = data
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
function SentenceSet:preprocess(...)
   error"Not Implemented"
end

-- builds a batch (factory method)
-- doesn't copy the inputs or targets (so don't modify them)
function SentenceSet:batch(batch_size)
   return dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=self:inputs():sub(1, batch_size),
      targets=self:targets() and self:targets():sub(1, batch_size)
   }
end

function SentenceSet:sub(start, stop)
   return dp.Batch{
      which_set=self:whichSet(), epoch_size=self:nSample(),
      inputs=self:inputs():sub(start, stop),
      targets=self:targets() and self:targets():sub(start, stop)
   }    
end

function SentenceSet:index(batch, indices)
   if (not batch) or (not indices) then 
      indices = indices or batch
      return dp.Batch{
         which_set=self:whichSet(), epoch_size=self:nSample(),
         inputs=self:inputs():index(indices),
         targets=self:targets() and self:targets():index(indices)
      }
   end
   assert(batch.isBatch, "Expecting dp.Batch at arg 1")
   self:inputs():index(batch:inputs(), indices)
   if self:targets() then
      self:targets():index(batch:targets(), indices)
   end
   return batch
end
