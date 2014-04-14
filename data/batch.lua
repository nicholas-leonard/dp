------------------------------------------------------------------------
--[[ Batch ]]--
-- State of a mini-batch to be fed into a model and criterion.
-- A batch of examples sampled from a dataset.
-- Serializable
------------------------------------------------------------------------
local Batch, parent = torch.class("dp.Batch", "dp.BaseSet")
Batch.isBatch = true

function Batch:__init(config)
   local args, epoch_size = xlua.unpack(
      {config or {}},
      'Batch', 
      'State of a mini-batch to be fed into a model and criterion.',
      {arg='epoch_size', type='number',
       help='number of samples in original dataset'}
   )
   self._epoch_size = epoch_size
   parent.__init(self, config)
end

function Batch:setup(config)
   local args, epoch_size
   args, self._batch_iter, self._batch_size, self._n_sample, 
      self._grad_type, self._indices, epoch_size
      = xlua.unpack(
      {config or {}},
      'Batch:setup', 
      'post-construction setup. Usually performed by Sampler.',
      {arg='batch_iter', type='number',
       help='Count of the number of examples seen so far. Used to '..
       'update progress bar. Shouldn\'t be larger than epoch_size.'}, 
      {arg='batch_size', type='number'
       help='Maximum batch_size.'},
      {arg='n_sample', type='number',
       help='Actual number of examples in batch. Shouldn\'t be '..
       'larger than batch_size.'},
      {arg='grad_type', type='string',
       help='Type of output gradient : cuda | float | double'},
      {arg='indices', type='torch.Tensor', 
       help='indices of the examples in the original dataset.'},
      {arg='epoch_size', type='number',
       help='number of samples in epoch dataset.'}
   )
   self._epoch_size = epoch_size or self._epoch_size
end

function Batch:setOutputs(outputs)
   self._outputs = outputs
end

function Batch:outputs()
   return self._outputs:double()
end

function Batch:setLoss(loss)
   self._loss = loss
end

function Batch:loss()
   return self._loss
end

function Batch:setOutputGradients(output_gradients)
   self._output_gradients = output_gradients
end

function Batch:outputGradients()
   return self._output_gradients:type(self._grad_type)
end

function Batch:batchSize()
   return self._batch_size
end

function Batch:nSample()
   return self._n_sample
end

function Batch:epochSize()
   return self._epoch_size
end

function Batch:batchIter()
   return self._batch_iter
end

function Batch:indices()
   return self._indices
end
   
