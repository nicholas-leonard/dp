------------------------------------------------------------------------
--[[ Batch ]]--
-- State of a mini-batch to be fed into a model and criterion.
-- A batch of examples sampled from a dataset.
-- Serializable
------------------------------------------------------------------------
local Batch, parent = torch.class("dp.Batch", "dp.BaseSet")
Batch.isBatch = true

function Batch:setup(config)
   local args
   args, self._batch_iter, self._epoch_size, self._batch_size, 
      self._n_sample, self._grad_type, self._indices 
      = xlua.unpack(
      {config or {}},
      'Batch:setup', 
      'post-construction setup. Usually performed by Sampler.',
      {arg='inputs', type='dp.DataTensor | table of dp.DataTensors', 
       help='Sample inputs to a model. These can be DataTensors or '..
       'a table of DataTensors (in which case these are converted '..
       'to a CompositeTensor'},
      {arg='targets', type='dp.DataTensor | table of dp.DataTensors', 
       help='Sample targets to a model. These can be DataTensors or '..
       'a table of DataTensors (in which case these are converted '..
       'to a CompositeTensor. The indices of examples must be '..
       'in both inputs and targets must be aligned.'},
      {arg='batch_iter', type='number'}, 
      {arg='epoch_size', type='number'},
      {arg='batch_size', type='number'},
      {arg='n_sample', type='number'},
      {arg='grad_type', type='string'},
      {arg='indices', type='torch.Tensor', 
       help='indices of the examples in the original dataset.'}
   )
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
   
