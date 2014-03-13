
------------------------------------------------------------------------
--[[ Batch ]]--
-- State
-- A batch of examples sampled from a dataset.

-- TODO :
-- Make this inherit DataSet. This means that Feedback, Model and such
-- will have to expect this new interface. Particularly the ability 
-- to deal with tables of DataTensors, instead of torch.Tensors.
-- Samples should create the Batch once every epoch, for speed?
-- Make this a table (gstate), or allow it a gstate table.
------------------------------------------------------------------------

local Batch = torch.class("dp.Batch")
Batch.isBatch = true

function Batch:__init(...)
   local args, inputs, targets, batch_iter, epoch_size, batch_size, 
      n_sample, classes, grad_type, indices
      = xlua.unpack(
      {... or {}},
      'Batch', nil,
      {arg='inputs', type='torch.Tensor', req=true,
       help='batch of inputs'},
      {arg='targets', type='torch.Tensor',
       help='batch of targets'},
      {arg='batch_iter', type='number'}, 
      {arg='epoch_size', type='number'},
      {arg='batch_size', type='number'},
      {arg='n_sample', type='number'},
      {arg='classes', type='table', 
       help='temporary hack for confusion feedback worker until this '..
      'class in made to inherit DataSet and spawned from a dataset ' ..
      'via a Sampler.'},
      {arg='grad_type', type='string'},
      {arg='indices', type='torch.Tensor', 
       help='indices of the examples in the original dataset.'}
   )
   self._inputs = inputs
   self._targets = targets
   self._batch_iter = batch_iter
   self._epoch_size = epoch_size
   self._batch_size = batch_size
   self._n_sample = n_sample
   self._classes = classes
   self._grad_type = grad_type
   self._indices = indices
end

function Batch:inputs()
   return self._inputs
end

function Batch:targets()
   return self._targets
end

function Batch:classes()
   return self._classes
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
   
