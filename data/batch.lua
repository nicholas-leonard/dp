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
local DataBatch, parent = torch.class("dp.DataBatch", "dp.DataSet")
DataBatch.isDataBatch = true

function DataBatch:__init(...)
   local args, inputs, targets, batch_iter, epoch_size, batch_size, 
      n_sample, classes, grad_type, indices
      = xlua.unpack(
      {... or {}},
      'DataBatch', nil,
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

-- TODO get classes from first target datatensor
function DataBatch:classes()
   return self._classes
end

function DataBatch:setOutputs(outputs)
   self._outputs = outputs
end

function DataBatch:outputs()
   return self._outputs:double()
end

function DataBatch:setLoss(loss)
   self._loss = loss
end

function DataBatch:loss()
   return self._loss
end

function DataBatch:setOutputGradients(output_gradients)
   self._output_gradients = output_gradients
end

function DataBatch:outputGradients()
   return self._output_gradients:type(self._grad_type)
end

function DataBatch:batchSize()
   return self._batch_size
end

function DataBatch:nSample()
   return self._n_sample
end

function DataBatch:epochSize()
   return self._epoch_size
end

function DataBatch:batchIter()
   return self._batch_iter
end

function DataBatch:indices()
   return self._indices
end
   
