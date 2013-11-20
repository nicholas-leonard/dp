
------------------------------------------------------------------------
--[[ Batch ]]--
-- Visitor
-- A batch of examples sampled from a dataset.
------------------------------------------------------------------------

local Batch = torch.class("dp.Batch")

function Batch:__init(...)
   local args, inputs, targets, batch_iter, epoch_size, batch_size
      = xlua.unpack(
      'Batch', nil,
      {arg='inputs', type='torch.Tensor', req=true,
       help='batch of inputs'},
      {arg='targets', type='torch.Tensor',
       help='batch of targets'},
      {arg='batch_iter', type='number'}, 
      {arg='epoch_size', type='number'},
      {arg='batch_size', type='number'}
   )
   self._inputs = inputs
   self._targets = targets
   self._batch_iter = batch_iter
   self._epoch_size = epoch_size
   self._batch_size = batch_size
end

function Batch:inputs()
   return self._inputs
end

function Batch:targets()
   return self._targets
end

function Batch:setOutputs(outputs)
   self._outputs = outputs
end

function Batch:outputs()
   return self._outputs
end

function Batch:setOutputGradients(output_gradients)
   self._output_gradients = output_gradients
end

function Batch:outputGradients(output_gradients)
   return self._output_gradients
end

function Batch:batchSize()
   return self._batch_size
end

function Batch:epochSize()
   return self._epoch_size
end

function Batch:batchIter()
   return self._batch_iter
end
   
