
------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------

local Optimizer = torch.class("dp.Optimizer", "dp.Propagator")

function Optimizer:__init(config)
   local args, sampler, learning_rate = xlua.unpack(
      {config or {}},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='sampler', type='dp.Sampler', default=dp.ShuffleSampler(),
       help='used to iterate through the train set'},
      {arg='learning_rate', type='number', req=true,
       help='learning rate at start of learning'}
   )
   self:setLearningRate(learning_rate)
   Propagator.__init(self, config)
   self:setSampler(sampler)
end
      
function Optimizer:propagateBatch(batch)   
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   batch:setOutputs(model:forward(batch, visitor))
   
   -- average loss (a scalar)
   batch:setLoss(self._criterion:forward(outputs, targets))
   
   self:updateLoss(batch)
   
   -- monitor error 
   self._feedback:add(batch)
   
   --[[ backpropagate ]]--
   -- estimate df/do (o is for outputs), a tensor
   batch:setOutputGradients(self._criterion:backward(outputs, targets))
   self._model:backward(batch, visitor)

   --[[ update parameters ]]--
   self._model:update(batch, visitor)
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          self:report(), batch)
end


