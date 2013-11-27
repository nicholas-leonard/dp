
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
      {arg='learning_rate', type='number',
       help='learning rate at start of learning'},
      {arg='visitor', type='dp.Visitor', req=true,
       help='visits models after forward-backward phase. ' .. 
       'Performs the parameter updates.'}
   )
   self._learning_rate = learning_rate
   Propagator.__init(self, config)
   self:setSampler(sampler)
end
      
function Optimizer:propagateBatch(batch)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   model.istate.act = batch:inputs()
   model:forward()
   
   -- average loss (a scalar)
   batch:setLoss(
      self._criterion:forward(batch:outputs(), batch:targets())
   )
   
   self:updateLoss(batch)
   
   -- monitor error 
   if self._feedback then
      self._feedback:add(batch)
   end
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneFeedback", 
                          self:report(), batch)
   
   --[[ backpropagate ]]--
   -- estimate df/do (f is for loss, o is for outputs), a tensor
   batch:setOutputGradients(
      self._criterion:backward(batch:outputs(), batch:targets())
   )
   model.ostate.grad = batch:outputGradients()
   model:backward()

   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          self:report(), batch)
end

function Optimizer:report()
   local report = parent.report(self)
   report.learning_rate = self._learning_rate 
   return report
end


