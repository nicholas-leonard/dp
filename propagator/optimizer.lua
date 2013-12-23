------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------

local Optimizer, parent = torch.class("dp.Optimizer", "dp.Propagator")
Optimizer.isOptimizer = true

function Optimizer:__init(config)
   config = config or {}
   local args, sampler, visitor, stats = xlua.unpack(
      {config},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='sampler', type='dp.Sampler', 
       help='used to iterate through the train set. ' ..
       'Defaults to dp.ShuffleSampler()'},
      {arg='visitor', type='dp.Visitor', req=true,
       help='visits models after forward-backward phase. ' .. 
       'Performs the parameter updates.'},
      {arg='stats', type='boolean', default=true,
       help='display statistics'}
   )
   config.sampler = sampler or dp.ShuffleSampler()
   config.stats = stats
   parent.__init(self, config)
end
      
function Optimizer:propagateBatch(batch, report)   
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local ostate = model:forward{input=batch:inputs()}
   batch:setOutputs(ostate.act:double())
   
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
                          report, batch)
   
   --[[ backpropagate ]]--
   -- estimate df/do (f is for loss, o is for outputs), a tensor
   batch:setOutputGradients(
      self._criterion:backward(batch:outputs(), batch:targets())
   )
   
   model:backward{output=batch:outputGradients()}

   
   --[[ update parameters ]]--
   model:accept(self._visitor)
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          report, batch)
end


