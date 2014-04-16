------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Propagator subclass
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
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local ostate, cstate = self._model:forward{input=batch:inputs()}
   
   -- used by loss and feedback
   local state = {input=ostate, target=batch:targets(), carry=cstate}
   
   -- measure loss and backprop gradients
   local loss, cstate = self._loss:forward(state)
   
   -- monitor error 
   if self._feedback then
      self._feedback:forward(state)
   end
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneFeedback", report, batch)
   
   --[[ backpropagate ]]--
   -- estimate gradient of loss w.r.t. outputs, a basetensor
   local istate, cstate = self._loss:backward(state)
   
   -- backprop through model
   self._model:backward{output=istate}

   --[[ update parameters ]]--
   -- visits models to perform updates
   self._model:accept(self._visitor)
   
   -- zero gradients, statistics, etc.
   self._model:doneBatch()
   self._loss:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:name()..':'.."doneBatch", report, batch)
end


