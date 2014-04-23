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
   local carry = self:forward(batch)
   carry = self:monitor(batch, report, carry) or carry
   carry = self:backward(batch, carry) or carry
   self:update()
   self:doneBatch(report, carry)
end

function Optimizer:forward(batch)
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local carry = batch:carry()
   self.output.act, carry = self._model:forward(batch:inputs(), carry)
   
   -- measure loss and backprop gradients
   self.loss, carry = self._loss:forward(self.output.act, batch:targets(), carry)
   return carry
end

function Optimizer:backward(batch, carry)
   --[[ backpropagate ]]--
   -- estimate gradient of loss w.r.t. outputs, a basetensor
   self.output.grad, carry = self._loss:backward(self.output.act, batch:targets(), carry)
   
   -- backprop through model
   self._model:backward(self.output.grad, carry)
end

function Optimizer:update()
   --[[ update parameters ]]--
   -- visits models to perform updates
   self._model:accept(self._visitor)
end


