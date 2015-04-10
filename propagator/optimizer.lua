------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Propagator subclass
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------
local Optimizer, parent = torch.class("dp.Optimizer", "dp.Propagator")
Optimizer.isOptimizer = true

function Optimizer:__init(config)
   config = config or {}
   local args, sampler, visitor, update_interval, stats = xlua.unpack(
      {config},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='sampler', type='dp.Sampler', 
       help='used to iterate through the train set. ' ..
       'Defaults to dp.ShuffleSampler()'},
      {arg='visitor', type='dp.Visitor', req=true,
       help='visits models after forward-backward phase. ' .. 
       'Performs the parameter updates.'},
      {arg='update_interval', type='number', default=1,
       help='update the model every update_interval'},
      {arg='stats', type='boolean', default=true,
       help='display statistics'}
   )
   self._update_interval = update_interval
   config.sampler = sampler or dp.ShuffleSampler()
   config.stats = stats
   parent.__init(self, config)
end
      
function Optimizer:propagateBatch(batch, report)
   self:training()
   self:forward(batch)
   self:monitor(batch, report)
   self:backward(batch)
   if report.epoch % self._update_interval == 0 then
      self:update()
   end
   self:doneBatch(report, carry)
end

function Optimizer:forward(batch)
   -- evaluate function for complete mini batch
   local input = batch:inputs():input()
   self.output = self._model:forward(input)
   
   -- measure loss and backprop gradients
   local target = batch:targets():input()
   self.err = self._loss:forward(self.output, target)
end

function Optimizer:backward(batch)
   -- estimate gradient of loss w.r.t. outputs
   local target = batch:targets():input()
   self.gradOutput = self._loss:backward(self.output, target)
   
   -- backprop through model
   local input = batch:inputs():input()
   self._model:backward(input, self.gradOutput)
end

function Optimizer:update()
   --[[ update parameters ]]--
   -- visits models to perform updates
   self._model:accept(self._visitor)
end


function Layer:_forward(carry)
   -- some modules like dropout have a different behavior during 
   -- evaluation vs training :
   if carry:getObj('evaluate') then 
      self._module:evaluate()
   else
      self._module:training()
   end
   self:outputAct(self._module:forward(self:inputAct()))
   return carry
end

function Layer:_backward(carry)
   local input_grad
   if self._acc_update then 
      input_grad = self._module:updateGradInput(self:inputAct(), self:outputGrad())
   else
      input_grad = self._module:backward(self:inputAct(), self:outputGrad(), self._acc_scale)
   end
   self:inputGrad(input_grad)
   return carry
end

function Layer:updateParameters(lr)
   if self._acc_update then
      self._module:accUpdateGradParameters(self:inputAct(), self:outputGrad(), lr*self._acc_scale)
   else
      self._module:updateParameters(lr)
   end
end
