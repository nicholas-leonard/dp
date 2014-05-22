------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Evaluates (tests) a model using a sampling distribution.
-- For evaluating the generalization of the model, separate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------
local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")
Evaluator.isEvaluator = true

function Evaluator:propagateBatch(batch, report) 
   local carry = self:evaluate(batch)
   carry = self:monitor(batch, report, carry) or carry
   self:visitModel()
   self:doneBatch(report, carry)
end

function Evaluator:evaluate(batch, report)
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local carry = batch:carry()
   carry.evaluate = true
   self.output, carry = self._model:evaluate(batch:inputs(), carry)
   
   -- measure loss and backprop gradients
   self.loss, carry = self._loss:evaluate(self.output, batch:targets(), carry)
   return carry
end

function Evaluator:visitModel()
   -- visits the model
   if self._visitor then
      self._model:accept(self._visitor)
   end
end
