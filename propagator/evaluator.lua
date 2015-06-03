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
   self._model:evaluate()
   self:forward(batch)
   self:monitor(batch, report)
   if self._callback then
      self._callback(self._model, report)
   end
   self:doneBatch(report)
end
