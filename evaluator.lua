

------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Tests (evaluates) a model using a sampling distribution.
-- For evaluating the generalization of the model, seperate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------


local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")

function Evaluator:propogateBatch(batch)
   local inputs = batch:inputs()
   local targets = batch:targets()
   -- get new parameters
   if x ~= parameters then
     parameters:copy(x)
   end

   -- reset gradients
   gradParameters:zero()

   --[[feedforward]]--
   -- evaluate function for complete mini batch
   local outputs = self._model:forward(inputs)
   -- average loss (a scalar)
   local f = self._criterion:forward(outputs, targets)

   --[[measure error]]--
   -- update confusion
   confusion:batchAdd(outputs, targets)
   self:doneBatch()
end
