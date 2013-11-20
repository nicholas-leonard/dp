

------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Tests (evaluates) a model using a sampling distribution.
-- For evaluating the generalization of the model, seperate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------


local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")

      
function Evaluator:propagateBatch(batch)   
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   batch:setOutputs(model:forward(batch, visitor))
   
   -- average loss (a scalar)
   batch:setLoss(self._criterion:forward(outputs, targets))
   
   self:updateLoss(batch)
   
   -- monitor error 
   self._feedback:add(batch)
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          self:report(), batch)
end
