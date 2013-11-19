

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
   
   --[[feedforward]]--
   -- evaluate function for complete mini batch
   local outputs = self._model:forward(inputs)
   -- average loss (a scalar)
   local loss = self._criterion:forward(outputs, targets)

   --[[measure error]]--
   -- update confusion
   self._feedback:batchAdd(outputs, targets)
end

      
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
