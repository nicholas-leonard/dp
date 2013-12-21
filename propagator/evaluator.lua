------------------------------------------------------------------------
--[[ Evaluator ]]--
-- Tests (evaluates) a model using a sampling distribution.
-- For evaluating the generalization of the model, seperate the 
-- training data from the test data. The Evaluator can also be used 
-- for early-stoping.
------------------------------------------------------------------------


local Evaluator = torch.class("dp.Evaluator", "dp.Propagator")

      
function Evaluator:propagateBatch(batch, report) 
   local model = self._model
   --[[ feedforward ]]--
   -- evaluate function for complete mini batch
   local ostate = model:evaluate{input=batch:inputs()}
   batch:setOutputs(ostate.act)
   
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

   
   --[[ update parameters ]]--
   if self._visitor then
      model:accept(self._visitor)
   end
   model:doneBatch()
   
   --publish report for this optimizer
   self._mediator:publish(self:id():name() .. ':' .. "doneBatch", 
                          report, batch)
end
