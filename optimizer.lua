
------------------------------------------------------------------------
--[[ Optimizer ]]--
-- Trains a model using a sampling distribution.
------------------------------------------------------------------------

local Optimizer = torch.class("dp.Optimizer", "dp.Propagator")

function Optimizer:__init(...)
   local args, sampler, learning_rate, weight_decay, momentum
      = xlua.unpack(
      {... or {}},
      'Optimizer', 
      'Optimizes a model on a training dataset',
      {arg='sampler', type='dp.Sampler', default=dp.ShuffleSampler(),
       help='used to iterate through the train set'},
      {arg='learning_rate', type='number', req=true,
       help='learning rate at start of learning'},
      {arg='weight_decay', type='number', default=0,
       help='weight decay coefficient'},
      {arg='momentum', type='number', default=0,
       help='momentum of the parameter gradients'}
   )
   self:setLearningRate(learning_rate)
   self:setWeightDecay(weight_decay)
   self:setMomentum(momentum)
   Propagator.__init(self, ...)
   self:setSampler(sampler)
end

function Optimizer:propogateBatch(batch)
   local inputs = batch:inputs()
   local targets = batch:targets()

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
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
        
        --[[backpropagate]]--
        -- estimate df/do (o is for outputs), a tensor
        local df_do = self._criterion:backward(outputs, targets)
        self._model:backward(inputs, df_do)
         
        --[[measure error]]--
        -- update confusion
        confusion:batchAdd(outputs, targets)
                       
        -- return f and df/dX
        return f, gradParameters
     end

   optim.sgd(feval, parameters, optimreport)
   self:doneBatch()
end
