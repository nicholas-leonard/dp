
function Optimizer:optimize(opfunc, parameters, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local feval = function(x)
        -- get new parameters
        if x ~= parameters then
           parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        --[[feedforward]]--
        -- evaluate function for complete mini batch
        local outputs = model:forward(inputs)
        -- average loss (a scalar)
        local loss = self._criterion:forward(outputs, targets)
        
        --[[backpropagate]]--
        -- estimate df/do (o is for outputs), a tensor
        local outputGrads = self._criterion:backward(outputs, targets)
        self._model:backward(inputs, outputGrads)
         
        --[[measure error]]--
        -- update feedback
        self._feedback:update(outputs, targets)
                       
        -- return f and df/dX
        return loss, gradParameters
     end
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   
   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      x:add(-clr, dfdx)
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
end

------------------------------------------------------------------------
--[[ Visitor ]]--
-- Visits the Sampler, Models and/or Criteria
-- Modifies their states
------------------------------------------------------------------------
local Visitor = torch.class("dp.Visitor")

function Visitor:__init(...)
   local args, id = xlua.unpack(
      'ModelVisitor', nil,
      {arg='id', type='dp.ObjectID'},
      {arg='include', type='table', 
       help='only models having a true value for the member named ' .. 
       'in this table are visited, unless the member is also listed ' ..
       'in the exclude table, in this case it is not visited'},
      {arg='exclude', type='table', default={},
       help='models having a member named in this table are not ' ..
       'visited, even if it is in the include table, i.e. ' ..
       'exclude has priority of include'}
   )
   self._id = id
   self._exclude = exclude
   self._include = include
end

function Visitor:id()
   return self._id
end

-- compares model to filter to see if it can be visited
function Visitor:canVisit(model)
   error"Not Implemented"
end

function Visitor:setup(mediator)
   self._mediator = mediator
end



------------------------------------------------------------------------
--[[ WeightDecay ]]--
-- ModelVisitor
-- Decays the weight of the visited parameterized models.
------------------------------------------------------------------------

local WeightDecay, parent = torch.class("dp.WeightDecay", "Visitor")

function WeightDecay:__init(config)
   local args, wd_factor = xlua.unpack(
      'WeightDecay', nil,
      {arg='wd_factor', type='number', help='Weight decay factor'},
   )
   self._wd_factor = wd_factor
   config.filter = config.filter or {}
   table.insert(config.filter, 'hasParameters')
   parent.__init(self, config)
end

function WeightDecay:preModel_setup(model)
   local state = model:state()
   state[self:id():name()] = {
      -- state for this visitor and model pair
      -- default is to initialize to empty table
   }
   model:setState(state, self:id():name())
end

function WeightDecay:preModel_update(model)
   if not self:canVisit(model) then
      return
   end
   local state = model:state(self:id():name)
   model:parameterGrads():add(wd, model:parameters())
   
   state.report = {
      -- report on weight decay statistics?
   }
   model:setState(state, self:id():name())
end


------------------------------------------------------------------------
--[[ Momentum ]]--
-- ModelVisitor
-- Applies momentum to parameterGradients
------------------------------------------------------------------------

local Momentum, parent = torch.class("dp.Momentum", "Visitor")

function Momentum:__init(...)
   self._momentum_factor = momentum_factor
   parent.__init(self, ...)
end

function WeightDecay:preModel_setup(model)
   local state = model:state()
   state[self:id():name()] = {
      -- state for this visitor and model pair
   }
   model:setState(state, self:id():name())
end

function Momentum:preModel_update(model)
   if not self:canVisit(model) then
      return
   end
   local state = model:state(self:id():name)
   -- Retrieve parameters and gradients:
   -- this extracts and flattens all the trainable parameters of the model
   -- into a 1-dim vector
   local parameters, gradients = model:parameters()
   state.parameters = parameters
   state.gradients = gradients 
   
   -- (3) apply momentum
   if self._momentum_factor == 0 then
      if not state.paramGrads then
         state.paramGrads = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end
   model:setState(state, self:id():name())
end


