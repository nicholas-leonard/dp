------------------------------------------------------------------------
--[[ Node ]]--
-- Abstract Class
-- Inherited by Node and Loss
-- Forward and backward propagates states.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")
Node.isNode = true

function Node:setup(...)
   local args, mediator, id = xlua.unpack(
      {... or {}},
      'Node:setup', nil,
      {arg='mediator', type='dp.Mediator'},
      {arg='id', type='dp.ObjectID',
       help='Uniquely identifies node.'}
   )
   self._mediator = mediator
   -- the id should be given by the experiment since the same 
   -- model is shared by all propagators.
   self._id = id
   mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function Node:id()
   return self._id
end

function Node:name()
   return self._id:name()
end

-- returns a report of the Node.
-- if statistics were being gathered, this is the time to report them.
-- Expect report to be called at least every epoch.
function Node:report()
end

function Node:setInputState(istate)
   assert(istate, "No Input State")
   if istate.isBaseTensor then
      -- istate is BaseTensor, assume it represents activations
      istate = {act=istate}
   end
   assert(type(istate) == 'table')
   self.istate = istate
end

function Node:setGlobalState(gstate)
   self.gstate = gstate or {}
end

function Node:setOutputState(ostate)
   if ostate == nil then
      return
   elseif ostate.isBaseTensor then
      -- ostate is BaseTensor, assume it represents gradients
      self.ostate.grad = ostate
      return
   end
   assert(type(ostate) == 'table')
   self.ostate = ostate
end

function Node:forward(state)
   -- state.input :
   --- input activation tensor or input state table
   -- state.global : 
   --- global state table accessible to all Nodes in the graph
   -- state.carry :
   --- a state that is carried throughout the graph. 
   --- Nodes may or may not use or modify it.
   --- useful when you want to forward information to a later Node
   --- in the graph seperated by an unknown number of Nodes
   self:setInputState(state.input)
   self:setGlobalState(state.global)
   local cstate = self:_forward(table.copy(state.carry)) or state.carry
   self.forwarded = true
   return self.ostate, cstate
end

function Node:_forward(cstate)
   error"Not Implemented"
end

--like forward, but for evaluation purposes (valid/test).
--this is useful for stochastic Modules like Dropout, which have 
--different behavior for training than for evaluation.
function Node:evaluate(state)
   self:setInputState(state.input)
   self:setGlobalState(state.global)
   self.gstate.evaluate = true
   local cstate = self:_evaluate(table.copy(state.carry)) or state.carry
   self.evaluated = true
   self.forwarded = true
   return self.ostate, cstate
end

--default is to call forward (only diff is 'evaluate' flag in gstate)
function Node:_evaluate(cstate)
   return self:_forward(cstate)
end

function Node:backward(state)
   self:setOutputState(state.output)
   self:setGlobalState(state.global)
   local cstate = self:_backward(table.copy(state.carry)) or state.carry
   self.backwarded = true
   return self.istate, cstate
end

function Node:_backward(cstate)
   error"Not Implemented"
end

function Node:doneEpoch(report, ...)
end

function Node:doneBatch(...)
   self:_doneBatch(...)
   self.forwarded = false
   self.backwarded = false
   self.evaluated = false
   self.istate = {} -- input state
   self.gstate = {} -- global state
end

function Node:_doneBatch(...)
end

function Node:type(type)
   error"Not Implemented"
end

function Node:float()
   return self:type('torch.FloatTensor')
end

function Node:double()
   return self:type('torch.DoubleTensor')
end

function Node:cuda()
   return self:type('torch.CudaTensor')
end

function Node:reset()
   error"Not Implemented"
end
