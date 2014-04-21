------------------------------------------------------------------------
--[[ Node ]]--
-- Abstract Class
-- Inherited by Node and Loss
-- Forward and backward propagates states.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")
Node.isNode = true

function Node:__init()
   self.input = {}
   self.output = {}
end

function Node:setup(config)
   local args, mediator, id = xlua.unpack(
      {config or {}},
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
   self._setup = true
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

--- input activation basetensor or input state table
-- state.global : 
--- global state table accessible to all Nodes in the graph
-- state.carry :
--- a state that is carried throughout the graph. 
--- Nodes can modify it but should avoid deleting attributes
--- Useful when you want to forward information to a later Node
--- in the graph seperated by an unknown number of Nodes
function Node:forward(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor instance")
   self.input.act = input
   local carry = self:_forward(table.copy(carry)) or carry
   self.forwarded = true
   return self.output.act, carry
end

function Node:_forward(cstate)
   error"Not Implemented"
end

-- like forward, but for evaluation purposes (valid/test).
-- this is useful for stochastic Modules like Dropout, which have 
-- different behavior for training than for evaluation.
function Node:evaluate(input, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor instance")
   self.input.act = input
   self.carry.evaluate = true
   local carry = self:_evaluate(table.copy(carry)) or carry
   self.evaluated = true
   self.forwarded = true
   return self.output.act, carry
end

--default is to call forward (only diff is 'evaluate' flag in gstate)
function Node:_evaluate(carry)
   return self:_forward(carry)
end

function Node:backward(output, carry)
   assert(output.isBaseTensor, "Expecting dp.BaseTensor instance")
   self.output.act = output
   local carry = self:_backward(table.copy(carry)) or carry
   self.backwarded = true
   return self.input.grad, carry
end

function Node:_backward(cstate)
   error"Not Implemented"
end

-- experimental (would allow for one chained RPC call for both backward forward)
function Node:flux(carry)
   local output, carry = self:forward()
   local input, carry = self._successor:flux{output, carry}
   local input, carry = self:backward{input, carry}
   return input, carry
end

function Node:doneEpoch(report, ...)
end

function Node:doneBatch(...)
   self:_doneBatch(...)
   self.forwarded = false
   self.backwarded = false
   self.evaluated = false
   self.input = {}
   self.output = {}
end

function Node:_doneBatch(...)
end


function Node:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function Node:share(mlp, ...)
   error"Not Implemented"
end

-- creates a clone with shared parameters
function Node:sharedClone()
   error"Not Implemented"
end

-- shares parameters and statistics (use to share nodes between coroutines)
function Node:coroutineClone()
   error"Not Implemented"
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
