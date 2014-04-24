------------------------------------------------------------------------
--[[ Node ]]--
-- Abstract Class
-- Inherited by Node and Loss
-- Forward and backward propagates states.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")
Node.isNode = true

function Node:__init()
   self:zeroStatistics()
   self:doneBatch()
end

function Node:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, id = xlua.unpack(
      {config},
      'Node:setup', nil,
      {arg='mediator', type='dp.Mediator', 
       help='allows Nodes to signal other object of events.'},
      {arg='id', type='dp.ObjectID',
       help='Uniquely identifies node.'}
   )
   self._mediator = mediator
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

--- carry table is carried throughout the graph. 
--- Nodes can modify it but should avoid deleting attributes.
--- Useful when you want to forward information to a later Node
--- in the graph seperated by an unknown number of Nodes
function Node:forward(input, carry)
   error"Not Implemented"
end

function Node:_forward(carry)
   error"Not Implemented"
end

-- like forward, but for evaluation purposes (valid/test).
-- this is useful for stochastic Modules like Dropout, which have 
-- different behavior for training than for evaluation.
function Node:evaluate(input, carry)
   error"Not Implemented"
end

--default is to call forward (only diff is 'evaluate' flag in carry)
function Node:_evaluate(carry)
   return self:_forward(carry)
end

function Node:backward(output, carry)
   error"Not Implemented"
end

function Node:_backward(carry)
   error"Not Implemented"
end

function Node:zeroStatistics()
   self._stats = {nSample=0}
   self:_zeroStatistics()
end

function Node:_zeroStatistics()
end

-- should only be called by forward or evaluate (once per batch)
function Node:updateStatistics(carry)
   self._stats.nSample = self._stats.nSample + carry.nSample
   self:_updateStatistics(carry)
end

function Node:_updateStatistics(carry)
end

function Node:doneBatch(...)
   self:_doneBatch(...)
   self.forwarded = false
   self.backwarded = false
   self.evaluated = false
   self.input = {}
end

function Node:_doneBatch(...)
end

function Node:doneEpoch(report, ...)
   --zeros statistics between epochs
   self:zeroStatistics()
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

--[[
-- experimental (would allow for one chained RPC call for both backward forward)
function Node:flux(carry)
   local output, carry = self:forward()
   local input, carry = self._successor:flux{output, carry}
   local input, carry = self:backward{input, carry}
   return input, carry
end
--]]
