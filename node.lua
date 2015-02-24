------------------------------------------------------------------------
--[[ Node ]]--
-- Abstract Class
-- Inherited by Model and Loss
-- Forward and backward propagates representations.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")
Node.isNode = true

function Node:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local default_type = torch.getdefaulttensortype()
   local args, input_type, output_type, module_type = xlua.unpack(
      {config},
      'Node', 
      'Forward and backward propagates representations.',
      {arg='input_type', type='string', default=default_type,
       help='type of input activation and gradient tensors'},
      {arg='output_type', type='string', default=default_type,
       help='type of output activation and gradient tensors'},
      {arg='module_type', type='string', default=default_type,
       help='type of modules used in this Node'}
   )
   self:inputType(input_type)
   self:outputType(output_type)
   self:moduleType(module_type)
   self:zeroStatistics()
   self:doneBatch()
end

function Node:setup(config)
   assert(type(config) == 'table', "Setup requires key-value arguments")
   local args, mediator, id = xlua.unpack(
      {config},
      'Node:setup', 
      'post-initialization setup method',
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

-- zero statistics between epochs
function Node:zeroStatistics()
   self._stats = {nSample=0}
   self:_zeroStatistics()
end

function Node:_zeroStatistics()
end

-- should only be called by forward or evaluate (once per batch)
function Node:updateStatistics(carry)
   self._stats.nSample = self._stats.nSample + carry:getObj('nSample')
   self:_updateStatistics(carry)
end

function Node:_updateStatistics(carry)
end

function Node:doneBatch(...)
   self:_doneBatch(...)
   self.forwarded = false
   self.backwarded = false
   self.evaluated = false
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

function Node:inputView(input_view)
   self._input_view = input_view or self._input_view
   return self._input_view
end

function Node:outputView(output_view)
   self._output_view = output_view or self._output_view
   return self._output_view
end

function Node:outputType(output_type)
   self._output_type = output_type or self._output_type
   return self._output_type
end

function Node:inputType(input_type)
   self._input_type = input_type or self._input_type
   return self._input_type
end

function Node:outputType(output_type)
   self._output_type = output_type or self._output_type
   return self._output_type
end

function Node:moduleType(module_type)
   self._module_type = module_type or self._module_type
   return self._module_type
end

function Node:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function Node:silent()
   self:verbose(false)
end

-- changes the type of internal variables inplace (same as nn)
-- returns self
function Node:type(new_type)
   if new_type then
      self:_type(new_type)
      self:moduleType(new_type)
      collectgarbage()
   end
   return self
end

-- this should only change the input, output, module or criteria 
-- types if the internal module or criteria permits it.
-- for example, NLL only changes the input type to float or double.
function Node:_type(new_type)
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

function Node:int()
   return self:type('torch.IntTensor')
end

function Node:long()
   return self:type('torch.LongTensor')
end

--default is to call forward (only diff is 'evaluate' flag in carry)
function Node:_evaluate(carry)
   return self:_forward(carry)
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
