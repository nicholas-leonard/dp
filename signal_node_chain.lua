require 'torch'

require 'utils'

------------------------------------------------------------------------
--[[ Signal ]]--
-- A signal is passed along a Chain of responsibility (design pattern) 
-- as a request. It can be decorated (design pattern) by any Node in 
-- the chain. It can be composed of sub-signals. It has a log interface 
-- that can be updated by any Node. 
------------------------------------------------------------------------
local Signal = torch.class("dp.Signal")

function Signal:__init(...)
   self._task = 'Setup'
end

function Signal:task()
   return self._task
end

------------------------------------------------------------------------
--[[ Node ]]--
-- A node handles Signals by updating its state and passes it along 
-- to the calling Chain. A node is a component of a Chain, which is 
-- a composite (design pattern) of Nodes. Nodes cannot see each other. 
-- They communicate through the Signal.
------------------------------------------------------------------------
local Node = torch.class("dp.Node")

function Node:__init(name)
   self._name = "NoName"
end

function Node:name()
   return self._name
end

-- setup the object.
-- by default a chain is setup following construction by calling
-- propagating a signal with task="Setup"
function Node:handleSetup(signal)
   return signal, true
end

function Node:handleTask(signal)
   -- call a function by the name of task
   local handler = self['handle' .. signal:task()]
   if handler then
      return handler(self, signal)
   end
   return signal, false
end

function Node:handleSignal(signal)
   local signal, handled = self:handleTask(signal)
   if not handled then
      signal, handled = self:handleAny(signal)
      print"Warning: Unhandled signal by Node " .. self:name() ..
         " for task " .. signal:task() .. ". Maybe signal was " .. 
         "handled but didn't notify the caller?")
   end
   return signal, handled
end

function Node:handleAny(signal)
   -- default is to enforce task handling
   error("Error: unhandled signal by Node " .. self:name() ..
      " for task " .. signal:task() .. ". Consider implementing " .. 
      "a handleAny method for handling unknown tasks.")
end
   


------------------------------------------------------------------------
--[[ Chain ]]--
-- A chain is also a Node, but it is composed of a list of Nodes to be 
-- iterated (composite design pattern). A node can be referenced many 
-- time in a chain.
------------------------------------------------------------------------
local Chain = torch.class("dp.Chain", "dp.Node")

function Chain:__init(name, nodes)
   self._nodes = nodes
   Node:__init(name)
end

function Chain:nodes()
   return self._nodes
end

function Chain:handleAny(signal)
   local handled
   local saved_signal = signal
   -- iterate through the chain of nodes
   for i, node in ipairs(self:nodes()) do
      signal, handled = node:handleSignal(signal)
   end
end
