------------------------------------------------------------------------
--[[ Packet ]]--
-- Transfered from Node to Node during forward/backward/updates, etc
-- Nodes pick the data that they need and clone these to forward
------------------------------------------------------------------------
local Packet = torch.class("dp.Packet")

function Packet:__init()
   local args, act, grad, carry = xlua.unpack(
      'Packet', nil,
      {arg='act', type='dp.BaseTensor'},
      {arg='grad', type='dp.BaseTensor'},
      {arg='carry', type='table'}
   )
   self._act = act
   self._grad = grad
   self._carry = carry or {}
end

function Packet:act()
   return self._act
end

function Packet:grad()
   return self._grad
end

function Packet:target()
   return self._target
end

function Packet:forwardClone()
   
end

function Packet:backwardClone()
   
end
