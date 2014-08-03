------------------------------------------------------------------------
--[[ Null ]]--
-- Loss subclass
-- Does nothing
------------------------------------------------------------------------
local Null, parent = torch.class("dp.Null", "dp.Loss")
Null.isNull = true

function Null:__init(config)
   config = config or {}
   config.target_type = config.target_type or 'torch.IntTensor'
   config.target_view = 'null'
   config.input_view = 'null'
   parent.__init(self, config)
end

function Null:forward(input, target, carry)
   self.loss = 0.000000001
   return carry
end

function Null:evaluate(input, target, carry)
   self.loss = 0.000000001
   return carry
end

function Null:backward(input, target, carry)
   self.loss = 0.000000001
   return carry
end

function Null:_type(type)
end
