------------------------------------------------------------------------
--[[ NLL ]]--
-- Loss subclass
-- Adapter of nn.ClassNLLCriterion
-- Negative Log Likelihood 
------------------------------------------------------------------------
local NLL, parent = torch.class("dp.NLL", "dp.Loss")
NLL.isNLL = true

function NLL:__init()
   self._criterion = nn.ClassNLLCriterion()
   parent.__init(self)
end

function NLL:_forward(carry)
   local input = self.input.act:feature()
   local target = self.input.target:class()
   self.loss = self._criterion:forward(input, target)
   print(self.loss, input:mean(), target:mean())
   return carry
end

function NLL:_backward(carry)
   local input = self.input.act:feature()
   local target = self.input.target:class()
   self.input.grad = self.input.act:featureClone(
      self._criterion:backward(input, target)
   )
   return carry
end
