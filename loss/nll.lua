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
   self:zeroStatistics()
end

function NLL:_forward(carry)
   local input = self.input.act:feature()
   local target = self.input.target:class()
   self.loss = self._criterion:forward(input, target)
   return carry
end

function NLL:_backward(carry)
   local input = self.input.act:feature()
   local target = self.input.target:class()
   self.input.grad = self._criterion:backward(input, target)
   return carry
end

function NLL:_updateStatistics()
   self._stats.loss = self._stats.loss + self.loss               
end

function NLL:_zeroStatistics()
   self._stats.loss = 0
end

function NLL:report()
   return {loss=self._stats.loss/self._stats.nSample}
end
