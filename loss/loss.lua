------------------------------------------------------------------------
--[[ Loss ]]--
-- Node subclass
-- Adapter of nn.Criterion
------------------------------------------------------------------------
local Loss, parent = torch.class("dp.Loss", "dp.Node")
Loss.isLoss = true

function Loss:forward(input, target, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor for input")
   assert(target.isBaseTensor, "Expecting dp.BaseTensor for target")
   self.input.act = input
   self.input.target = target
   carry = self:_forward(carry) or carry
   self:updateStatistics(carry)
   self.forwarded = true
   return self.loss, carry
end

function Loss:evaluate(input, target, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor for input")
   assert(target.isBaseTensor, "Expecting dp.BaseTensor for target")
   self.input.act = input
   self.input.target = target
   carry = self:_evaluate(carry) or carry
   self:updateStatistics(carry)
   self.evaluated = true
   self.forwarded = true
   return self.loss, carry
end

function Loss:backward(input, target, carry)
   assert(input.isBaseTensor, "Expecting dp.BaseTensor for input")
   assert(target.isBaseTensor, "Expecting dp.BaseTensor for target")
   self.input.act = input
   self.input.target = target
   carry = self:_backward(carry) or carry
   assert(self.input.grad.isBaseTensor, "Expecting dp.BaseTensor grad")
   self.backwarded = true
   return self.input.grad, carry
end

function Loss:_updateStatistics()
   self._stats.loss = self._stats.loss + self.loss               
end

function Loss:_zeroStatistics()
   self._stats.loss = 0
end

function Loss:loss()
   return self._stats.loss/(self._stats.nSample+0.0000000001)
end

function Loss:report()
   self:loss()
   print(self:id():toString() .. ' loss ' .. loss)
   return {loss=loss}
end
