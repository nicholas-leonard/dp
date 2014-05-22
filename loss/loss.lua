------------------------------------------------------------------------
--[[ Loss ]]--
-- Node subclass
-- Adapter of nn.Criterion
------------------------------------------------------------------------
local Loss, parent = torch.class("dp.Loss", "dp.Node")
Loss.isLoss = true

function Loss:__init(config)
   local args, input_view, target_view, input_type, target_type 
      = xlua.unpack(
      'Loss', 
      'Adapter of nn.Criterion.',
      {arg='input_view', type='string', req=true,
       help='view of the input like "bf", "bhwc", etc.'},
      {arg='target_view', type='string', req=true,
       help='view of the target like "bt", "b", etc.'},
      {arg='target_type', type='string', req=true,
       'type of target tensors'},
      {arg='input_type', type='string', default='torch.DoubleTensor',
       'type of input activation and gradient tensors'}
   )
   self:inputView(input_view)
   self:outputView(target_view)
   config.input_type = input_type
   config.output_type = target_type
   parent.__init(self, config)
end

function Loss:forward(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input.act = input
   self.input.target = target
   carry = self:_forward(carry) or carry
   self:updateStatistics(carry)
   self.forwarded = true
   return self.loss, carry
end

function Loss:evaluate(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input.act = input
   self.input.target = target
   carry = self:_evaluate(carry) or carry
   self:updateStatistics(carry)
   self.evaluated = true
   self.forwarded = true
   return self.loss, carry
end

function Loss:backward(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input.act = input
   self.input.target = target
   carry = self:_backward(carry) or carry
   assert(self.input.grad.isView, "Expecting dp.View grad")
   self.backwarded = true
   return self.input.grad, carry
end

function Loss:inputAct()
   return self.input.act:feature(self._input_type)
end

function Loss:inputGrad(input_grad)
   if input_grad then
      self.input.grad = self.input.act:shallowClone()
      self.input.grad:setData(input_grad)
      return
   end
   return self.input.grad:feature(self._input_type)
end

function Loss:_updateStatistics()
   self._stats.loss = self._stats.loss + self.loss               
end

function Loss:_zeroStatistics()
   self._stats.loss = 0
end

function Loss:avgError()
   return self._stats.loss/(self._stats.nSample+0.0000000001)
end

function Loss:report()
   local err = self:avgError()
   print(self:id():toString() .. ' avgError ' .. err)
   local report = {avgError=err}
   return self._report(report) or report
end

function Loss:_report(report)
end
