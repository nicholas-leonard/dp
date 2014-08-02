------------------------------------------------------------------------
--[[ Loss ]]--
-- Node subclass
-- Adapter of nn.Criterion
------------------------------------------------------------------------
local Loss, parent = torch.class("dp.Loss", "dp.Node")
Loss.isLoss = true

function Loss:__init(config)
    assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, input_view, target_view, target_type 
      = xlua.unpack(
      {config},
      'Loss', 
      'Adapter of nn.Criterion.',
      {arg='input_view', type='string', req=true,
       help='view of the input like "bf", "bhwc", etc.'},
      {arg='target_view', type='string', req=true,
       help='view of the target like "bt", "b", etc.'},
      {arg='target_type', type='string', 
       default=torch.getdefaulttensortype(),
       'type of target tensors'}
   )
   self:inputView(input_view)
   self:outputView(target_view)
   config.output_type = target_type
   parent.__init(self, config)
end

function Loss:forward(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input = input
   self.target = target
   carry = self:_forward(carry) or carry
   self:updateStatistics(carry)
   self.forwarded = true
   return self.loss, carry
end

function Loss:evaluate(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input = input
   self.target = target
   carry = self:_evaluate(carry) or carry
   self:updateStatistics(carry)
   self.evaluated = true
   self.forwarded = true
   return self.loss, carry
end

function Loss:backward(input, target, carry)
   assert(input.isView, "Expecting dp.View for input")
   assert(target.isView, "Expecting dp.View for target")
   self.input = input
   self.target = target
   carry = self:_backward(carry) or carry
   self.backwarded = true
   return self.input, carry
end

function Loss:_forward(carry)
   local input, target = self:inputAct(), self:targetAct()
   self.loss = self._criterion:forward(input, target)
   return carry
end

function Loss:_backward(carry)
   local input, target = self:inputAct(), self:targetAct()
   self:inputGrad(self._criterion:backward(input, target))
   return carry
end

-- Get
function Loss:inputAct()
   return self.input:forward(self._input_view, self._input_type)
end

function Loss:targetAct()
   return self.target:forward(self._output_view, self._output_type)
end

-- Set
function Loss:inputGrad(input_grad)
   self.input:backward(self._input_view, input_grad)
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

function Loss:_type(type)
   self:inputType(type)
   self:outputType(type)
   self._criterion:type(type)
end
