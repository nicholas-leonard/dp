------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a transfer function.
-- For a linear transformation, you can use nn.Identity.
------------------------------------------------------------------------
local Neural, parent = torch.class("dp.Neural", "dp.Layer")
Neural.isNeural = true

function Neural:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, transfer, typename 
      = xlua.unpack(
      {config},
      'Neural', 
      'An affine transformation followed by a transfer function.',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._transfer = transfer
   self._affine = nn.Linear(input_size, output_size)
   config.typename = typename
   config.output = dp.DataView()
   config.input_view = 'bf'
   config.output_view = 'bf'
   parent.__init(self, config)
end

function Neural:_forward(carry)
   local activation = self:inputAct()
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not carry.evaluate)
      activation = self._dropout:forward(activation)
      self.mvstate.dropoutAct = activation
   end
   activation = self._affine:forward(activation)
   self.mvstate.affineAct = activation
   activation = self._transfer:forward(activation)
   self:outputAct(activation)
   return carry
end

function Neural:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.affineAct
   local output_grad = self:outputGrad()
   output_grad = self._transfer:backward(input_act, output_grad, scale)
   self.mvstate.affineGrad = output_grad
   input_act = self.mvstate.dropoutAct or self:inputAct()
   output_grad = self._affine:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self:inputAct()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self:inputGrad(output_grad)
   return carry
end

function Neural:paramModule()
   return self._affine
end

function Neural:_type(type)
   self:inputType(type)
   self:outputType(type)
   self._affine:type(type)
   self._transfer:type(type)
   if self._dropout then
      self._dropout:type(type)
   end
   return self
end

