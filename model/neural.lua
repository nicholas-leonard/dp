------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a transfer function.
-- For a linear transformation, you can use nn.Identity.
-- Works on a DataTensor:feature() view.
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
   self._uncuda = false -- TODO: should detect non-cuda modules
   config.typename = typename
   parent.__init(self, config)
end

function Neural:inputAct()
   return self.input.act:feature(self._tensor_type)
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
   if self._uncuda then
      if self._recuda == nil then
         self._recuda = (activation:type() == 'torch.CudaTensor')
      end
      activation = activation:double()
   end
   self.mvstate.affineAct = activation
   activation = self._transfer:forward(activation)
   -- wrap torch.Tensor in a dp.DataTensor
   self.output.act = dp.DataTensor{data=activation}
   return carry
end

function Neural:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local input_act = self.mvstate.affineAct
   local output_grad = self.output.grad:feature()
   output_grad = self._transfer:backward(input_act, output_grad, scale)
   if self._recuda then
      output_grad = output_grad:cuda()
   end
   self.mvstate.affineGrad = output_grad
   input_act = self.mvstate.dropoutAct or self.input.act:feature()
   output_grad = self._affine:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self.input.act:feature()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self.input.grad = self.input.act:featureClone(output_grad)
   return carry
end

function Neural:paramModule()
   return self._affine
end

function Neural:_type(type)
   self._affine:type(type)
   if type ~= 'torch.CudaTensor' and not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
   return self
end

