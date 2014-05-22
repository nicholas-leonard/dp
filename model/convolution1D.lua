------------------------------------------------------------------------
--[[ Convolution1D ]]--
-- [dropout] + convolution + max pooling + transfer function
-- Works on a SequenceTensor:conv1D() view.
------------------------------------------------------------------------
local Convolution1D, parent = torch.class("dp.Convolution1D", "dp.Layer")
Convolution1D.isConvolution1D = true

function Convolution1D:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, kernel_size, kernel_stride, 
         pool_size, pool_stride, transfer, typename
      = xlua.unpack(
      {config},
      'Convolution1D', 
      '[dropout] + TemporalConvolution + TemporalMaxPooling + transfer function',
      {arg='input_size', type='number', req=true,
       help='Number of input channels (the size of the word embedding)'},
      {arg='output_size', type='number', req=true,
       help='Number of output channels. (outputFrameSize)'},
      {arg='kernel_size', type='number', req=true,
       help='The size of the temporal convolution kernel.'},
      {arg='kernel_stride', type='number', default=1,
       help='The stride of the temporal convolution. '..
       'Note that depending of the size of your kernel, several (of '..
       'the last) columns of the input sequence might be lost. '..
       'It is up to the user to add proper padding in images.'},
      {arg='pool_size', type='number', req=true,
       help='The size of the temporal max pooling.'},
      {arg='pool_stride', type='number', req=true,
       help='The stride of the temporal max pooling.'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc.'},
      {arg='typename', type='string', default='convolution1d', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._kernel_size = kernel_size
   self._kernel_stride = kernel_stride
   self._pool_size = pool_size
   self._pool_stride = pool_stride
   self._transfer = transfer
   self._conv = nn.TemporalConvolution(
      input_size, output_size, kernel_size, kernel_stride
   )
   self._pool = nn.TemporalMaxPooling(pool_size, pool_stride)
   self._module = nn.Sequential()
   self._module:add(self._conv)
   self._module:add(self._transfer)
   self._module:add(self._pool)
   config.typename = typename
   config.input_view = 'bwc'
   config.output_view = 'bwc'
   parent.__init(self, config)
end

function Convolution1D:_forward(carry)
   local activation = self:inputAct()
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not carry.evaluate)
      activation = self._dropout:forward(activation)
      self.mvstate.dropoutAct = activation
   end
   activation = self._module:forward(activation)
   self:outputAct(activation)
   return carry
end

function Convolution1D:_backward(carry)
   local scale = carry.scale
   self._report.scale = scale
   local output_grad = self:outputGrad()
   local input_act = self.mvstate.dropoutAct or self:inputAct()
   output_grad = self._module:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self:inputAct()
      output_grad = self._dropout:backward(input_act, output_grad, scale)
   end
   self:inputGrad(output_grad)
   return carry
end

function Convolution1D:zeroGradParameters()
   self._conv:zeroGradParameters()
end

function Convolution1D:_type(type)
   self._input_type = type
   self._output_type = type
   self._module:type(type)
   if self._dropout then
      self._dropout:type(type)
   end
   return self
end

function Convolution1D:reset()
   self._conv:reset()
   if self._sparse_init then
      local W = self:parameters().weight.param
      self._sparseReset(W:t())
   end
end

function Convolution1D:share(conv2d, ...)
   assert(conv2d.isConvolution1D)
   return parent.share(self, conv2d, ...)
end

function Convolution1D:sharedClone()
   local clone = self:clone()
   return self:share(clone, 'weight', 'bias')
end

function Convolution1D:paramModule()
   return self._conv
end

-- number of output frames of the convolution1D layer
function Convolution1D:nOutputFrame(nInputFrame)
   assert(torch.type(nInputFrame) == 'number', "Expecting number")
   local nFrame = (nInputFrame - self._kernel_size) / self._kernel_stride + 1
   return (nFrame - self._pool_size) / self._pool_stride + 1
end
