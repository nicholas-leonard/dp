-- WORK IN PROGRESS : NOT READY FOR USE
------------------------------------------------------------------------
--[[ Convolution2D ]]--
-- Uses CUDA 
-- [dropout] + convolution + max pooling + transfer function
-- Works on a ImageTensor:imageCHWB() view.
------------------------------------------------------------------------
local Convolution2D, parent = torch.class("dp.Convolution2D", "dp.Layer")
Convolution2D.isConvolution2D = true

function Convolution2D:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, kernel_size, kernel_stride, 
         pool_size, pool_stride, transfer, typename
      = xlua.unpack(
      {config},
      'Convolution2D', 
      '[dropout] + SpartialConvolution + SpatialMaxPooling + transfer function',
      {arg='input_size', type='number', req=true,
       help='Number of input channels or colors'},
      {arg='output_size', type='number', req=true,
       help='Number of output channels.'},
      {arg='kernel_size', type='number tuple', req=true,
       help='The size (height, width) of the convolution kernel.'},
      {arg='kernel_stride', type='number tuple',
       help='The stride (height, width) of the convolution. '..
       'Note that depending of the size of your kernel, several (of '..
       'the last) columns or rows of the input image might be lost. '..
       'It is up to the user to add proper padding in images.'..
       'Defaults to {1,1}.'},
      {arg='pool_size', type='number tuple', req=true,
       help='The size (height, width) of the spatial max pooling.'},
      {arg='pool_stride', type='number tuple', req=true,
       help='The stride (height, width) of the spatial max pooling. '..
       'Must be a square (height == width).'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc.'},
      {arg='typename', type='string', default='convolution2d', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._kernel_size = kernel_size
   self._kernel_stride = kernel_stride or {1,1}
   self._pool_size = pool_size
   self._pool_stride = pool_stride
   self._transfer = transfer
   self._conv = nn.SpatialConvolution(
      input_size, output_size, 
      kernel_size[1], kernel_size[2], 
      kernel_stride[1], kernel_stride[2]
   )
   self._pool = nn.SpatialMaxPooling(
      pool_size[1], pool_size[2],
      pool_stride[1], pool_stride[2]
   )
   self._module = nn.Sequential()
   self._module:add(self._conv)
   self._module:add(self._transfer)
   self._module:add(self._pool)
   config.typename = typename
   config.input_view = 'bchw'
   config.output_view = 'bchw'
   config.output = dp.ImageView()
   parent.__init(self, config)
end

function Convolution2D:_forward(carry)
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

function Convolution2D:_backward(carry)
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

function Convolution2D:zeroGradParameters()
   self._conv:zeroGradParameters()
end

function Convolution2D:_type(type)
   self._input_type = type
   self._output_type = type
   if type == 'torch.CudaTensor' 
         and torch.type(self._conv) == 'nn.SpatialConvolution' then
      local conv = nn.SpatialConvolutionCUDA(
         self._input_size, self._output_size, 
         self._kernel_size[1], self._kernel_size[2], 
         self._kernel_stride[1], self._kernel_stride[2]
      )
      self._conv:copy(conv)
      self._pool = nn.SpatialMaxPoolingCUDA(
         self._pool_size[1], self._pool_size[2],
         self._pool_stride[1], self._pool_stride[2]
      )
      self._module = nn.Sequential()
      self._module:add(self._conv)
      self._module:add(self._transfer)
      self._module:add(self._pool)
      self._input_view = 'chwb'
      self._output_view = 'chwb'
   elseif type ~= 'torch.CudaTensor'
         and torch.type(self._conv) == 'nn.SpatialConvolutionCUDA' then
      error"NotImplemented"
      self._conv = nn.SpatialConvolution(
         self._input_size, self._output_size, 
         self._kernel_size[1], self._kernel_size[2], 
         self._kernel_stride[1], self._kernel_stride[2]
      )
      -- TODO: share weights
      self._pool = nn.SpatialMaxPooling(
         self._pool_size[1], self._pool_size[2],
         self._pool_stride[1], self._pool_stride[2]
      )
   end
   if type ~= 'torch.CudaTensor' and not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
   return self
end

function Convolution2D:reset()
   self._conv:reset()
   if self._sparse_init then
      print"Warning : this wont work with SpatialConvolutionCUDA"
      local W = self._conv.weight
      W = W:reshape(W:size(1)*W:size(2)*W:size(3), W:size(4))
      self._sparseReset(W:t())
   end
end

function Convolution2D:maxNorm(max_out_norm, max_in_norm)
   error"Not Implemented"
   -- TODO : max_in_norm is pylearn2's max kernel norm?
   if not self.backwarded then return end
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = params.weight.param
   if max_out_norm then
      -- rows feed into output neurons 
      dp.constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      dp.constrain_norms(max_in_norm, 1, weight)
   end
end

function Convolution2D:share(conv2d, ...)
   assert(conv2d.isConvolution2D)
   return parent.share(self, conv2d, ...)
end

function Convolution2D:paramModule()
   return self._conv
end
