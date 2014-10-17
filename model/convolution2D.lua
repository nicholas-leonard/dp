------------------------------------------------------------------------
--[[ Convolution2D ]]--
-- Spatial Convolution Layer
-- [reduce] + convolution + [max pooling] + transfer function
------------------------------------------------------------------------
local Convolution2D, parent = torch.class("dp.Convolution2D", "dp.Layer")
Convolution2D.isConvolution2D = true

function Convolution2D:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, kernel_size, kernel_stride, 
         pool_size, pool_stride, reduce_size, reduce_stride, 
         transfer, typename = xlua.unpack(
      {config},
      'Convolution2D', 
      '[reduce] + SpartialConvolution + [SpatialMaxPooling] + transfer function',
      {arg='input_size', type='number', req=true,
       help='Number of input channels or colors'},
      {arg='output_size', type='number', req=true,
       help='Number of output channels (number of filters). '..
       'For cuda, this should be a multiple of 8'},
      {arg='kernel_size', type='number | table', default=5,
       help='The size (height=width) of the convolution kernel.'},
      {arg='kernel_stride', type='number | table', default=1,
       help='The stride (height=width) of the convolution. '..
       'Note that depending of the size of your kernel, several (of '..
       'the last) columns or rows of the input image might be lost. '..
       'It is up to the user to add proper padding in images.'},
      {arg='pool_size', type='number | table', default=2,
       help='The size (height=width) of the spatial max pooling. '..
       'A pool_size < 2 disables pooling'},
      {arg='pool_stride', type='number | table', default=2,
       help='The stride (height=width) of the spatial max pooling.'},
      {arg='reduce_size', type='number',
       help='The number of channels (filters) used in an optional '..
       'reduction module preceding the convolution. '..
       'A reduction is a 1x1 convolution. See Inception Model'},
      {arg='reduce_stride', type='number | table', 
       help='The stride of the reduction'},
      {arg='transfer', type='nn.Module',
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc. Defaults to nn.ReLU'},
      {arg='typename', type='string', default='convolution2d', 
       help='identifies Model type in reports.'}
   )
   local function toPair(v)
      return torch.type(v) == 'table' and v or {v,v}
   end
   self._input_size = input_size
   self._output_size = output_size
   self._kernel_size = toPair(kernel_size)
   self._kernel_stride = toPair(kernel_stride)
   self._pool_size = toPair(pool_size)
   self._pool_stride = toPair(pool_stride)
   
   self._param_modules = {}
   self._module = nn.Sequential()
   self._transfer = transfer or nn.ReLU()
   
   if reduce_size then
      -- optional 1x1 reduction/projection module
      self._reduce_size = reduce_size
      self._reduce_stride = toPair(reduce_stride)
      self._reduce = nn.SpatialConvolutionMM(
         input_size, reduce_size, 1, 1, 
         self._reduce_stride[1], self._reduce_stride[2]
      )
      input_size = reduce_size
      self._module:add(self._reduce)
      self._module:add(self._transfer:clone())
      table.insert(self._param_modules, self._reduce)
   end
   
   self._conv = nn.SpatialConvolutionMM(
      input_size, output_size, 
      self._kernel_size[1], self._kernel_size[2], 
      self._kernel_stride[1], self._kernel_stride[2]
   )
   table.insert(self._param_modules, self._conv)
   
   self._module:add(self._conv)
   self._module:add(self._transfer)
   
   if self._pool_size[1] >= 1.5 then
      self._pool = nn.SpatialMaxPooling(
         self._pool_size[1], self._pool_size[2],
         self._pool_stride[1], self._pool_stride[2]
      )
      self._module:add(self._pool)
   end
   
   config.typename = typename
   config.input_view = 'bchw'
   config.output_view = 'bchw'
   config.output = dp.ImageView()
   parent.__init(self, config)
end

function Convolution2D:reset()
   self._module:reset()
   if self._sparse_init then
      for i, modula in ipairs(self._param_modules) do
         local W = modula.weight
         self._sparseReset(W:view(W:size(1), -1))
      end
   end
end

function Convolution2D:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   for i, modula in ipairs(self._param_modules) do
      local W = modula.weight
      W = W:view(W:size(1), -1)
      if max_out_norm then
         W:renorm(1, 2, max_out_norm)
      end
      if max_in_norm then
         W:renorm(2, 2, max_in_norm)
      end
   end
end

function Convolution2D:share(conv2d, ...)
   assert(conv2d.isConvolution2D)
   return parent.share(self, conv2d, ...)
end

-- output size of the model (excluding batch dim)
function Convolution2D:outputSize(inputHeight, inputWidth, view)
   local input = torch.Tensor(2, self._input_size, inputHeight, inputWidth)
   local inputView = dp.ImageView('bchw', input)
   -- just propagate this dummy input through to know the output size
   local output = self:forward(inputView, dp.Carry{nSample=2}):forward(view or 'bchw')
   self:zeroStatistics()
   return output:size(2), output:size(3), output:size(4)
end
