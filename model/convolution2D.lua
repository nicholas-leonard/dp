------------------------------------------------------------------------
--[[ Convolution2D ]]--
-- Spatial Convolution Layer
-- [dropout] + convolution + max pooling + transfer function
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
       help='Number of output channels. For cuda, this should be a multiple of 8'},
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
   self._conv = nn.SpatialConvolutionMM(
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

function Convolution2D:reset()
   self._conv:reset()
   if self._sparse_init then
      local W = self._conv.weight
      self._sparseReset(W:view(W:size(1), -1))
   end
end

function Convolution2D:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local W = self._conv.weight
   W = W:view(W:size(1), -1)
   if max_out_norm then
      W:renorm(1, 2, max_out_norm)
   end
   if max_in_norm then
      W:renorm(2, 2, max_in_norm)
   end
end

function Convolution2D:share(conv2d, ...)
   assert(conv2d.isConvolution2D)
   return parent.share(self, conv2d, ...)
end

-- number of output frames (height or width) of the convolution2D layer
function Convolution2D:nOutputFrame(nInputFrame, idx)
   assert(torch.type(nInputFrame) == 'number', "Expecting number")
   assert(torch.type(idx) == 'number', "Expecting number")
   local nFrame = (nInputFrame - self._kernel_size[idx]) / self._kernel_stride[idx] + 1
   nFrame = (nFrame - self._pool_size[idx]) / self._pool_stride[idx] + 1
   return math.floor(nFrame)
end
