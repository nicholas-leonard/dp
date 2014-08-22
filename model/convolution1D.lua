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
      '[dropout] + TemporalConvolution + transfer function + TemporalMaxPooling',
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
   config.output = dp.SequenceView()
   parent.__init(self, config)
end

function Convolution1D:reset()
   self._conv:reset()
   if self._sparse_init then
      self._sparseReset(self._conv.weight:t())
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

-- number of output frames of the convolution1D layer
function Convolution1D:nOutputFrame(nInputFrame)
   assert(torch.type(nInputFrame) == 'number', "Expecting number")
   local nFrame = (nInputFrame - self._kernel_size) / self._kernel_stride + 1
   return (nFrame - self._pool_size) / self._pool_stride + 1
end
