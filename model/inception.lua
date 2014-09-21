------------------------------------------------------------------------
--[[ Inception ]]--
-- A layer of parallel columns where each column is a combination 
-- of reduction, max-pooling and convolutions (in different orders).
-- Based on the Going Deeper with Convolutions paper :
-- http://arxiv.org/pdf/1409.4842v1.pdf which is also our source for 
-- the "column" and "reduce" terminology. 
------------------------------------------------------------------------
local Inception, parent = torch.class("dp.Inception", "dp.Layer")
Inception.isInception = true

function Inception:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, reduce_size, reduce_stride, 
         kernel_size, kernel_stride, pool_size, pool_stride, 
         transfer, output_pool, typename
      = xlua.unpack(
      {config},
      'Inception', 
      'Uses n+2 parallel "columns". The original paper uses 2+2 where'..
      'the first two are (but there could be more than two): \n'..
      '1x1 conv (reduce) -> relu -> 5x5 conv -> relu \n'..
      '1x1 conv (reduce) -> relu -> 3x3 conv -> relu \n'..,
      'and where the other two are : \n'
      '3x3 maxpool -> 1x1 conv (reduce/project) -> relu \n'..,
      '1x1 conv (reduce) -> relu. \n'
      'This Model allows the first group of columns to be of any '..
      'number while the last group consist of exactly two columns.'..
      'The 1x1 conv are used to reduce the number of input channels '..
      ' (or filters) such that the capacity of the network doesnt '..
      'explode. We refer to these here has "reduce". Since each '..
      'column seems to have one and only one reduce, their initial '..
      'configuration options are specified in lists of n+2 elements.'
      {arg='input_size', type='number', req=true,
       help='Number of input channels or colors'},
      {arg='output_size', type='table', req=true,
       help='Number of filters in the non-1x1 convolution '..
       'kernel sizes, e.g. {32,48}'},
      {arg='reduce_size', type='table', req=true,
       help='Number of filters in the 1x1 convolutions (reduction) '..
       'used in each column, e.g. {48,64,32,32}. The last 2 are '..
       'used respectively for the max pooling (projection) column '..
       '(the last column in the paper) and the column that has '..
       'nothing but a 1x1 conv (the first column in the paper).'..
       'This table should have two elements more than the output_size'},
      {arg='reduce_stride', type='table', 
       help='the strides of the 1x1 (reduction) convolutions. '..
       'Defaults to {1,1,1,..}'},
      {arg='kernel_size', type='table',
       help='The size (height=width) of the non-1x1 convolution '..
       'kernels. Defaults to {5,3}, i.e. 5x5 and 3x3'},
      {arg='kernel_stride', type='table',
       help='The stride (height=width) of the convolution. '..
       'Note that depending of the size of your kernel, several (of '..
       'the last) columns or rows of the input image might be lost. '..
       'It is up to the user to add proper padding in images.'..
       'Defaults to {1,1}.'},
      {arg='pool_size', type='number', default=3,
       help='The size (height=width) of the spatial max pooling used '..
       'in the next-to-last column.'},
      {arg='pool_stride', type='number', default=1, 
       help='The stride (height=width) of the spatial max pooling.'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc. It is used after each reduction '..
       '(1x1 convolution) and convolution'},
      {arg='output_pool', type='nn.Module',
       help='an optional nn.Module used at the output of the Model. '..
       'In the original paper, some Inception models have an '..
       'additional max-pooling layer at the output. This is were '..
       'it could be specified'},
      {arg='typename', type='string', default='inception', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._reduce_size = reduce_size
   self._reduce_stride = reduce_stride or {}
   self._kernel_size = kernel_size or {5,3}
   self._kernel_stride = kernel_stride or {1,1}
   self._pool_size = pool_size
   self._pool_stride = pool_stride
   self._transfer = transfer
   self._output_pool = output_pool
   self._depth_concat = nn.ConcatTable()
   
   -- 1x1 conv (reduce) -> 3x3 conv
   -- 1x1 conv (reduce) -> 5x5 conv
   -- ...
   for i, kernel_size in ipairs(self._kernel_size) do
      local mlp = nn.Sequential()
      -- 1x1 conv
      local reduce = nn.SpatialConvolutionMM(
         input_size, reduce_size[i], 1, 1, reduce_stride[i], reduce_stride[i]
      )
      mlp:add(reduce)
      mlp:add(transfer:clone())
      -- nxn conv
      local conv = nn.SpatialConvolutionMM(
         reduce_size[i], kernel_size, kernel_size, kernel_stride[i], kernel_stride[i]
      )
      mlp:add(conv)
      mlp:add(transfer:clone())
      self._depth_concat:add(mlp)
   end
   
   -- 3x3 max pool -> 1x1 conv
   local mlp = nn.Sequential()
   local maxPool = nn.SpatialMaxPooling(pool_size, pool_size, pool_stride, pool_stride)
   mlp:add(maxPool)
   -- not sure if transfer should go here? mlp:add(transfer:clone())
   local i = #(self._kernel_size) + 1
   local reduce = nn.SpatialConvolutionMM(
      input_size, reduce_size[i], 1, 1, reduce_stride[i], reduce_stride[i]
   )
   mlp:add(reduce)
   mlp:add(transfer:clone())
   self._depth_concat:add(mlp)
      
   -- reduce: 1x1 conv (channel-wise pooling)
   local mlp = nn.Sequential()
   i = i + 1
   local reduce = nn.SpatialConvolutionMM(
      input_size, reduce_size[i], 1, 1, reduce_stride[i], reduce_stride[i]
   )
   mlp:add(reduce)
   mlp:add(transfer:clone())
   self._depth_concat:add(mlp)
   
   -- optional output max-pooling
   self._module = self._depth_concat
   if self._output_pool then
      local mlp = nn.Sequential()
      mlp:add(self._depth_concat)
      mlp:add(self._output_pool)
      self._module = mlp
   end
   
   self._module:add(self._conv)
   self._module:add(self._transfer)
   self._module:add(self._pool)
   config.typename = typename
   config.input_view = 'bchw'
   config.output_view = 'bchw'
   config.output = dp.ImageView()
   parent.__init(self, config)
end

