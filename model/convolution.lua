------------------------------------------------------------------------
--[[ Convolution ]]--
-- Uses CUDA 
-- [dropout] + convolution + max pooling + transfer function
-- Works on a ImageTensor:imageCUDA() view.
------------------------------------------------------------------------
local Convolution, parent = torch.class("dp.Convolution", "dp.Model")
Convolution.isConvolution = true

function Convolution:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, kernel_size, kernel_stride, 
         pool_size, pool_stride, transfer, dropout, sparse_init, 
         gather_stats, typename
      = xlua.unpack(
      {config},
      'Convolution', 
      '[dropout] + convolution + max pooling + transfer function',
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
       'Defaults to {1,1}.'}
      {arg='pool_size', type='number tuple', req=true,
       help='The size (height, width) of the spatial max pooling.'},
      {arg='pool_stride', type='number tuple', req=true,
       help='The stride (height, width) of the spatial max pooling. '..
       'Must be a square (height == width).'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, '..
       'nn.ReLU, nn.Identity, etc.'},
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='sparse_init', type='boolean', default=true,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='gather_stats', type='boolean', default=false,
       help='gather statistics on gradients'},
      {arg='typename', type='string', default='convolution', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._kernel_size = kernel_size
   self._kernel_stride = kernel_stride or {1,1}
   self._pool_size = pool_size
   self._pool_stride = pool_stride
   self._transfer = transfer
   self._conv = nn.SpatialConvolutionCUDA(
      input_size, output_size, 
      kernel_size[1], kernel_size[2], 
      kernel_stride[1], kernel_stride[2]
   )
   self._pool = nn.SpatialMaxPoolingCUDA(
      pool_size[1], pool_size[2],
      pool_stride[1], pool_stride[2]
   )
   self._dropout = dropout
   self._sparse_init = sparse_init
   self._gather_stats = gather_stats
   config.typename = typename
   parent.__init(self, config)
   self._sparse_init = sparse_init
   if sparse_init then
      self:sparseReset()
   end
   self._tags.hasParams = true
   self:zeroGradParameters()
   self:checkParams()
end

function Convolution:_zeroStatistics()
   if self._gather_stats then
      for param_name, param_table in pairs(self:parameters()) do
         self._stats[param_name] = {
            grad={sum=0, mean=0, min=0, max=0, count=0, std=0}
         }
      end
   end
end

function Convolution:checkParams()
   for param_name,param_table in pairs(self:parameters()) do
      for k,v in pairs(param_table) do
         assert(not _.isNaN(v:sum()), 
                "NaN Error for " .. k .. " " .. param_name)
      end
   end
end

function Convolution:_forward(carry)
   local activation = self.input.act:feature()
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

function Convolution:_backward(carry)
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

function Convolution:_accept(visitor)
   if self._gather_stats then
      local params = self:parameters()
      for param_name, param_table in pairs(self:parameters()) do
         local grad = torch.abs(param_table.grad:double())
         local grad_stats = self._stats[param_name].grad 
         grad_stats.sum = grad_stats.sum + grad:sum()
         grad_stats.min = grad_stats.min + grad:min()
         grad_stats.max = grad_stats.max + grad:max()
         grad_stats.mean = grad_stats.mean + grad:mean()
         grad_stats.std = grad_stats.std + grad:std()
         grad_stats.count = grad_stats.count + 1
      end
   end
   parent._accept(self, visitor)
end

function Convolution:zeroGradParameters()
   self._affine:zeroGradParameters()
end

function Convolution:type(type)
   self._affine:type(type)
   if not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
end

function Convolution:reset()
   self._conv:reset()
   if self._sparse_init then
      local W = self:parameters().weight.param
      W = W:reshape(W:size(1)*W:size(2)*W:size(3), W:size(4))
      self._sparseReset(W:t())
   end
end

-- do not use this to change the type of parameters.
function Convolution:parameters()
   local params = {}
   local module = self._affine
   if module.weight and module.weight:dim() ~= 0 then
      params.weight = { param=module.weight, grad=module.gradWeight }
   end
   if module.bias and module.bias:dim() ~= 0 then
      params.bias = { param=module.bias, grad=module.gradBias }
   end
   return params
end

function Convolution:maxNorm(max_out_norm, max_in_norm)
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

function Convolution:share(neural, ...)
   assert(neural.isConvolution)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self._affine[v] ~= nil then
         self._affine[v]:set(neural._affine[v])
      end
   end
   return self      
end

function Convolution:sharedClone()
   local clone = self:clone()
   return self:share(clone, 'weight', 'bias')
end

function Convolution:report()
   local report = parent.report(self) or {}
   if self._gather_stats then
      for param_name, param_table in pairs(self:parameters()) do
         local param_stats = self._stats[param_name]
         if param_stats and param_stats.grad and param_stats.grad.count > 0 then
            local grad = param_stats.grad
            local param_report = self._report[param_name] or {}
            local count = grad.count
            local grad_report = {
                  sum=grad.sum/count, mean=grad.mean/count,
                  min=grad.min/count, max=grad.max/count,
                  std=grad.std/count, count=grad.count
            }
            self._report[param_name] = {grad=grad_report}
         end
      end
   end
   return table.merge(report, self._report)
end
