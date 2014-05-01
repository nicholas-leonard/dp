------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a transfer function.
-- For a linear transformation, you can use nn.Identity.
-- Works on a DataTensor:feature() view.
------------------------------------------------------------------------
local Neural, parent = torch.class("dp.Neural", "dp.Model")
Neural.isNeural = true

function Neural:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_size, output_size, transfer, dropout, typename, 
         sparse_init, gather_stats = xlua.unpack(
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
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'},
      {arg='sparse_init', type='boolean', default=true,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='gather_stats', type='boolean', default=false,
       help='gather statistics on gradients'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._transfer = transfer
   self._affine = nn.Linear(input_size, output_size)
   self._dropout = dropout
   self._uncuda = false -- TODO: should detect non-cuda modules
   self._sparse_init = sparse_init
   self._gather_stats = gather_stats
   config.typename = typename
   parent.__init(self, config)
   self:reset()
   self._tags.hasParams = true
   self:zeroGradParameters()
   self:checkParams()
end

function Neural:_zeroStatistics()
   if self._gather_stats then
      for param_name, param_table in pairs(self:parameters()) do
         self._stats[param_name] = {
            grad={sum=0, mean=0, min=0, max=0, count=0, std=0}
         }
      end
   end
end

function Neural:checkParams()
   for param_name,param_table in pairs(self:parameters()) do
      for k,v in pairs(param_table) do
         assert(not _.isNaN(v:sum()), 
                "NaN Error for " .. k .. " " .. param_name)
      end
   end
end

function Neural:_forward(carry)
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

function Neural:_accept(visitor)
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

function Neural:zeroGradParameters()
   self._affine:zeroGradParameters()
end

function Neural:type(type)
   self._affine:type(type)
   if not self._uncuda then
      self._transfer:type(type)
   end
   if self._dropout then
      self._dropout:type(type)
   end
end

function Neural:reset()
   self._affine:reset()
   if self._sparse_init then
      self._sparseReset(self:parameters().weight.param)
   end
end

-- do not use this to change the type of parameters.
function Neural:parameters()
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

function Neural:maxNorm(max_out_norm, max_in_norm)
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

function Neural:share(neural, ...)
   assert(neural.isNeural)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self._affine[v] ~= nil then
         self._affine[v]:set(neural._affine[v])
      end
   end
   return self      
end

function Neural:sharedClone()
   local clone = self:clone()
   return self:share(clone, 'weight', 'bias')
end

function Neural:report()
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


-- static method for initializing weights matrices
-- first dim is for outputs, second is for inputs
function Neural._sparseReset(W, stdev)
   assert(W:dim() == 2, 
      "Model.sparseInit requires a tensor with two dims at arg 1")
   stdev = stdev or 1
   W:zero()
   local output_size, input_size = W:size(1), W:size(2)
   local sparse_init = math.min(math.ceil(input_size/2), 15)
   -- for each output unit:
   for i = 1, output_size do
      -- initialize self.sparse_init input weights:
      for j = 1, sparse_init do
         local idx = math.ceil(math.random() * input_size)
         while W[{i, idx}] ~= 0 do
            idx = math.ceil(math.random() * input_size)
         end
         W[{i, idx}] = torch.normal(0, stdev)
      end
   end
end
