------------------------------------------------------------------------
--[[ Neural ]]--
-- An affine transformation followed by a non-linearity
------------------------------------------------------------------------
local Neural, parent = torch.class("dp.Neural", "dp.Model")
Neural.isNeural = true

function Neural:__init(config)
   config = config or {}
   local args, input_size, output_size, transfer, dropout, typename 
      = xlua.unpack(
      {config},
      'Neural', 
      'An affine transformation followed by a non-linearity',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='transfer', type='nn.Module', req=true,
       help='a transfer function like nn.Tanh, nn.Sigmoid, nn.ReLU'},
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='typename', type='string', default='neural', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   self._transfer = transfer
   config.typename = config.typename or typename
   self._affine = nn.Linear(input_size, output_size)
   self._dropout = dropout
   parent.__init(self, config)
   self._tags.hasParams = true
   self._uncuda = (torch.typename(self._transfer) == 'nn.SoftMax')
   self:zeroGradParameters()
   self:checkParams()
   self:zeroStatistics()
end
--[[
function Neural:sparseInit(stdev)
   W = np.zeros(shape)
   -- for each output unit:
   for i in xrange(output_dim):
      assert self.sparse_init <= input_dim
      # initialize self.sparse_init input weights:
      for j in xrange(self.sparse_init):
          idx = rng.randint(0, input_dim)
          while W[idx, i] != 0 or mask_rejects(idx, i):
              idx = rng.randint(0, input_dim)
          W[idx, i] = rng.randn()
   W *= self.sparse_stdev
   return W
end]]--

function Neural:zeroStatistics()
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
      self._stats[param_name] = {
         grad={sum=0, mean=0, min=0, max=0, count=0, std=0}
      }
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

function Neural:setup(config)
   config.data_view = 'feature'
   parent.setup(self, config)
end

function Neural:_forward(cstate)
   local activation = self.istate.act
   if self._dropout then
      -- dropout has a different behavior during evaluation vs training
      self._dropout.train = (not self.gstate.evaluate)
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
   self.ostate.act = self._transfer:forward(activation)
end

function Neural:_backward(cstate)
   local scale = cstate.scale or self.gstate.scale
   self._report.scale = scale
   local input_act = self.mvstate.affineAct
   local output_grad = self.ostate.grad
   output_grad = self._transfer:backward(input_act, output_grad, scale)
   if self._recuda then
      output_grad = output_grad:cuda()
   end
   self.mvstate.affineGrad = output_grad
   input_act = self.mvstate.dropoutAct or self.istate.act
   output_grad = self._affine:backward(input_act, output_grad, scale)
   if self._dropout then
      self.mvstate.dropoutGrad = output_grad
      input_act = self.istate.act
      output_grad = self._dropout:backward(input_act,output_grad,scale)
   end
   self.istate.grad = output_grad
end

function Neural:_accept(visitor)
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
   return parent.type(self, type)
end

function Neural:reset()
   return self._affine:reset()
end

-- TODO move this to __init() and test it!
function Neural:parameters()
   local params = self._params
   local module = self._affine
   if module.weight and module.weight:dim() ~= 0 then
      if not params.weight then
         params.weight = {}
      end
      params.weight.param=module.weight
      params.weight.grad=module.gradWeight
   end
   if module.bias and module.bias:dim() ~= 0 then
      if not params.bias then
         params.bias = {}
      end
      params.bias.param=module.bias
      params.bias.grad=module.gradBias
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

function Neural:report()
   local report = parent.report(self) or {}
   self:zeroStatistics()
   return table.merge(report, self._report)
end
