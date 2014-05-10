------------------------------------------------------------------------
--[[ Layer ]]--
-- Abstract class
-- Opposite of Container. An indivisable component.
------------------------------------------------------------------------
local Layer, parent = torch.class("dp.Layer", "dp.Model")
Layer.isLayer = true

function Layer:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, dropout, sparse_init, gather_stats = xlua.unpack(
      {config},
      'Layer', 
      'An abstract parameterized layer.',
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='sparse_init', type='boolean', default=true,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='gather_stats', type='boolean', default=false,
       help='gather statistics on gradients and such.'}
   )
   self._dropout = dropout
   self._sparse_init = sparse_init
   self._gather_stats = gather_stats
   parent.__init(self, config)
   self:reset()
   self._tags.hasParams = true
   self:zeroGradParameters()
   self:checkParams()
end

function Layer:inputAct()
   return self.input.act:feature(self._input_type)
end

function Layer:inputGrad(input_grad)
   if input_grad then
      assert(torch.isTensor(input_grad))
      self.input.grad = self.input.act:shallowClone()
      self.input.grad:setData(input_grad)
      return
   end
   return self.input.grad:feature(self._input_type)
end

function Layer:outputAct(output_act)
   if output_act then
      -- wrap torch.Tensor in a dp.DataTensor
      assert(torch.isTensor(output_act))
      self.output.act = dp.DataTensor{data=output_act}
      return
   end
   return self.output.act:feature(self._output_type)
end

function Layer:outputGrad()
   return self.output.grad:feature(self._output_type)
end

-- this should return a parameterized module
-- TODO: support a table of modules
function Layer:paramModule()
   error"Not Implemented"
end

function Layer:_zeroStatistics()
   if self._gather_stats then
      for param_name, param_table in pairs(self:parameters()) do
         self._stats[param_name] = {
            grad={sum=0, mean=0, min=0, max=0, count=0, std=0}
         }
      end
   end
end

function Layer:checkParams()
   for param_name,param_table in pairs(self:parameters()) do
      for k,v in pairs(param_table) do
         assert(not _.isNaN(v:sum()), 
                "NaN Error for " .. k .. " " .. param_name)
      end
   end
end

function Layer:_accept(visitor)
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

function Layer:zeroGradParameters()
   self:paramModule():zeroGradParameters()
end

function Layer:reset()
   self:paramModule():reset()
   if self._sparse_init then
      self._sparseReset(self:parameters().weight.param)
   end
end

-- do not use this to change the type of parameters.
function Layer:parameters()
   local params = {}
   local module = self:paramModule()
   if module.weight and module.weight:dim() ~= 0 then
      params.weight = { param=module.weight, grad=module.gradWeight }
   end
   if module.bias and module.bias:dim() ~= 0 then
      params.bias = { param=module.bias, grad=module.gradBias }
   end
   return params
end

function Layer:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = params.weight.param
   assert(weight:dim() == 2, "Only works with two dims. "..
      "Re-implement your own version in the subclass that calls this.")
   if max_out_norm then
      -- rows feed into output neurons 
      dp.constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      dp.constrain_norms(max_in_norm, 1, weight)
   end
end

function Layer:share(layer, ...)
   assert(layer.isLayer)
   local arg = {...}
   local module = self:paramModule()
   for i,v in ipairs(arg) do
      if module[v] ~= nil then
         module[v]:set(layer:paramModule()[v])
      end
   end
   return self      
end

function Layer:sharedClone()
   local clone = self:clone()
   return self:share(clone, 'weight', 'bias')
end

function Layer:report()
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
function Layer._sparseReset(W, stdev)
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
