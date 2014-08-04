------------------------------------------------------------------------
--[[ Layer ]]--
-- Abstract class
-- Opposite of Container. An indivisable component.
------------------------------------------------------------------------
local Layer, parent = torch.class("dp.Layer", "dp.Model")
Layer.isLayer = true

function Layer:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_view, output_view, output, dropout, sparse_init,
      acc_update = xlua.unpack(
      {config},
      'Layer', 
      'An abstract parameterized layer.',
      {arg='input_view', type='string', req=true,
       help='view of the input like "bf", "bhwc", etc.'},
      {arg='output_view', type='string', req=true,
       help='view of the output like "bf", "bhwc", etc.'},
      {arg='output', type='dp.View', req=true,
       help='the View used for communicating outputs and gradOutputs'},
      {arg='dropout', type='nn.Dropout', 
       help='applies dropout to the inputs of this model.'},
      {arg='sparse_init', type='boolean', default=true,
       help='sparse initialization of weights. See Martens (2010), '..
       '"Deep learning via Hessian-free optimization"'},
      {arg='acc_update', type='boolean', default=false,
       help='when true, uses the faster accUpdateGradParameters, '..
       'which performs an inplace update (no need for param gradients). '..
       'However, this also means that Momentum, WeightDecay and other '..
       'such gradient modifying Visitors cannot be used.'}
   )
   if not (self._module and self._module.forward) then
      error"self._module (a nn.Module) should be set by child"
   end
   self:inputView(input_view)
   self:outputView(output_view)
   self.output = output
   if dropout then
      self:pushDropout(dropout)
   end
   self._sparse_init = sparse_init
   self._acc_update = acc_update
   parent.__init(self, config)
   self:reset()
   self._tags.hasParams = true
   if acc_update then
      self._tags.accUpdate = true
   end
   self:zeroGradParameters()
   self:checkParams()
end

function Layer:pushDropout(dropout)
   if torch.type(self._module) == 'nn.Sequential' then
      self._module:insert(dropout, 1)
   else
      local mlp = nn.Sequential()
      mlp:add(dropout)
      mlp:add(self._module)
      self._module = mlp
   end
end

-- Get
function Layer:inputAct()
   return self.input:forward(self._input_view, self._input_type)
end

function Layer:outputGrad()
   return self.output:backward(self._output_view, self._output_type)
end

-- Set
function Layer:inputGrad(input_grad)
   self.input:backward(self._input_view, input_grad)
end

function Layer:outputAct(output_act)
   self.output:forward(self._output_view, output_act)
end

function Layer:_forward(carry)
   -- some modules like dropout have a different behavior during 
   -- evaluation vs training :
   if carry.evaluate then 
      self._module:evaluate()
   else
      self._module:training()
   end
   self:outputAct(self._module:forward(self:inputAct()))
   return carry
end

function Layer:_backward(carry)
   local input_grad
   if self._acc_update then 
      input_grad = self._module:updateGradInput(self:inputAct(), self:outputGrad())
   else
      input_grad = self._module:backward(self:inputAct(), self:outputGrad(), self._acc_scale)
   end
   self:inputGrad(input_grad)
   return carry
end

function Layer:updateParameters(lr)
   if self._acc_update then
      self._module:accUpdateGradParameters(self:inputAct(), self:outputGrad(), lr*self._acc_scale)
   else
      self._module:updateParameters(lr)
   end
end

function Layer:checkParams()
   local params = self:parameters()
   for k,param in pairs(params) do
      if _.isNaN(param:sum()) then
         error(self:name().." NaN Error for param at index" ..k)
      end
   end
end

function Layer:zeroGradParameters()
   if not self._acc_update then
      self._module:zeroGradParameters()
   end
end

function Layer:reset()
   self._module:reset()
   if self._sparse_init then
      local params = self:parameters()
      -- Only affects 2D parameters.
      -- Assumes that 2D parameters are aranged (output_dim x input_dim)
      for k,param in pairs(params) do
         if param:dim() == 2 then
            self._sparseReset(param)
         end
      end
   end
end

-- do not use this to change the type of parameters.
function Layer:parameters()
   return self._module:parameters() or {},{}
end

-- Only affects 2D parameters.
-- Assumes that 2D parameters are arranged (output_dim x input_dim)
function Layer:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params, gradParams = self:parameters()
   for k,param in pairs(params) do
      if param:dim() == 2 then
         if max_out_norm then
            -- rows feed into output neurons 
            param:renorm(1, 2, max_out_norm)
         end
         if max_in_norm then
            -- cols feed out from input neurons
            param:renorm(2, 2, max_in_norm)
         end
      end
   end
end

function Layer:share(layer, ...)
   assert(layer.isLayer)
   local arg = {...}
   local module = self._module
   for i,v in ipairs(arg) do
      if module[v] ~= nil then
         module[v]:set(layer._module()[v])
      end
   end
   return self      
end

function Layer:sharedClone()
   local clone = self:clone()
   return clone:share(self, 'weight', 'bias')
end

-- changes the type of internal variables inplace (same as nn)
-- returns self
function Layer:type(new_type)
   if new_type then
      self:_type(new_type)
      self.output:flush() -- this is why we reimplement this method
      self:moduleType(new_type)
      collectgarbage()
   end
   return self
end

function Layer:_type(type)
   self:inputType(type)
   self:outputType(type)
   self._module:type(type)
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
