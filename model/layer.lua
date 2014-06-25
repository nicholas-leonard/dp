------------------------------------------------------------------------
--[[ Layer ]]--
-- Abstract class
-- Opposite of Container. An indivisable component.
------------------------------------------------------------------------
local Layer, parent = torch.class("dp.Layer", "dp.Model")
Layer.isLayer = true

function Layer:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, input_view, output_view, output, dropout, sparse_init 
      = xlua.unpack(
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
       '"Deep learning via Hessian-free optimization"'}
   )
   self:inputView(input_view)
   self:outputView(output_view)
   self.output = output
   self._dropout = dropout
   self._sparse_init = sparse_init
   parent.__init(self, config)
   self:reset()
   self._tags.hasParams = true
   self:zeroGradParameters()
   self:checkParams()
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

-- this should return a parameterized module
-- TODO: support a table of modules
function Layer:paramModule()
   error"Not Implemented"
end

function Layer:checkParams()
   local params = self:parameters()
   for k,param in pairs(params) do
      if _.isNaN(param:sum()) then
         error("NaN Error for param at index" ..k)
      end
   end
end

function Layer:zeroGradParameters()
   self:paramModule():zeroGradParameters()
end

function Layer:reset()
   self:paramModule():reset()
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
   return self:paramModule():parameters()
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
            param:renorm(2, 2, max_out_norm)
         end
      end
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
