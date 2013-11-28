require 'nn'
require 'torch'

------------------------------------------------------------------------
--[[ Module ]]--
-- Decorates/Adapts a nn.Module to the dp.Model interface
-- A temporary fix until these are implemented into their own 
-- Model subclasses
------------------------------------------------------------------------
local Module, parent = torch.class("dp.Module", "dp.Model")

function Module:__init(config)
   config = config or {}
   local args, module = xlua.unpack(
      {config},
      'Module', 
      'Decorates/Adapts a nn.Module to the dp.Model interface',
      {arg='module', type='nn.Module'}
   )
   self._module = module
   -- typename of this model
   local typename = _.split(torch.typename(module), '.')
   typename = string.lower(typename[#typename]) .. 'Adapter'
   config.typename = config.typename or typename
   parent.__init(self, config)
   -- try to guess if module has parameters
   local params, gradParams = module:parameters()
   assert(not params or #params <= 2, "Error : unknown extra parameters")
   if self._tags.hasParams == nil then
      if (not params) or (#params == 0) then
         self._tags.hasParams = false
      else
         for i, param in ipairs(params) do
            if param:dim() ~= 0 then
               self._tags.hasParams = true
               break
            end
         end
      end
   end
end

function Module:_forward(gstate)
   self.ostate.act = self._module:forward(self.istate.act)
end

function Module:_backward(gstate, scale)
   self.istate.grad 
      = self._module:backward(self.istate.act, self.ostate.grad, scale)
end

function Module:_update(gstate)
   self._module:updateParameters(gstate.learning_rate)
end

function Module:zeroGradParameters()
   self._module:zeroGradParameters()
end

function Module:type(type)
   return self._module:type(type)
end

function Module:reset()
   return self._module:reset()
end

function Module:parameters()
   local params = self._params
   local module = self._module
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


------------------------------------------------------------------------
--[[ Linear ]]--
-- Adapts a nn.Linear to the dp.Model interface
------------------------------------------------------------------------
local Linear, parent = torch.class("dp.Linear", "dp.Module")

function Linear:__init(config)
   config = config or {}
   local args, input_size, output_size, typename = xlua.unpack(
      {config},
      'Linear', 
      'Adapts a nn.Linear to the dp.Model interface',
      {arg='input_size', type='number', req=true,
       help='Number of input neurons'},
      {arg='output_size', type='number', req=true,
       help='Number of output neurons'},
      {arg='typename', type='string', default='linear', 
       help='identifies Model type in reports.'}
   )
   self._input_size = input_size
   self._output_size = output_size
   config.typename = config.typename or typename
   config.module = nn.Linear(input_size, output_size)
   parent.__init(self, config)
end

function Linear:setup(config)
   config.data_view = 'feature'
   parent.setup(self, config)
end

function Linear:maxNorm(max_out_norm, max_in_norm)
   if not self._backwarded then return end
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   local params = self:parameters()
   local weight = param.weight.param
   if max_out_norm then
      -- rows feed into output neurons 
      constrain_norms(max_out_norm, 2, weight)
   end
   if max_in_norm then
      -- cols feed out from input neurons
      constrain_norms(max_in_norm, 1, weight)
   end
end
