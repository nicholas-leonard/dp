------------------------------------------------------------------------
--[[ Module ]]--
-- Decorates/Adapts a nn.Module to the dp.Model interface
-- A temporary fix until these are implemented into their own 
-- Model subclasses
------------------------------------------------------------------------
local Module, parent = torch.class("dp.Module", "dp.Model")
Module.isModule = true

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
   local typename = _.split(torch.typename(module), '[.]')
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

function Module:_forward(cstate)
   self.ostate.act = self._module:forward(self.istate.act)
end

function Module:_backward(cstate)
   self.istate.grad = self._module:backward(
      self.istate.act, 
      self.ostate.grad, 
      self.gstate.scale
   )
end

function Module:zeroGradParameters()
   self._module:zeroGradParameters()
end

function Module:type(type)
   self._module:type(type)
   return parent.type(self, type)
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
