------------------------------------------------------------------------
--[[ Module ]]--
-- Decorates/Adapts a nn.Module to the dp.Model interface
-- A temporary fix until these are implemented into their own 
-- Model subclasses
------------------------------------------------------------------------
local Module, parent = torch.class("dp.Module", "dp.Model")
Module.isModule = true

function Module:__init(config)
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

function Module:_forward(carry)
   self:outputAct(self._module:forward(self:inputAct()))
end

function Module:_backward(carry)
   local act, grad = self:inputAct(), self:outputGrad()
   self:inputGrad(self._module:backward(act, grad, carry.scale))
end

function Module:zeroGradParameters()
   self._module:zeroGradParameters()
end

function Module:_type(type)
   self:inputType(type)
   self:outputType(type)
   self._module:type(type)
end

function Module:reset()
   return self._module:reset()
end

-- use at your own risk. 
-- dp.Visitors expect each param/paramGrad to be identified by a 
-- unique key that stays the same from batch to batch.
-- This wont be true for modules like nnx.SoftMaxTree or nnx.LookupTable
function Module:parameters()
   return self._module:parameters()
end
