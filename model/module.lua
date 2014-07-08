------------------------------------------------------------------------
--[[ Module ]]--
-- Decorates/Adapts a nn.Module to the dp.Model interface
-- A temporary fix until these are implemented into their own 
-- Model subclasses

-- For all intents in purposes, this Module should do a great job 
-- of integrating your existing Modules into dp. Just wrap them using 
-- thie Model. However, some dp.Visitors expect 
-- each param/gradParam to be identified by a 
-- unique key that stays the same from batch to batch.
-- This wont be true for modules like nnx.SoftMaxTree or nnx.LookupTable
-- so be careful.
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
