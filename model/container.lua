------------------------------------------------------------------------
--[[ dp.Container ]]--
-- Model Composite of Model Components
------------------------------------------------------------------------
local Container, parent = torch.class("dp.Container", "dp.Model")
Container.isContainer = true

function Container:__init(config)
   config = config or {}
   local args, models = xlua.unpack(
      {config},
      'Sequential', nil,
      {arg='models', type='table', help='a table of models'}
   )
   self._models = {}      
   parent.__init(self, config)
   if models then
      self:extend(models)
   end
end

function Container:extend(models)
   for model_idx, model in pairs(models) do
      self:add(model)
   end
end

function Container:add(model)
   table.insert(self._models, model)
end

function Container:type(type)
   -- find submodels in classic containers 'models'
   if not _.isEmpty(self._models) then
      for i, model in ipairs(self._models) do
         model:type(type)
      end
   end
end
