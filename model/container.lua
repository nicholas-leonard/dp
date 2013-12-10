------------------------------------------------------------------------
--[[ dp.Container ]]--
-- Model Composite of Model Components
------------------------------------------------------------------------
local Container, parent = torch.class("dp.Container", "dp.Model")
Container.isContainer = true

function Container:__init(config)
   self._models = {}
   parent.__init(self, config)
end

function Container:type(type)
   -- find submodels in classic containers 'models'
   if not _.isEmpty(self._models) then
      for i, model in ipairs(self._models) do
         model:type(type)
      end
   end
end
