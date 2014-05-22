-----------------------------------------------------------------------
--[[ Pipeline ]]--
-- Preprocess subclass
-- Composite of Preprocesses
-- Sequentially applies a list of Preprocesses.
-----------------------------------------------------------------------
local Pipeline = torch.class("dp.Pipeline", "dp.Preprocess")
Pipeline.isPipeline = true

function Pipeline:__init(items)
   self._items = items or {}
end

function Pipeline:apply(dv, can_fit)
   for i, item in ipairs(self._items) do
      item:apply(dv, can_fit)
   end
end
