-----------------------------------------------------------------------
--[[ Pipeline ]]--
-- Preprocess Composite
-- A Preprocessor that sequentially applies a list 
-- of other Preprocessors.
-----------------------------------------------------------------------
local Pipeline = torch.class("dp.Pipeline", "dp.Preprocess")
Pipeline.isPipeline = true

function Pipeline:__init(items)
   self._items = items or {}
end

function Pipeline:apply(datatensor, can_fit)
   for i, item in ipairs(self._items) do
      item:apply(datatensor, can_fit)
   end
end
