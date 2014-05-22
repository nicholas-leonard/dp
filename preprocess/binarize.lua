-----------------------------------------------------------------------
--[[ Binarize ]]-- 
-- A Preprocessor that sets to 0 any pixel strictly below the 
-- threshold, sets to 1 those above (or equal) to the threshold.
-----------------------------------------------------------------------
local Binarize = torch.class("dp.Binarize", "dp.Preprocess")
Binarize.isBinarize = true

function Binarize:__init(threshold)
   self._threshold = threshold
end

function Binarize:apply(dv)
   local data = dv:input()
   data[data:lt(self._threshold)] = 0;
   data[data:ge(self._threshold)] = 1;
   dv:input(data)
end
