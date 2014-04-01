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

function Binarize:apply(datatensor)
   local data = datatensor:data()
   inputs[data:lt(self._threshold)] = 0;
   inputs[data:ge(self._threshold)] = 1;
   datatensor:setData(data)
end
