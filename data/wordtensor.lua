------------------------------------------------------------------------
--[[ WordTensor ]]--
-- Inherits ClassTensor.
-- A DataTensor holding a sequence of words.
------------------------------------------------------------------------
local WordTensor, parent = torch.class("dp.WordTensor", "dp.ClassTensor")
WordTensor.isWordTensor = true

function WordTensor:default()
   return self:context()
end

function WordTensor:classes()
   return self._classes
end

function WordTensor:words()
   return self._classes
end

function WordTensor:context(inplace, contiguous)
   return self:multiclass(inplace, contiguous)
end

function WordTensor:feature(inplace, contiguous)
   -- when request as features (could be for inputs), use many-hot view
   return self:context(inplace, contiguous)
end
