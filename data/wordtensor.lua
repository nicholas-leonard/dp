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

function WordTensor:context(tensortype, inplace, contiguous)
   return self:multiclass(tensortype, inplace, contiguous)
end

function WordTensor:feature(tensortype, inplace, contiguous)
   -- when request as features (could be for inputs), use many-hot view
   return self:context(tensortype, inplace, contiguous)
end
