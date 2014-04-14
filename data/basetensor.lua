-----------------------------------------------------------------------
--[[ BaseTensor ]]-- 
-- Abstract class
-- Adapter (design pattern) for torch.Tensor 
------------------------------------------------------------------------
local BaseTensor = torch.class("dp.BaseTensor")
BaseTensor.isBaseTensor = true

-- Returns number of samples
function BaseTensor:nSample()
   error"Not Implemented"
end

--Decorator/Adapter for torch.Tensor
--Returns a batch of data. 
--Note that the batch uses different storage (because of :index())
function BaseTensor:index(indices)
   error"Not Implemented"
end

function BaseTensor:sub(start, stop)
   error"Not Implemented"
end

-- return iterator over components
function BaseTensor:pairs()
   error"Not Implemented"
end
