-----------------------------------------------------------------------
--[[ CompositeTensor ]]-- 
-- Composite (design pattern) BaseTensor
------------------------------------------------------------------------
local CompositeTensor, parent = torch.class("dp.CompositeTensor", "dp.BaseTensor")
CompositeTensor.isCompositeTensor = true

function CompositeTensor:__init(config)
   local args, components = xlua.unpack(
      {config or {}},
      'CompositeTensor', 
      'Builds a dp.CompositeTensor out of torch.Tensor data.',
      {arg='components', type='list of dp.BaseTensor', 
       help='A list of dp.BaseTensors', req=true}
   )   
   dp.DataTensor.assertInstances(inputs)
   self._components = components
end

-- Returns number of samples
function CompositeTensor:nSample()
   assert(#self:storedAxes() == self:storedSize():nElement(),
      "Error : unequal number of axis for size and axes")
   return self:storedSize()[self:b()]
end

--Decorator/Adapter for torch.Tensor
--Returns a batch of data. 
--Note that the batch uses different storage (because of :index())
function CompositeTensor:index(indices)
   return _.map(self._components, 
      function(component) 
         return component:index(indices)
      end
   )
end

function CompositeTensor:sub(start, stop)
   return _.map(self._components, 
      function(component) 
         return component:sub(start, stop)
      end
   )
end

-- return iterator over components
function CompositeTensor:pairs()
   return pairs(self._components)
end
