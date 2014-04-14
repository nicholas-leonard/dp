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
   parent.assertInstances(components)
   self._components = components
end

-- Returns number of samples
function CompositeTensor:nSample()
   for k,component in self:pairs() do
      return component:nSample()
   end
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


-- return a clone without data 
function CompositeTensor:emptyClone()
   return dp.CompositeTensor{
      components=_.map(self._components,
         function(component)
            return component:emptyClone()
         end
      )
   }
end

-- copy data into existing memory allocated for data
function CompositeTensor:copy(compositetensor)
   assert(compositetensor.isCompositeTensor, 
      "CompositeTensor:copy() error : expecting CompositeTensor")
   for k,v in self:pairs() do
      v:copy(compositetensor:components()[k])
   end
end

function CompositeTensor:components()
   return self._components
end
