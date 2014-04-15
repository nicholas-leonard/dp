-----------------------------------------------------------------------
--[[ CompositeTensor ]]-- 
-- Composite (design pattern) BaseTensor

--TODO : TableTensor vs ListTensor (for performance reasons)
------------------------------------------------------------------------
local CompositeTensor, parent = torch.class("dp.CompositeTensor", "dp.BaseTensor")
CompositeTensor.isCompositeTensor = true

function CompositeTensor:__init(config)
   local args, components = xlua.unpack(
      {config or {}},
      'CompositeTensor', 
      'Builds a dp.CompositeTensor out of a table of dp.BaseTensors',
      {arg='components', type='table of dp.BaseTensor', 
       help='A list of dp.BaseTensors', req=true}
   )   
   parent.assertInstances(components)
   self._components = components
end

function CompositeTensor:feature()
   -- sort keys to get consistent view
   local keys = _.sort(_.keys(self._components))
   local features = _.map(keys, 
      function(key)
         return self._components[key]:feature()
      end
   )
   -- flatten in case of nested composites
   features = _.flatten(features)
   -- concat features (emptyClones first torch.Tensor of features)
   -- we also save the memory for later use
   self._data = torch.concat(self._data, features)
   return self._data
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

function CompositeTensor:size()
   return table.length(self._components)
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
