--TODO : TableTensor vs ListTensor (for performance reasons)

-----------------------------------------------------------------------
--[[ CompositeTensor ]]-- 
-- Composite (design pattern) BaseTensor
------------------------------------------------------------------------
local CompositeTensor, parent = torch.class("dp.CompositeTensor", "dp.BaseTensor")
CompositeTensor.isCompositeTensor = true

function CompositeTensor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, components = xlua.unpack(
      {config},
      'CompositeTensor', 
      'Builds a dp.CompositeTensor out of a table of dp.BaseTensors',
      {arg='components', type='table of dp.BaseTensor', 
       help='A list of dp.BaseTensors', req=true}
   )   
   parent.assertInstances(components)
   self._components = components
end

function CompositeTensor:_feature(inplace, contiguous)
   -- sort keys to get consistent view
   local keys = _.sort(_.keys(self._components))
   local features = _.map(keys, 
      function(i, key)
         local component = self._components[key]
         return component:feature(inplace, contiguous)
      end
   )
   -- flatten in case of nested composites
   features = _.flatten(features)
   -- concat features (protoClone's first torch.Tensor of features)
   -- we also save the memory for later use
   self._data = torch.concat(self._data, features, 2)
   return self._data
end

-- Returns number of samples
function CompositeTensor:nSample()
   for k,component in self:pairs() do
      return component:nSample()
   end
end

function CompositeTensor:index(dt, indices)
   return _.map(self._components, 
      function(key, component) 
         return component:index(dt, indices)
      end
   )
end

function CompositeTensor:sub(start, stop)
   return _.map(self._components, 
      function(key, component) 
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

-- return a clone with self's metadata initialized with some data 
function CompositeTensor:featureClone(data)
   local components = {}
   local start = 1
   for i,k in ipairs(_.sort(_.keys(self._components))) do
      local component = self._components[k]
      local size = component:feature(false):size(2)
      local clone = component:featureClone(data:narrow(2, start, size))
      components[k] = clone
   end
   return dp.CompositeTensor{components=components}
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
