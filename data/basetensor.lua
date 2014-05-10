-----------------------------------------------------------------------
--[[ BaseTensor ]]-- 
-- Abstract class
-- Adapter (design pattern) for torch.Tensor 
------------------------------------------------------------------------
local BaseTensor = torch.class("dp.BaseTensor")
BaseTensor.isBaseTensor = true

function BaseTensor:feature(tensortype, inplace, contiguous)
   error"Not Implemented"
end

-- this should return the most expanded view of the tensor
-- or at least the one with the most information
-- and which is easiest to return and construct from
function BaseTensor:default(tensortype, inplace, contiguous)
   return self:feature(tensortype, inplace, contiguous)
end

-- Returns number of samples
function BaseTensor:nSample()
   error"Not Implemented"
end

-- Returns a subet of data. 
-- If dt is provided, copy indexed elements into its existing memory.
-- However, use the metadata found in self.
-- Providing dt is faster (no memory allocation).
-- TODO actually reuse the same dt (not just its torch.Tensor)
function BaseTensor:index(dt, indices)
   error"Not Implemented"
end

function BaseTensor:sub(start, stop)
   error"Not Implemented"
end

-- return iterator over components
function BaseTensor:pairs()
   error"Not Implemented"
end

-- copy data into existing memory allocated for data
function BaseTensor:copy(basetensor)
   error"Not Implemented"
end

function BaseTensor:shallowClone(config)
   error"Not Implemented"
end

-- DEPRECATED
-- return a clone with self's metadata initialized with some data 
function BaseTensor:featureClone(data)
   error"Not Implemented"
end

-- Changes the internal type of the data.
-- different behavior than torch.Tensor:type() (torch returns a tensor
-- with new type and keeps old)
function BaseTensor:type(type)
   error"Not Implemented"
end

function BaseTensor:float()
   return self:type('torch.FloatTensor')
end

function BaseTensor:double()
   return self:type('torch.DoubleTensor')
end

function BaseTensor:cuda()
   return self:type('torch.CudaTensor')
end

---- static methods ----
--returns true if all indices in obj_table are instances of BaseTensor
--else return false and index of first non-element
function BaseTensor.areInstances(obj_table)
   local map = _.values(
      _.map(obj_table, 
         function(key, obj)
            return obj.isBaseTensor
         end
      )
   )
   return _.all(map, function(k,v) return v; end), _.indexOf(map, false)
end

function BaseTensor.assertInstances(obj_table)
   local areInstances, index = BaseTensor.areInstances(obj_table)
   index = index or 0
   assert(areInstances, "Error : object at index " .. index .. 
      " is of wrong type. Expecting type dp.DataTensor.")
end
