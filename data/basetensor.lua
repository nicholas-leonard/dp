-----------------------------------------------------------------------
--[[ BaseTensor ]]-- 
-- Abstract class
-- Adapter (design pattern) for torch.Tensor 
------------------------------------------------------------------------
local BaseTensor = torch.class("dp.BaseTensor")
BaseTensor.isBaseTensor = true

function BaseTensor:feature(tensortype, inplace, contiguous)
   -- When true, makes stored data a contiguous view for future use :
   inplace = inplace or true
   -- When true makes sure the returned tensor contiguous. 
   -- Only considered when inplace is false, since inplace
   -- implicitly makes the returned tensor contiguous :
   contiguous = contiguous or false
   return self:_feature(tensortype, inplace, contiguous)
end

-- Returns number of samples
function BaseTensor:nSample()
   error"Not Implemented"
end

-- Returns a subet of data. 
-- If dt is provided, copy indexed elements into its existing memory.
-- However, use the metadata found in self.
-- Providing dt is faster (no memory allocation).
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
function BaseTensor.transpose(axis, new_dim, axes, size, data)
   -- copy
   axes = _.omit(axes)
   local current_dim = _.indexOf(axes, axis)
   if current_dim == 0 then
      error("Axis " .. axis .. 'is not in axes ' .. axes)
   end
   if new_dim < 0 then
      new_dim = #axes + 1 - new_dim
   end
   if current_dim ~= new_dim then
      axes[current_dim] = axes[new_dim]
      axes[new_dim] = axis
      local new_size = size[new_dim]
      size[new_dim] = size[current_dim]
      size[current_dim] = new_size
      if data then
         data = data:transpose(new_dim, current_dim)
      end
   end
   return axes, size, data
end

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
