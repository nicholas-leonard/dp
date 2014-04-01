-- TODO:
--- Flatten images only (define permitted conversions)
--- Allow construction from existing DataTensor.
--- Transpose expanded_size 'b' dims when data's 'b' is transposed
--- Default axes depends on size of sizes, or size of data.
--- One-hot encoding
--- Adapt doc to specific tensors
--- Update doc (sizes and axis can be one dim less, sizes can be number)
--- Make private members (_memberName)
--- print sizes horizontally
--- :b() should be optimized. You set it via a function. setB()
--- type of tensor (:cuda(), :double(), etc. set in constructor)

local DataTensor = torch.class("dp.DataTensor")
DataTensor.isDataTensor = true

--returns true if all indices in obj_table are instances of DataTensor
--else return false and index of first non-element
function DataTensor.areInstances(obj_table)
   local map = _.map(obj_table, function(obj) return obj.isDataTensor end)
   return _.all(map), _.indexOf(map, false)
end

function DataTensor.assertInstances(obj_table)
   local areInstances, index = DataTensor.areInstances(obj_table)
   assert(areInstances, "Error : object at index " .. index .. 
      " is of wrong type. Expecting type dp.DataTensor.")
end

function DataTensor:__init(...)
   local args, sizes
   args, self._data, self.axes, sizes
      = xlua.unpack(
      {... or {}},
      'DataTensor', 
      [[Constuctor. Builds a data.DataTensor out of torch.Tensor data.
      A DataTensor can be used to convert data into new axes formats 
      using torch.Tensor:resize, :transpose, :contiguous. The 
      conversions may be done in-place(default), or may be simply 
      returned using the conversion methods (bf, bhwc, bt, etc.).
      A DataTensor also holds metadata about the provided data.]],
      {arg='data', type='torch.Tensor', 
       help='A torch.Tensor with 2 dimensions or more.', req=true},
      {arg='axes', type='table', 
       help=[[A table defining the order and nature of each dimension
            of a tensor. Two common examples would be the archtypical 
            MLP input : {'b', 'f'}, or a common image representation : 
            {'b', 'h', 'w', 'c'}. 
            Possible axis symbols are :
            1. Standard Axes:
              'b' : Batch/Example
              'f' : Feature
              't' : Class
            2. Image Axes
              'c' : Color/Channel
              'h' : Height
              'w' : Width
              'd' : Dept
            The provided axes should be the most expanded version of the
            storage. For example, while an image can be represented 
            as a vector, in which case it takes the form of {'b','f'},
            its expanded axes format could be {'b', 'h', 'w', 'c'} 
            ]], default={'b','f'}},
      {arg='sizes', type='table | torch.LongTensor', 
       help=[[A table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions in axes. This should be supplied 
            if the dimensions of the data is different from the number
            of elements in the axes table, in which case it will be used
            to : data:reshape(sizes). Default is data:size().
            ]]}
   )   
   -- Keeps track of the most expanded size (the one with more dims) of the
   -- data. An example would be {'b','h','w','c'} being the expanded_size
   -- of an set of images currently stored as {'b', 'f'}
   self.expanded_size = torch.LongTensor(self._data:size())
   local b = _.indexOf(self.axes, 'b')
   if b == 0 then
      self.axes = _.concat({'b'},self.axes)
      local b = 1
      print("DataTensor Warning: no 'b' axis provided, assuming axes=" .. 
         table.tostring(self.axes))
   end
      
   self.expanded_axes = self.axes
   
   if sizes == nil then
      sizes = torch.LongTensor(self._data:size())
   else 
      if type(sizes) == 'number' then
         sizes = {sizes}
      end
      if type(sizes) == 'table' then
         if (#sizes + 1) ==  #self.axes then
            -- The user only specified the size of the non-b dimensions
            if b == 1 then
               sizes = {self._data:size(1), unpack(sizes)}
            elseif b == #self.axes then
               sizes = {unpack(sizes), self._data:size(-1)}
            else
               error([['b' is not in first or last axis, and provided 
                     sizes specifies one dim less than axes, such that  
                     the mapping of sizes to axes cannot be determined. 
                     Please specify full size, or other axes.]])
            end
            print("DataTensor Warning: sizes specifies one dim less than axes. " ..
                  "Assuming sizes omits 'b' axis. Sizes =" .. 
                  table.tostring(sizes))
         end
      end
      -- convert sizes to LongTensor
      sizes = torch.LongTensor(sizes)
      assert(sizes:prod(1)[1] == self._data:nElement(),
            "Error: sizes specify more elements than available in " .. 
            "data: sizes:prod(1)[1]=" .. sizes:prod(1)[1] .. 
            ", while data:nElement()=" .. self._data:nElement())
      assert(sizes:size(1) == #(self.axes),
             "Error: sizes should specify as many dims as axes:" ..
             tostring(sizes) .. " " .. table.tostring(self.axes))
      if self._data:dim() ~= #(self.axes) then
         print("DataTensor Warning: data:size() is different than sizes. " ..
               "Assuming data is appropriately contiguous. " ..
               "Resizing data to sizes")
         self._data:resize(sizes:storage())
      end
   end
   -- Keep track of the most expanded size
   self:storeExpandedSize(sizes)
   assert(self._data:dim() == #(self.axes), 
         "Error: data should have as many dims as specified in axes" )
end

--Returns the axis of the batch/example index ('b') 
function DataTensor:b()
   local b = _.indexOf(self:storedAxes(), 'b')
   assert(b ~= 0, "Error : no batch axis")
   return b
end

-- Returns number of samples
function DataTensor:nSample()
   assert(#self:storedAxes() == self:storedSize():nElement(),
      "Error : unequal number of axis for size and axes")
   return self:storedSize()[self:b()]
end


function DataTensor:expandedSize()
   return torch.LongTensor(self.expanded_size)
end

function DataTensor:expandedAxes()
   return self.expanded_axes
end

-- Keeps track of the most expanded size (the one with more dims) of the
-- data. An example would be {'b','h','w','c'} being the expanded_size
-- of an set of images currently stored as {'b', 'f'}
function DataTensor:storeExpandedSize(new_size)
   new_size = torch.LongTensor(new_size)
   if new_size:size(1) >= self:expandedSize():size(1) then
      --new_sizes is a more expanded version, store it as expanded_size
      self.expanded_size = new_size
   end
end

function DataTensor:storeExpandedAxes(new_axes)
   if #new_axes >= #(self:expandedAxes()) then
      --new_axes is a more expanded version, store it as expanded_axes
      --furthermore, if new_axes is same length as old expanded_axes,
      --assume some axes have been transposed, such that new_axes is the
      --new expanded representation
      self.expanded_axes = new_axes
   end
end

function DataTensor:storedSize()
   return torch.LongTensor(self._data:size())
end

function DataTensor:storedAxes()
   return self.axes
end

-- Stores a new representation of the data, where new_axes specifies
-- the format of this new data.
function DataTensor:store(new_data, new_axes)
   assert(new_data:nElement() == self._data:nElement(),
          "new_data should have same number of elements as self._data")
   self:storeExpandedAxes(new_axes)
   self:storeExpandedSize(new_data:size())
   self.axes = new_axes
   self._data = new_data
end

function DataTensor:feature(...)
   local axes = {'b','f'}
   local sizes = self:expandedSize()
   assert(sizes:size(1) > 1, "Error: cannot guess size of features")
   local args, inplace, contiguous = xlua.unpack(
      {... or {}},
      'DataTensor:feature',
      [[Returns a 2D-tensor of examples by features : {'b', 'f'}]],
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data a contiguous view of axes 
       {'b', 'f'} for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help=[[When true, makes sure the returned data is contiguous.
            Only considered when inplace is false, since inplace 
            makes it contiguous anyway.]], 
       default=false}
   )
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if self._data:dim() == 2 and table.eq(self.axes, axes) then
      return self._data
   end
   local b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim the first dim
      data = data:transpose(1, b)
      --make contiguous for a later resize (may result in new storage)
      data = data:contiguous()
   end
   if data:dim() > 2 then
      --convert non-b axes to f :
      --reduce tensor down to 2 dimensions: first dim stays the same, 
      --remainder are flattened
      --Note.: convert to LongTensor in order to sub...
      data = data:reshape(
         data:size(1), 
         torch.LongTensor(data:size()):sub(2,data:dim()):prod(1)[1]
      )
   end
   if contiguous or inplace then
      data = data:contiguous()
   end
   if inplace then
      self:store(data, axes)
   else
      self:storeExpandedSize(data:size())
      self:storeExpandedAxes(axes)
   end
   return data
end

function DataTensor:array(...)
   local axes = {'b'} 
   local sizes = self:expandedSize()
   local args, inplace, contiguous = xlua.unpack(
      {... or {}},
      'DataTensor:class',
      'Returns a 1D-tensor of examples.',
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data is a contiguous view of axes 
       {'b'} for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   --use feature:
   local data = self:feature{inplace=inplace,contiguous=contiguous}
   --Takes the first class of each example
   data = data:select(2, 1)
   return data
end

--Returns default view of data
function DataTensor:data()
   return self._data
end

function DataTensor:setData(data)
   self._data = data
end

--Decorator/Adapter for torch.Tensor
--Returns a batch of data. 
--Note that the batch uses different storage (because of :index())
function DataTensor:index(indices)
   return self._data:index(self:b(), indices)
end

function DataTensor.transpose(axis, new_dim, axes, size, data)
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
