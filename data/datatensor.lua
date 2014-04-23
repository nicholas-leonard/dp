-- TODO:
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

-----------------------------------------------------------------------
--[[ DataTensor ]]-- 
-- Encapsulates a torch.Tensor. Provides access to it using different
-- viewing methods. A view may reshape the tensor inplace and render it
-- contiguous. Views can be used to convert data into new axes formats 
-- using torch.Tensor:resize, :transpose, :contiguous. The 
-- conversions may be done in-place (default), or may be simply  
-- returned using the conversion methods (feature, class, image, etc.). 
-- A DataTensor may also holds metadata about the provided data.
------------------------------------------------------------------------
local DataTensor, parent = torch.class("dp.DataTensor", "dp.BaseTensor")
DataTensor.isDataTensor = true

function DataTensor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data, axes, sizes = xlua.unpack(
      {config},
      'DataTensor', 
      'Builds a dp.DataTensor out of torch.Tensor data.',
      {arg='data', type='torch.Tensor | dp.DataTensor', req=true,
       help='A torch.Tensor with 1 dimensions or more. Or setup from '..
       'an existing dp.DataTensor (in which case, other params are '..
       'ignored.'},
      {arg='axes', type='table',
       help='A table defining the order and nature of each dimension '..
       'of a tensor. Two common examples would be the archtypical '..
       'MLP input : {"b", "f"}, or a common image representation : '..
       '{"b", "h", "w", "c"}. \n'..
       'Possible axis symbols are : \n'..
       '1. Standard Axes: \n'..
       ' "b" : Batch/Example \n'..
       ' "f" : Feature \n'..
       ' "t" : Class \n'..
       '2. Image Axes \n'..
       ' "c" : Color/Channel \n'..
       ' "h" : Height \n'..
       ' "w" : Width \n'..
       ' "d" : Dept \n'..
       'The provided axes should be the most expanded version of the '..
       'storage. For example, while an image can be represented '..
       'as a vector, in which case it takes the form of {"b","f"}, '..
       'its expanded axes format could be {"b", "h", "w", "c"}. '..
       '[Default={"b","f"}]'},
      {arg='sizes', type='table | torch.LongTensor | torch.LongStorage', 
       help='A table or torch.LongTensor holding the sizes of the '.. 
       'commensurate dimensions in axes. This should be supplied '..
       'if the dimensions of the data is different from the number '..
       'of elements in the axes table, in which case it will be used '..
       'to : data:reshape(sizes). Default is data:size().'}
   )   
   if data.isDataTensor then
      self._axes = table.copy(data:storedAxes())
      self._expanded_axes = table.copy(data:expandedAxes())
      self._expanded_size = data:expandedSize():clone()
      data = data:data()
      return
   end
   self._data = data
   self._axes = axes or {'b','f'}
   -- Keeps track of the most expanded size (the one with more dims) of the
   -- data. An example would be {'b','h','w','c'} being the expanded_size
   -- of an set of images currently stored as {'b', 'f'}
   self._expanded_size = torch.LongTensor(self._data:size())
   local b = _.indexOf(self._axes, 'b')
   if b == 0 then
      self._axes = _.concat({'b'},self._axes)
      local b = 1
      print("DataTensor Warning: no 'b' axis provided, assuming axes=" .. 
         table.tostring(self._axes))
   end
      
   self._expanded_axes = self._axes
   
   if sizes == nil then
      sizes = torch.LongTensor(self._data:size())
   else 
      if type(sizes) == 'number' then
         sizes = {sizes}
      end
      if type(sizes) == 'table' then
         if (#sizes + 1) ==  #self._axes then
            -- The user only specified the size of the non-b dimensions
            if b == 1 then
               sizes = {self._data:size(1), unpack(sizes)}
            elseif b == #self._axes then
               sizes = {unpack(sizes), self._data:size(-1)}
            else
               error("'b' is not in first or last axis, and provided "..
                  "sizes specifies one dim less than axes, such that ".. 
                  "the mapping of sizes to axes can't be determined. "..
                  "Please specify full size, or other axes.")
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
      assert(sizes:size(1) == #(self._axes),
             "Error: sizes should specify as many dims as axes:" ..
             tostring(sizes) .. " " .. table.tostring(self._axes))
      if self._data:dim() ~= #(self._axes) then
         print("DataTensor Warning: data:size() is different than sizes. " ..
               "Assuming data is appropriately contiguous. " ..
               "Resizing data to sizes")
         self._data:resize(sizes:storage())
      end
   end
   -- Keep track of the most expanded size
   self:storeExpandedSize(sizes)
   assert(self._data:dim() == #(self._axes), 
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
   return torch.LongTensor(self._expanded_size)
end

function DataTensor:expandedAxes()
   return self._expanded_axes
end

-- Keeps track of the most expanded size (the one with more dims) of the
-- data. An example would be {'b','h','w','c'} being the expanded_size
-- of an set of images currently stored as {'b', 'f'}
function DataTensor:storeExpandedSize(new_size)
   new_size = torch.LongTensor(new_size)
   if new_size:size(1) >= self:expandedSize():size(1) then
      --new_sizes is a more expanded version, store it as expanded_size
      self._expanded_size = new_size
   end
end

function DataTensor:storeExpandedAxes(new_axes)
   if #new_axes >= #(self:expandedAxes()) then
      --new_axes is a more expanded version, store it as expanded_axes
      --furthermore, if new_axes is same length as old expanded_axes,
      --assume some axes have been transposed, such that new_axes is the
      --new expanded representation
      self._expanded_axes = new_axes
   end
end

function DataTensor:storedSize()
   return torch.LongTensor(self._data:size())
end

function DataTensor:storedAxes()
   return self._axes
end

-- Stores a new representation of the data, where new_axes specifies
-- the format of this new data.
function DataTensor:store(new_data, new_axes)
   assert(new_data:nElement() == self._data:nElement(),
          "new_data should have same number of elements as self._data")
   self:storeExpandedAxes(new_axes)
   self:storeExpandedSize(new_data:size())
   self._axes = new_axes
   self._data = new_data
end

function DataTensor:_feature(inplace, contiguous)
   local axes = {'b','f'}
   local sizes = self:expandedSize()
   assert(sizes:size(1) > 1, "Error: cannot guess size of features")
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if self._data:dim() == 2 and table.eq(self._axes, axes) then
      return self._data
   end
   local b = _.indexOf(self._axes, 'b')
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

--Returns current view of data
function DataTensor:data()
   return self._data
end

function DataTensor:setData(data)
   self._data = data
end

function DataTensor:pairs()
   return pairs{self}
end

-- When dt is provided, reuse its data (a torch.Tensor).
function DataTensor:index(dt, indices)
   local sizes = self:expandedSize():clone()
   sizes[self:b()] = indices:size(1)
   local data
   if indices then
      assert(dt.isBaseTensor, "Expecting BaseTensor as first argument")
      data = dt:data()
      torch.Tensor.index(data, self._data, self:b(), indices)
   else
      data = self._data:index(self:b(), indices)
   end
   return torch.protoClone(self, {
      data=data, sizes=sizes, axes=table.copy(self:expandedAxes())
   })
end

--Returns a sub-datatensor narrowed on the batch dimension
function DataTensor:sub(start, stop)
   local sizes = self:expandedSize():clone()
   sizes[self:b()] = stop-start+1
   return torch.protoClone(self, {
      data=self._data:narrow(self:b(), start, stop-start+1),
      axes=table.copy(self:expandedAxes()), sizes=sizes
   })
end

-- return a clone with self's metadata initialized with some data 
function DataTensor:featureClone(data)
   local sizes = self:expandedSize():clone()
   assert(sizes[self:b()] == data:size(1))
   return torch.protoClone(self, {
      data=data, 
      axes=table.copy(self:expandedAxes()),
      sizes=self:expandedSize():clone()
   })
end

-- copy data into existing memory allocated for data
function DataTensor:copy(datatensor)
   self._data:resizeAs(datatensor:data())
   self._data:copy(datatensor:data())
   self._axes = table.copy(datatensor:storedAxes())
   self._expanded_axes = table.copy(datatensor:expandedAxes())
   self._expanded_size = datatensor:expandedSize()
end


