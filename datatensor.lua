require 'torch'
require 'image'

require 'utils'


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
   local args
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
            to : data:resize(sizes). Default is data:size().
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
      print("Warning: no 'b' axis provided, assuming axes=" .. 
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
            print("Warning: sizes specifies one dim less than axes. " ..
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
         print("Warning: data:size() is different than sizes. " ..
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
   assert(#self:storedAxes() == self:storedSize():nElements(),
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
   return self._data:size()
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
   local inplace, contiguous = xlua.unpack(
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
   local data = torch.Tensor(self._data)
   if self._data:dim() == 2 and table.eq(self.axes, axes) then
      return self._data
   end
   local b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim the first dim
      data:transpose(1, b)
      --make contiguous for a later resize (may result in new storage)
      data = data:contiguous()
   end
   if data:dim() > 2 then
      --convert non-b axes to f :
      --reduce tensor down to 2 dimensions: first dim stays the same, 
      --remainder are flattened
      --Note.: convert to LongTensor in order to sub...
      data:resize(data:size(1), torch.LongTensor(data:size()):sub(2,data:dim()):prod(1)[1])
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

function DataTensor:image(...)
   local axes = self:expandedAxes()
   local sizes = self:expandedSize()
   if #axes ~= 4 then
      print"Warning: no image axis provided at construction, assuming {'b','h','w','c'}"
      axes = {'b','h','w','c'}
   end
   assert((sizes:size(1) == 4), "Error: no image size provided at construction")
   local inplace, contiguous = xlua.unpack(
      {... or {}},
      'DataTensor:images',
      'Returns a 4D-tensor of axes format : ' .. table.tostring(axes),
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data a contiguous view of axes 
       ]] .. table.tostring(axes) .. [[ for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   --creates a new view of the same storage
   local data = torch.Tensor(self._data)
   if data:dim() == 4 and table.eq(self.axes, axes) then
      return self._data
   end
   local b = _.indexOf(self.axes, 'b')
   local new_b = _.indexOf(axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= new_b then
      --make (transpose) the batch dim first
      data:transpose(new_b, b)
      sizes:transpose(new_b, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   if data:dim() == 2 and _.contains(self.axes, 'f') then
      assert(sizes:size(1) == 4, "Error: sizes doesn't have enough dimensions")
      --convert {'b','f'} to {'b','h','w','c'}
      data:resize(sizes:storage())      
   else
      error("unsupported conversion of axes formats")
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
   return data, axes
end

function DataTensor:imageCUDA(...)
   local axes = {'c','h','w','b'}
   --TODO
   error("Error Not Implemented")
end

function DataTensor:multiclass(...)
   local axes = {'b', 't'} 
   local sizes = self:expandedSize()
   local inplace, contiguous, sizes = xlua.unpack(
      {... or {}},
      'DataTensor:multiclass',
      [[Returns a 2D-tensor of examples by classes: {'b', 't'}]],
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data a contiguous view of axes 
       {'b', 'f'} for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   --creates a new view of the same storage
   local data = torch.Tensor(self._data)
   if data:dim() == 2 and table.eq(self.axes, axes) then
      return self._data
   end
   assert(table.eq(self.axes, {'b'}) or table.eq(self.axes, {'t', 'b'}),
          "Error: DataTensor doesn't support conversion to {'b', 't'}")
   local b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim first
      data:transpose(1, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   if data:dim() == 1 then
      if sizes:size(1) == 1 then
         print"Warning: Assuming one class per example."
         sizes = torch.LongTensor({self._data:size(1), 1})
      end
      --convert {'b'} to {'b','t'}
      data:resize(sizes:storage())
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
   return data, self._classes
end

function DataTensor:class(...)
   local axes = {'b'} 
   local sizes = self:expandedSize()
   local inplace, contiguous, sizes = xlua.unpack(
      {... or {}},
      'DataTensor:class',
      [[Returns a 1D-tensor of example classes: {'b'}]],
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data is a contiguous view of axes 
       {'b'} for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   --use multiclass:
   data, classes = self:multiclass(args)
   --Takes the first class of each example
   data = data:select(2, 1)
   return data, classes
end

function DataTensor:onehot(...)
   error("Not Implemented")
end

function DataTensor:manyhot(...)
   error("Not Implemented")
end

function DataTensor:array(...)
   error("Not Implemented")
   local axes = {'b'} 
   local inplace, contiguous, sizes = xlua.unpack(
      {... or {}},
      'DataTensor:b',
      'Returns a 1D-tensor of examples.',
      {arg='inplace', type='boolean', 
       help=[[When true, makes self._data a contiguous view of axes 
       {'b'} for future use.]], 
       default=true},
      {arg='contiguous', type='boolean', 
       help='When true makes sure the returned tensor is contiguous.', 
       default=false}
   )
   --creates a new view of the same storage
   local data = torch.Tensor(self._data)
   if data:dim() == 1 and table.eq(self.axes, axes) then
      return self._data
   end
   local b = _.indexOf(self.axes, 'b')
   if b == 0 then
      error("No batch ('b') dimension")
   elseif b ~= 1 then
      --make (transpose) the batch dim first
      data:transpose(1, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   --convert to {'b'}
   data:resize(torch.LongTensor(data:size()):prod(1)[1])
   if contiguous or inplace then
      data = data:contiguous()
   end
   if inplace then
      self:store(data, axes)
   end
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


------------------------------------------------------------------------
-- ImageTensor : A DataTensor holding a tensor of images.
------------------------------------------------------------------------
local ImageTensor = torch.class("dp.ImageTensor", "dp.DataTensor")

--TODO : enforce image representions, or deny non-image ones
function ImageTensor:__init(...)
   local args, data, axes, sizes
      = xlua.unpack(
      {... or {}},
      'ImageTensor', 
      [[Constuctor. Builds a data.ClassTensor out of torch.Tensor data.
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
            ]], default={'b','h','w','c'}},
      {arg='sizes', type='table | torch.LongTensor', 
       help=[[A table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions in axes. This should be supplied 
            if the dimensions of the data is different from the number
            of elements in the axes table, in which case it will be used
            to : data:resize(sizes). Default is data:size().
            ]]}
   )   
   --TODO error when sizes is not provided for unknown axes.
   DataTensor.__init(self, {data=data, axes=axes, sizes=sizes})
end


------------------------------------------------------------------------
-- ClassTensor : A DataTensor holding a tensor of classes.
------------------------------------------------------------------------
local ClassTensor = torch.class("dp.ClassTensor", "dp.DataTensor")
--TODO validate range of classes
function ClassTensor:__init(...)
   local args, data, axes, sizes, classes
      = xlua.unpack(
      {... or {}},
      'ClassTensor', 
      [[Constuctor. Builds a data.ClassTensor out of torch.Tensor data.
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
            ]], default={'b'}},
      {arg='sizes', type='table | torch.LongTensor', 
       help=[[A table or torch.LongTensor identifying the sizes of the 
            commensurate dimensions in axes. This should be supplied 
            if the dimensions of the data is different from the number
            of elements in the axes table, in which case it will be used
            to : data:resize(sizes). Default is data:size().
            ]]},
      {arg='classes', type='table',
       help=[[A table containing class ids.]]} 
   )   
   self._classes = classes
   DataTensor.__init(self, {data=data, axes=axes, sizes=sizes})
end

function ClassTensor:default()
   return self:class()
end

function ClassTensor:classes()
   return self._classes
end
