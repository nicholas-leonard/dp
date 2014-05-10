------------------------------------------------------------------------
--[[ ImageTensor ]]-- 
-- A DataTensor holding a tensor of images.
------------------------------------------------------------------------
local ImageTensor, parent = torch.class("dp.ImageTensor", "dp.DataTensor")
ImageTensor.isImageTensor = true

--TODO : enforce image representations, or deny non-image ones
function ImageTensor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data, axes, sizes
      = xlua.unpack(
      {config},
      'ImageTensor', 
      'Builds a data.ClassTensor out of torch.Tensor data. ',
      {arg='data', type='torch.Tensor', 
       help='A torch.Tensor with 2 dimensions or more.', req=true},
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
       '[Default={"b", "h", "w", "c"}].'},
      {arg='sizes', type='table | torch.LongTensor', 
       help='A table or torch.LongTensor holding the sizes of the '.. 
       'commensurate dimensions in axes. This should be supplied '..
       'if the dimensions of the data is different from the number '..
       'of elements in the axes table, in which case it will be used '..
       'to : data:reshape(sizes). Default is data:size().'}
   )   
   --TODO error when sizes is not provided for unknown axes.
   axes = axes or {'b','h','w','c'}
   assert(#axes > 2, "Provide at least 3 axes for images")
   parent.__init(self, {data=data, axes=axes, sizes=sizes})
end

-- default view used for image preprocessing
function ImageTensor:image(tensortype, inplace, contiguous)
   return self:imageBHWC(tensortype, inplace, contiguous)
end

function ImageTensor:imageBHWC(tensortype, inplace, contiguous)
   local desired_axes = {'b','h','w','c'}
   local current_axes = self:storedAxes()
   local current_size = self:storedSize()
   local expanded_axes = self:expandedAxes()
   local expanded_size = self:expandedSize()
   assert((expanded_size:size(1) == 4), "No image size provided at construction")
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if not (data:dim() == 4 and table.eq(current_axes, desired_axes)) then
      expanded_axes, expanded_size 
         = parent.transpose('b', 1, expanded_axes, expanded_size)
      --resize to image size
      if data:dim() == 2 and _.contains(current_axes, 'f') then
         --expand {'b','f'}
         current_axes, current_size, data 
            = parent.transpose('b', 1, current_axes, current_size, data)
         data = data:resize(expanded_size:storage())      
      elseif data:dim() == 4 then
         current_axes, current_size, data
            = parent.transpose('b', 1, current_axes, current_size, data)
      end
      --convert between image representations
      expanded_axes, expanded_size, data 
         = parent.transpose('c', 4, expanded_axes, expanded_size, data)
      expanded_axes, expanded_size, data 
         = parent.transpose('h', 2, expanded_axes, expanded_size, data)
      assert(table.eq(expanded_axes, desired_axes),
            "Error: unsupported conversion of axes formats")
   end
   data = self:_format(data, tensortype, contiguous)
   self:_store(data, desired_axes, inplace)
   return data
end

-- View used by SpacialConvolutionCUDA
function ImageTensor:imageCHWB(tensortype, inplace, contiguous)
   -- Dept/Color x Height x Width x Batch (CHWB)
   local desired_axes = {'c','h','w','b'}
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if not (data:dim() == 4 and table.eq(self:storedAxes(), desired_axes)) then
      data = self:imageBHWC(nil, false, false)
      expanded_axes, expanded_size, data
         = parent.transpose('b', 4, self:expandedAxes(), self:expandedSize(), data)
      assert(table.eq(expanded_axes, desired_axes),
            "Error: unsupported conversion of axes formats")
   end
   data = self:_format(data, tensortype, contiguous)
   self:_store(data, desired_axes, inplace)
   return data
end

-- View used by SpacialConvolution
function ImageTensor:imageBCHW(tensortype, inplace, contiguous)
   local desired_axes = {'b','c','h','w'}
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if not (data:dim() == 4 and table.eq(self:storedAxes(), desired_axes)) then
      data = self:imageBHWC(nil, false, false)
      expanded_axes, expanded_size, data
         = parent.transpose('c', 2, self:expandedAxes(), self:expandedSize(), data)
      expanded_axes, expanded_size, data
         = parent.transpose('h', 3, expanded_axes, expanded_size, data)
      if not table.eq(expanded_axes, desired_axes) then
            error("Unsupported conversion of axes formats :"..
               table.tostring(expanded_axes))
      end
   end
   data = self:_format(data, tensortype, contiguous)
   self:_store(data, desired_axes, inplace)
   return data
end

function ImageTensor:conv2D(tensortype, inplace, contiguous)
   if tensortype and tensortype == 'torch.CudaTensor' then
      -- assume its for SpatialConvolutionCUDA
      return self:imageCHWB(tensortype, inplace, contiguous)
   end
   -- works with SpatialConvolution
   return self:imageBCHW(tensortype, inplace, contiguous)
end

function ImageTensor:default(tensortype, inplace, contiguous)
   tensortype = tensortype or torch.typename(self._data)
   return self:image(tensortype, inplace, contiguous)
end


