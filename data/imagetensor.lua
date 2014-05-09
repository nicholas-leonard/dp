------------------------------------------------------------------------
--[[ ImageTensor ]]-- 
-- A DataTensor holding a tensor of images.
------------------------------------------------------------------------
local ImageTensor, parent = torch.class("dp.ImageTensor", "dp.DataTensor")
ImageTensor.isImageTensor = true

--TODO : enforce image representions, or deny non-image ones
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

function ImageTensor:image(tensortype, inplace, contiguous)
   if tensortype and tensortype == 'torch.CudaTensor' then
      -- assume its for CUDA convolutions
      return self:imageCHWB(tensortype, inplace, contiguous)
   end
   return self:imageBHWC(tensortype, inplace, contiguous)
end

function ImageTensor:imageBHWC(tensortype, inplace, contiguous)
   -- When true, makes stored data a contiguous view for future use :
   inplace = inplace or true
   -- When true makes sure the returned tensor contiguous. 
   -- Only considered when inplace is false, since inplace
   -- implicitly makes the returned tensor contiguous :
   contiguous = contiguous or false
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
   data = tensortype and data:type(tensortype) or data
   if contiguous or inplace then
      data = data:contiguous()
   end
   if contiguous or inplace then
      data = data:contiguous()
      if inplace then
         self:store(data, desired_axes)
      end
   end
   return data, desired_axes
end

-- View used by SpacialConvolutionCUDA
function ImageTensor:imageCHWB(tensortype, inplace, contiguous)
   --Depth x Height x Width x Batch (CHWB)
   -- When true, makes stored data a contiguous view for future use :
   inplace = inplace or true
   -- When true makes sure the returned tensor contiguous. 
   -- Only considered when inplace is false, since inplace
   -- implicitly makes the returned tensor contiguous :
   contiguous = contiguous or false
   local desired_axes = {'c','h','w','b'}
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if not (data:dim() == 4 and table.eq(self:storedAxes(), desired_axes)) then
      data = self:imageBHWC(nil, false, false)
      expanded_axes, expanded_size, data
         = parent.transpose('b', 4, self:expandedAxes(), self:expandedSize(), data)
   end
   data = tensortype and data:type(tensortype) or data
   if contiguous or inplace then
      data = data:contiguous()
   end
   if contiguous or inplace then
      data = data:contiguous()
      if inplace then
         self:store(data, desired_axes)
      end
   end
   return data, desired_axes
end

function ImageTensor:default(tensortype, inplace, contiguous)
   -- self:image() depends on tensortype. Used for cloning
   tensortype = tensortype or torch.typename(self._data)
   return self:image(tensortype, inplace, contiguous)
end

function ImageTensor:imageClone(data)
   assert(data:dim() == 4, "data is not an image")
   local sizes = self:expandedSize():clone()
   if not (sizes[1] == data:size(1) and sizes[2] == data:size(2) and 
         sizes[3] == data:size(3) and sizes[4] == data:size(4)) then
      --data has different size than self, try another view
      self:image(torch.type(data), false, false)
      sizes = self:expandedSize():clone()
      assert(sizes[1] == data:size(1) and sizes[2] == data:size(2)
         and sizes[3] == data:size(3) and sizes[4] == data:size(4),
         "Image size doesnt match any known views")
   end
   local clone = torch.protoClone(self, {
      data=data, axes=table.copy(self:expandedAxes()), sizes=sizes
   })
   if not clone.isImageTensor then
      error("Clone failed. data:"..torch.type(data).." clone:"..torch.type(clone))
   end
   return clone
end


