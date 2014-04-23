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
   parent.__init(self, {data=data, axes=axes, sizes=sizes})
end

function ImageTensor:image(inplace, contiguous)
   return self:imageBHWC(inplace, contiguous)
end

function ImageTensor:imageBHWC(inplace, contiguous)
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
   if #expanded_axes ~= 4 then
      print"DataTensor Warning: no image axis provided at construction, assuming {'b','h','w','c'}"
      expanded_axes = desired_axes
   end
   assert((expanded_size:size(1) == 4), "Error: no image size provided at construction")
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if data:dim() == 4 and table.eq(current_axes, desired_axes) then
      return self._data, desired_axes
   end
   expanded_axes, expanded_size 
      = parent.transpose('b', 1, expanded_axes, expanded_size)
   --resize to image size
   if data:dim() == 2 and _.contains(current_axes, 'f') then
      --expand {'b','f'}
      assert(expanded_size:size(1) == 4, 
             "Error: sizes doesn't have enough dimensions")
      current_axes, current_size, data 
         = parent.transpose('b', 1, current_axes, current_size, data)
      data:contiguous():resize(expanded_size:storage())      
   end
   --convert between image representations
   expanded_axes, expanded_size, data 
      = parent.transpose('c', 4, expanded_axes, expanded_size, data)
   expanded_axes, expanded_size, data 
      = parent.transpose('h', 2, expanded_axes, expanded_size, data)
   assert(table.eq(expanded_axes, desired_axes),
         "Error: unsupported conversion of axes formats")
   if contiguous or inplace then
      data = data:contiguous()
      if inplace then
         self:store(data, desired_axes)
      end
   end
   return data, desired_axes
end

function ImageTensor:imageCUDA(inplace, contiguous)
   inplace = inplace or true
   contiguous = contiguous or false
   local axes = {'c','h','w','b'}
   --TODO
   error("Error Not Implemented")
end

