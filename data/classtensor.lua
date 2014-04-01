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


function ClassTensor:multiclass(...)
   local axes = {'b', 't'} 
   local sizes = self:expandedSize()
   local args, inplace, contiguous = xlua.unpack(
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
   local data = torch.view(self._data)
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
      data = data:transpose(1, b)
      --make contiguous (new storage) for a later resize
      data = data:contiguous()
   end
   if data:dim() == 1 then
      if sizes:size(1) == 1 then
         print"DataTensor Warning: Assuming one class per example."
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

function ClassTensor:class(...)
   local axes = {'b'} 
   local sizes = self:expandedSize()
   local args, inplace, contiguous = xlua.unpack(
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
   local data, classes = self:multiclass{
      inplace=inplace,contiguous=contiguous
   }
   --Takes the first class of each example
   data = data:select(2, 1)
   return data, classes
end

function ClassTensor:onehot(...)
   error("Not Implemented")
end

function ClassTensor:manyhot(...)
   error("Not Implemented")
end
