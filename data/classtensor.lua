------------------------------------------------------------------------
--[[ ClassTensor ]]--
-- A DataTensor holding a tensor of classes.
------------------------------------------------------------------------
local ClassTensor, parent = torch.class("dp.ClassTensor", "dp.DataTensor")
ClassTensor.isClassTensor = true

--TODO validate range of classes
function ClassTensor:__init(...)
   local args, data, axes, sizes, classes
      = xlua.unpack(
      {... or {}},
      'ClassTensor', 
      'Builds a data.ClassTensor out of torch.Tensor data.',
      {arg='data', type='torch.Tensor', 
       help='A torch.Tensor with 1 dimensions or more.', req=true},
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
       '[Default={"b"} for #sizes==1, or {"b","t"} for #sizes==2].'},
      {arg='sizes', type='table | torch.LongTensor', 
       help='A table or torch.LongTensor holding the sizes of the '.. 
       'commensurate dimensions in axes. This should be supplied '..
       'if the dimensions of the data is different from the number '..
       'of elements in the axes table, in which case it will be used '..
       'to : data:reshape(sizes). Default is data:size().'},
      {arg='classes', type='table', help='A list of class IDs.'} 
   )   
   if not axes then
      local lsizes = sizes or data:size()
      if #lsizes == 1 then 
         axes = {'b'}
      elseif #lsizes == 2 then
         axes = {'b', 't'}
      end
   end
   self._classes = classes
   parent.__init(self, {data=data, axes=axes, sizes=sizes})
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
      'Returns a 2D-tensor of examples by classes: {"b", "t"}',
      {arg='inplace', type='boolean', default=true, 
       help='When true, makes self._data a contiguous view of axes '..
       '{"b", "t"} for future use.'},
      {arg='contiguous', type='boolean', default=false,
       help='When true makes sure the returned tensor is contiguous.'}
   )
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if data:dim() == 2 and table.eq(self._axes, axes) then
      return self._data
   end
   assert(table.eq(self._axes, {'b'}) or table.eq(self._axes, {'t', 'b'}),
          "Error: DataTensor doesn't support conversion to {'b', 't'}")
   local b = self:b()
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
      'Returns a 1D-tensor of example classes: {"b"}',
      {arg='inplace', type='boolean', default=true,
       help='When true, makes self._data is a contiguous view of '..
       'axes {"b"} for future use.'},
      {arg='contiguous', type='boolean', default=false,
       help='When true makes sure the returned tensor is contiguous.'}
   )
   --use multiclass:
   local data, classes = self:multiclass{
      inplace=inplace, contiguous=contiguous
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

function ClassTensor:feature(config)
   -- when request as features (usually as inputs), use many-hot view
   return self:manyhot()
end

--Returns a sub-datatensor narrowed on the batch dimension
function ClassTensor:sub(start, stop)
   return torch.factory(torch.typename(self)){
      data=self._data:narrow(self:b(), start, stop-start+1),
      axes=table.copy(self:expandedAxes()),
      sizes=self:expandedSize():clone(),
      classes=self:classes()
   }
end

-- return a clone without data 
function ClassTensor:emptyClone()
   return torch.factory(torch.typename(self)){
      data=torch.emptyClone(self._data),
      axes=table.copy(self:expandedAxes()),
      sizes=self:expandedSize():clone(),
      classes=self:classes()
   }
end

-- copy data into existing memory allocated for data
function ClassTensor:copy(classtensor)
   parent.copy(self, classtensor)
   self._classes = table.copy(classtensor:classes())
end

