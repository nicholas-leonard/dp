------------------------------------------------------------------------
--[[ ClassTensor ]]--
-- A DataTensor holding a tensor of classes.
------------------------------------------------------------------------
local ClassTensor, parent = torch.class("dp.ClassTensor", "dp.DataTensor")
ClassTensor.isClassTensor = true

--TODO validate range of classes
function ClassTensor:__init(config)
   assert(type(config) == 'table', "Constructor requires key-value arguments")
   local args, data, axes, sizes, classes
      = xlua.unpack(
      {config},
      'ClassTensor', 
      'Builds a dp.ClassTensor out of torch.Tensor data.',
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

function ClassTensor:multiclass(inplace, contiguous)
   -- When true, makes stored data a contiguous view for future use :
   inplace = inplace or true
   -- When true makes sure the returned tensor contiguous. 
   -- Only considered when inplace is false, since inplace
   -- implicitly makes the returned tensor contiguous :
   contiguous = contiguous or false
   local axes = {'b', 't'} 
   local sizes = self:expandedSize()
   --creates a new view of the same storage
   local data = torch.view(self._data)
   if data:dim() == 2 and table.eq(self._axes, axes) then
      return self._data
   end
   assert(table.eq(self._axes, {'b'}) or table.eq(self._axes, {'t','b'}),
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
         if self._warn then
            print"DataTensor Warning: Assuming one class per example."
         end
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

function ClassTensor:class(inplace, contiguous)
   -- When true, makes stored data a contiguous view for future use :
   inplace = inplace or true
   -- When true makes sure the returned tensor contiguous. 
   -- Only considered when inplace is false, since inplace
   -- implicitly makes the returned tensor contiguous :
   contiguous = contiguous or false
   local axes = {'b'} 
   local sizes = self:expandedSize()
   --use multiclass:
   local data, classes = self:multiclass(inplace, contiguous)
   --Takes the first class of each example
   data = data:select(2, 1)
   return data, classes
end

function ClassTensor:onehot()
   -- doesn't convert data inplace
   local t = self:class()
   assert(self._classes, "onehot requires self._classes to be set")
   local nClasses = table.length(self._classes)
   local data = torch.IntTensor(t:size(1), nClasses):zero()
   for i=1,t:size(1) do
      data[{i,t[i]}] = 1
   end
   return data, self._classes
end

function ClassTensor:manyhot()
   -- doesn't convert data inplace
   local t = self:multiclass()
   assert(self._classes, "onehot requires self._classes to be set")
   local nClasses = table.length(self._classes)
   local data = torch.IntTensor(t:size(1), nClasses):zero()
   for i=1,t:size(1) do
      local t_x = t:select(1,i)
      local data_x = data:select(1,i)
      for j=1,t:size(2) do
         data_x[t_x[j]] = 1
      end
   end
   return data, self._classes
end

function ClassTensor:feature(inplace, contiguous)
   -- when request as features (could be for inputs), use many-hot view
   return self:manyhot(inplace, contiguous)
end

-- returns a batch of examples indexed by indices
function ClassTensor:index(dt, indices)
   local data
   if indices then
      data = dt and dt:multiclass()
      torch.Tensor.index(data, self._data, self:b(), indices)
   end
   data = data or self:multiclass():index(self:b(), indices)
   local sizes = self:expandedSize():clone()
   indices = indices or dt
   if dt then
      assert(dt.isClassTensor, "Expecting BaseTensor as first argument")
   end
   sizes[self:b()] = indices:size(1)
   return torch.protoClone(self, {
      data=data, sizes=sizes, axes=table.copy(self:expandedAxes()),
      classes=self:classes()
   })
end

--Returns a sub-datatensor narrowed on the batch dimension
function ClassTensor:sub(start, stop)
   local data = self:multiclass()
   local sizes=self:expandedSize():clone()
   sizes[self:b()] = stop-start+1
   local clone = torch.protoClone(self, {
      data=data:narrow(self:b(), start, stop-start+1),
      axes=table.copy(self:expandedAxes()),
      sizes=sizes, classes=self:classes()
   })
   assert(clone.isClassTensor, "Clone failed")
   return clone
end

-- return a clone with self's metadata initialized with some data 
function ClassTensor:featureClone(data)
   local sizes = self:expandedSize():clone()
   assert(sizes[self:b()] == data:size(self:b()))
   return torch.protoClone(self, {
      data=data, 
      axes=table.copy(self:expandedAxes()),
      sizes=self:expandedSize():clone(),
      classes=self:classes()
   })
end

-- copy data into existing memory allocated for data
function ClassTensor:copy(classtensor)
   parent.copy(self, classtensor)
   self._classes = classtensor:classes()
end

