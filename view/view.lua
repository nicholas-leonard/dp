-----------------------------------------------------------------------
--[[ View ]]-- 
-- Allows for efficiently communicating tensors between Models.
------------------------------------------------------------------------
local View, parent = torch.class("dp.View")
View.isView = true

function View:__init()
   -- caches
   self._modules = {}
   self._tensors = {}
   self._warn = false
end

-- View is a string. Acceptable views include:
-- b, bf, bhwc, chwb, bchw, bsf, bfs, etc.
function View:forward(view, inputORtype)
   assert(torch.type(view) == 'string', "Expecting string at arg 1")
   if torch.type(inputORtype) == 'string' then
      return self:forwardGet(view, inputORtype)
   end
   return self:forwardPut(view, inputORtype)
end
   
-- It is assumed that any input Tensor to forward is represented as
-- the most expanded size of the orignal data. For example, an image
-- batch would be forwarded with its 4 dimensions, and never with 
-- collapses dimensions (2D). 
function View:forwardGet(view, tensor_type)
   -- retrieve a viewTable
   local viewTable = self._tensors[view]
   if not viewTable then
      -- no viewTable: get tensor from module
      return self:tensorFromModule(view, tensor_type)
   else
      local tensor = viewTable[tensor_type]
      if not tensor then
         return self:tensorFromModule(view, tensor_type)
      end
      return tensor
   end
end

function View:tensorFromModule(view, tensor_type)
   local moduleTable = self._modules[view]
   if not moduleTable then
      -- no moduleTable: build a module
      local modula = self[view](self)
      local copy = nn.Copy(torch.typename(self._data), tensor_type)
      self._modules[view] = {modula, {[tensor_type] = copy}}
      local tensor = modula:forward(self._data)
      local viewTable = {[tensor_type] = tensor}
      tensor = copy:forward(tensor)
      viewTable[tensor_type] = tensor
      self._tensors[view] = viewTable
      return tensor
   end
   local modula, copyTable = unpack(moduleTable)
   local tensor = modula:forward(self._data)
   local viewTable = {[tensor_type] = tensor}
   local copy = copyTable[tensor_type]
   if not copy then
      -- no copy : build copy module
      copy = nn.Copy(torch.typename(self._data), tensor_type)
      copyTable[tensor_type] = copy
   end
   tensor = copy:forward(tensor)
   viewTable[tensor_type] = tensor
   self._tensors[view] = viewTable
   return tensor
end

function View:forwardPut(view, input)
   -- store input for later use
   self._dim = #self._view
   if self._data:dim() ~= self._dim then
      error("view has more axes than input has dims", 3)
   end
   -- TODO: optimize this (use tensor cache if not contiguous)
   --self._data = input:contiguous()
   self._view = view
   self._tensors[view] = input
end

function View:findAxis(axis_char, view)
   view = view or self._view
   local axis_pos = view:find(axis_char)
   if not axis_pos then
      error("Provided view '"..view.."' has no axis '"..axis_char.."'", 2)
   end
   return axis_pos
end

function View:sampleSize(b_pos, view, data)
   b_pos = b_pos or self:findAxis('b', view)
   data = data or self._data
   local size = 1
   for i=1,data:dim() do
      if i ~= b_pos then
         size = size * self._data:size(i)
      end
   end
   return size
end

-- batch feature
function View:bf()
   local view, data, dim = self._view, self._data, self._dim
   local b_pos = self:findAxis('b', view)
   -- was b
   if dim == 1 then
      if self._warn then
         print("View:feature Warning: provided data has one "..
               "dim. Assuming dim is 'b' axis with feature size of 1")
      end
      return nn.Reshape(1)
   end
   -- was b...
   local modula
   if b_pos ~= 1 then
      modula = nn.Transpose({1, b_pos})
   end
   if dim > 2 then
      local transpose = modula
      local reshape = nn.Reshape(self:sampleSize(b_pos))
      if transpose then
         modula = nn.Sequential()
         modula:add(transpose)
         modula:add(reshape)
      else
         modula = reshape
      end
   end
   return modula or nn.Identity()
end


function View:backward(view, gradInputORtype)
   assert(torch.type(view) == 'string', "Expecting string at arg 1")
   if torch.type(input) == 'string' then
      return self:forwardGet(view, inputORtype)
   end
   return self:forwardPut(view, inputORtype)
end

--Returns the axis of the batch/example index ('b') 
function View:b()
   local b = _.indexOf(self:storedAxes(), 'b')
   assert(b ~= 0, "Error : no batch axis")
   
   return b
end

-- Returns number of samples
function View:nSample()
   assert(#self:storedAxes() == self:storedSize():nElement(),
      "Error : unequal number of axis for size and axes")
   return self:storedSize()[self:b()]
end

--Returns current view of data
function View:data()
   return self._data
end


function View:pairs()
   return pairs{self}
end

-- When dt is provided, reuse its data (a torch.Tensor).
-- config is a table of key-values overwriting the clones constructor's
function View:index(dt, indices, config)
   assert(self._data:type() ~= 'torch.CudaTensor', 
      "View:index doesn't work with torch.CudaTensors.")
   config = config or {}
   local sizes = self:expandedSize():clone()
   local data
   if indices and dt then
      if torch.type(dt) ~= torch.type(self) then
         error("Expecting "..torch.type(self).." at arg 1 "..
               "got "..torch.type(dt).." instead")
      end
      data = dt:default()
      -- dont use datatensor if cuda (index not in cutorch yet)
      if data:type() ~= 'torch.CudaTensor' then
         torch.Tensor.index(data, self:default(), self:b(), indices)
      else
         data = self:default():index(self:b(), indices)
      end
      sizes[self:b()] = indices:size(1)
      dt:__init(table.merge(config, {
         data=data, sizes=sizes, axes=table.copy(self:expandedAxes())
      }))
      --TODO : copy to existing cuda memory + use index cache
      return dt
   end
   indices = indices or dt
   data = self:default():index(self:b(), indices)
   sizes[self:b()] = indices:size(1)
   return torch.protoClone(self, table.merge(config, {
      data=data, sizes=sizes, axes=table.copy(self:expandedAxes())
   }))
end

--Returns a sub-datatensor narrowed on the batch dimension
function View:sub(start, stop)
   local sizes = self:expandedSize():clone()
   sizes[self:b()] = stop-start+1
   return torch.protoClone(self, {
      data=self:feature():narrow(self:b(), start, stop-start+1),
      axes=table.copy(self:expandedAxes()), sizes=sizes
   })
end

-- returns a clone sharing the same data
function View:shallowClone(config)
   config = config or {}
   local clone = torch.protoClone(self, table.merge(config, {
      data=self._data, axes=table.copy(self:expandedAxes()), 
      sizes=self:expandedSize():clone()
   }))
   return clone
end

function View:type(type)
   error"Not Implemented"
   self._data = self._data:type(type)
end
