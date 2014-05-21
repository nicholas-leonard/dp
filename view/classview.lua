------------------------------------------------------------------------
--[[ ClassView ]]--
-- A DataTensor holding a tensor of classes like training targets. 
-- Can also be used to host text where each word is represented as an 
-- integer.
------------------------------------------------------------------------
local ClassView, parent = torch.class("dp.ClassView", "dp.View")
ClassView.isClassView = true

function ClassView:setClasses(classes)
   self._classes = classes
end

function ClassView:classes()
   return self._classes
end

-- multi-class
function ClassView:bt()
   local view, dim = self._view, self._dim
   local b_pos = self:findAxis('b', view)
   -- was b
   if dim == 1 then
      return nn.Reshape(1)
   end
   -- was bt
   local modula
   if b_pos ~= 1 then
      modula = nn.Transpose({1, b_pos})
   end
   if view ~= 'bt' or view ~= 'tb' then
      error("cannot convert view '"..view.."' to bt")
   end
   return modula or nn.Identity()
end

-- single-class
function ClassView:b()
   local view, dim = self._view, self._dim
   local b_pos = self:findAxis('b', view)
   -- was bt
   if dim == 2 and view == 'bt' then
      -- select first set of classes
      return nn.Select(2, 1)
   end
   return nn.Identity()
end

-- one-hot / many-hot
-- assumes that no backward will be called
function ClassView:bf()
   error"Not Implemented" --TODO encapsulate this in a module
   assert(self._classes, "onehot requires self._classes to be set")
   local tensor = self:forwardGet('bt', self._data)
   local nClasses = table.length(self._classes)
   local data = torch.Tensor(tensor:size(1), nClasses):zero()
   for i=1,t:size(1) do
      data[{i,t[i]}] = 1
   end
   data = tensortype and data:type(tensortype) or data
   return data, self._classes
end

-- returns a batch of examples indexed by indices
function ClassView:index(v, indices, config)
   config = config or {}
   config = table.merge(config, {classes=self:classes()})
   return parent.index(self, v, indices, config)
end

--Returns a sub-datatensor narrowed on the batch dimension
function ClassView:sub(start, stop)
   local data = self:multiclass()
   local sizes=self:expandedSize():clone()
   sizes[self:b()] = stop-start+1
   local clone = torch.protoClone(self, {
      data=data:narrow(self:b(), start, stop-start+1),
      axes=table.copy(self:expandedAxes()),
      sizes=sizes, classes=self:classes()
   })
   assert(clone.isClassView, "Clone failed")
   return clone
end

