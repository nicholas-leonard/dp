-----------------------------------------------------------------------
--[[ ListView ]]-- 
-- Composite (design pattern) of component Views
-- Encapsulates a list (a table with keys : 1,2,3,...) of Views.
------------------------------------------------------------------------
local ListView, parent = torch.class("dp.ListView", "dp.View")
ListView.isListView = true

function ListView:__init(components)
   self._components = components or {}
   parent.assertInstances(self._components)
   self._modules = {}
   parent.__init(self)
end

function ListView:forwardPut(views, inputs)
   for i=1,#self._components do
      if torch.type(views) == 'table' then
         self._components[i]:forwardPut(views[i], inputs[i])
      else
         self._components[i]:forwardPut(views, inputs[i])
      end
   end
end

function ListView:forwardGet(views, tensor_types)
   self._got = true
   local tensors = {}
   for i=1,#self._components do
      local view = torch.type(views) == 'table' and views[i] or views
      local tensor_type = torch.type(tensor_types) == 'table' and tensor_types[i] or tensor_types
      tensors[i] = self._components[i]:forwardGet(view, tensor_type)
   end
   return tensors
end

function ListView:backwardPut(views, gradOutputs)
   for i=1,#self._components do
      local view = torch.type(views) == 'table' and views[i] or views
      self._components[i]:backwardPut(view, gradOutputs[i])
   end
end

function ListView:backwardGet(views, tensor_type)
   local gradInputs = {}
   for i=1,#self._components do
      local view = torch.type(views) == 'table' and views[i] or views
      local tensor_type = torch.type(tensor_types) == 'table' and tensor_types[i] or tensor_types
      gradInputs[i] = self._components[i]:backwardGet(view, tensor_type)
   end
   return gradInputs
end

-- Returns number of samples
function ListView:nSample()
   for k,component in self:pairs() do
      return component:nSample()
   end
end

function ListView:index(v, indices)
   if indices then
      if not torch.isTypeOf(v, self) then
         error("Expecting "..torch.type(self).." at arg 1 "..
               "got "..torch.type(v).." instead")
      end
      if v:size() ~= self:size() then
         error("Expecting "..torch.type(self).." ar arg 1 " ..
               "having same number of components as self")
      end
      for i, component in self:pairs() do
         component:index(v:components()[i], indices)
      end
   else
      indices = v
      v = self.new(
         _.map(self._components, 
            function(key, component) 
               return component:index(indices)
            end
         )
      )
   end
   return v
end

function ListView:sub(v, start, stop, inplace)
   if v and stop then
      if not torch.isTypeOf(v, self) then
         error("Expecting "..torch.type(self).." at arg 1 "..
               "got "..torch.type(v).." instead")
      end
      if v:size() ~= self:size() then
         error("Expecting "..torch.type(self).." ar arg 1 " ..
               "having same number of components as self")
      end
      for i, component in self:pairs() do
         component:sub(v:components()[i], start, stop, inplace)
      end
   else
      if v then
         inplace = stop
         stop = start
         start = v
      end
      v = self.new(
         _.map(self._components, 
            function(key, component) 
               return component:sub(start, stop, inplace)
            end
         )
      )
   end
   return v
end

function ListView:size()
   return #self._components
end

-- return iterator over components
function ListView:pairs()
   return ipairs(self._components)
end

function ListView:components()
   return self._components
end

function ListView:flush()
   for i,component in self:pairs() do
      component:flush()
   end
end

function ListView:input()
   local input = {}
   for i,component in ipairs(self._components) do
      input[i] = component:input()
   end
   return input
end
