-- WORK IN PROGRESS : (only works for forward('bf')
-----------------------------------------------------------------------
--[[ ListView ]]-- 
-- Composite (design pattern) of component Views
-- Encapsulates a list (a table with keys : 1,2,3,...) of Views.
------------------------------------------------------------------------
local ListView, parent = torch.class("dp.ListView", "dp.View")
ListView.isListView = true

function ListView:__init(components)
   parent.assertInstances(components)
   self._components = components
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

function ListView:backwardGet(view, tensor_type)
   local gradInputs
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
   error"Not Implemented"
   if indices then
      assert(v.isListView, "Expecting ListView as first argument")
      return torch.protoClone(self, 
         _.map(self._components, 
            function(key, component) 
               return component:index(v, indices)
            end
         )
      )
   else
      indices = v
   end
   return torch.protoClone(self, 
      _.map(self._components, 
         function(key, component) 
            return component:index(v, indices)
         end
      )
   )
end

function ListView:sub(start, stop)
   error"Not Implemented"
   return torch.protoClone(self,
      _.map(self._components, 
         function(key, component) 
            return component:sub(start, stop)
         end
      )
   )
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
