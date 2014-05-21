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
end

function ListView:forwardPut(views, inputs)
   local tensors = {}
   for i=1,#self._components do
      self._components[i]:forwardPut(views[i], inputs[i])
   end
   self._tensors = {}
   self._gradOutputs = {}
end

function ListView:forwardGet(view, tensor_type)
   -- TODO this should work with 'table' view
   if view ~= 'bf' then
      error("only works with bf(concatenate tensors on f axis)", 2)
   end
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

function ListView:tensorFromModule(view, tensor_type)
   -- forwardGet from components
   local tensors = {}
   for i=1,#self._components do
      local tensor = self._components[i]:forwardGet(view, tensor_type)
      table.insert(tensors, tensor)
   end
   -- build module or forward through it
   local viewTable = self._tensors[view] or {}
   local moduleTable = self._modules[view] or {}
   local modula = moduleTable[tensor_type]
   if not modula then
      -- no moduleTable: build a module
      modula = self[view](self)
      modula:type(tensor_type)
      moduleTable[tensor_type] = modual
      self._modules[view] = moduleTable
   end
   local tensor = modula:forward(tensors)
   viewTable[tensor_type] = tensor
   self._tensors[view] = viewTable
   return tensor
end

-- This method could be called from multiple output Models
function ListView:backwardPut(view, gradOutput)
   -- store gradOutput in list
   table.insert(self._gradOutputs, {view, gradOutput})
end

-- This method should be called by a maximum of one input Model.
-- In the case of multiple output models having called backwardPut, 
-- the different gradInputs must be accumulated (sum grads).
function ListView:backwardGet(view, tensor_type)
   local view, gradOutput, gradInput
   
   -- optimization : one-to-one backward
   if #self._gradOutputs == 1 then
      view, gradOutput = unpack(self._gradOutputs[1])
      tensor_type = torch.typename(gradOutput)
      local moduleTable = self._modules[view]
      assert(moduleTable, "backward must follow a forward")
      local modula = moduleTable[tensor_type]
      assert(modula, "backward must follow a forward")
      gradOutputs = modula:backward(nil, gradOutput)
      local gradInputs = {}
      for i=1,#self._components do
         self._components[i]:backwardPut(view, gradOutput[i])
         gradInput = self._components[i]:backwardGet(view, tensor_type)
         table.insert(gradInputs, gradInput)
      end
      return gradInputs
   end
   
   -- slower : many-to-one backward
   error"Not Implemented" --TODO
   for i, gradOutputTable in ipairs(self._gradOutputs) do
      view, gradOutput = unpack(gradOutputTable)
      local moduleTable = self._modules[view]
      assert(moduleTable, "backward must follow a forward")
      local modula, copyTable = unpack(moduleTable)
      assert(copyTable, "backward must follow a forward")
      tensor_type = torch.typename(gradOutput)
      gradInput = copyTable[tensor_type]:backward(nil, gradOutput)
      gradInput = modula:backward(nil, gradInput)
      -- accumulate
      if i == 1 then
         self._gradInput:copy(gradInput)
      else
         self._gradInput:add(gradInput)
      end
   end
   return self._gradInput
end

function ListView:bf()   
   return nn.JoinTable(2)
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
