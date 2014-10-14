------------------------------------------------------------------------
--[[ Carry ]]--
-- An object store that is carried (passed) around the network 
-- during a propagation. 
-- Useful for passing information between decoupled 
-- objects like DataSources and Feedbacks. 
-- Dangerous if not used correctly (could be the meatball 
-- in the spaghetti code...).
------------------------------------------------------------------------
local Carry = torch.class("dp.Carry")

function Carry:__init(obj_store, view_store)
   self._obj_store = obj_store or {}
   self._view_store = view_store or {}
end

function Carry:putObj(key, val)
   self._obj_store[key] = val
end

function Carry:getObj(key)
   return self._obj_store[key]
end

function Carry:putView(key, val)
   self._view_store[key] = val
end

function Carry:getView(key, val)
   return self._view_store[key]
end

function Carry:sub(carry, start, stop, inplace)
   local obj_store, view_store
   if carry and stop then
      if torch.type(carry) ~= torch.type(self) then
         error("Expecting "..torch.type(self).." at arg 1 "..
               "got "..torch.type(carry).." instead")
      end
      carry:flushObjStore()
   else
      if carry then
         inplace = stop
         stop = start
         start = carry
      end
      carry = torch.protoClone(self)
   end
   
   obj_store, view_store = carry._obj_store, carry._view_store
   -- copy object table (deep copy of tables, except for torch.classes)
   table.merge(obj_store, self._obj_store)
   -- use dp.View:sub() when possible to reuse memory
   for k,view in pairs(self._view_store) do
      view_store[k] = view:sub(view_store[k], start, stop, inplace)
   end
   return dp.Carry(obj_store, view_store)
end

function Carry:index(carry, indices)
   local obj_store, view_store
   if indices and carry then
      if torch.type(carry) ~= torch.type(self) then
         error("Expecting "..torch.type(self).." at arg 1 "..
               "got "..torch.type(carry).." instead")
      end
      carry:flushObjStore()
   else
      indices = indices or carry
      carry = dp.Carry()
   end
   
   obj_store, view_store = carry._obj_store, carry._view_store
   -- copy object table (deep copy of tables, except for torch.classes)
   table.merge(obj_store, self._obj_store)
   -- use dp.View:index() when possible to reuse memory
   for k,view in pairs(self._view_store) do
      view_store[k] = view:index(view_store[k], indices)
   end
   return dp.Carry(obj_store, view_store)
end

function Carry:clone()
   local obj_store = table.merge({}, self._obj_store)
   local view_store = table.copy(self._view_store)
   return dp.Carry(obj_store, view_store)
end

function Carry:flushObjStore()
   self._obj_store = {}
end
