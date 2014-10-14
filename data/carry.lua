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

function Carry:sub(...)
   local obj_store = table.merge({}, self._obj_store)
   local view_store = {}
   for k,view in pairs(self._view_store) do
      view_store[k] = view:sub(...)
   end
   return dp.Carry(obj_store, view_store)
end

function Carry:index(...)
   local obj_store = table.merge({}, self._obj_store)
   local view_store = {}
   for k,view in pairs(self._view_store) do
      view_store[k] = view:index(...)
   end
   return dp.Carry(obj_store, view_store)
end

function Carry:clone()
   local obj_store = table.merge({}, self._obj_store)
   local view_store = table.copy(self._view_store)
   return dp.Carry(obj_store, view_store)
end
