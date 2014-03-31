------------------------------------------------------------------------
--[[ ObjectID ]]--
-- An identifier than can be used to save files, objects, etc.
-- Provides a unique name.
------------------------------------------------------------------------
local ObjectID = torch.class("dp.ObjectID")
ObjectID.isObjectID = true

function ObjectID:__init(name, parent)
   self._parent = parent
   self._name = name
end

function ObjectID:toList()
   local obj_list = {}
   if self._parent then 
      obj_list = self._parent:toList()
   end
   table.insert(obj_list, self._name)
   return obj_list
end

function ObjectID:toString(seperator)
   seperator = seperator or ':'
   local obj_string = ''
   if self._parent then
      obj_string = self._parent:toString() .. seperator 
   end
   return obj_string .. self._name
end

function ObjectID:toPath()
   return self:toString('/')
end   

function ObjectID:name()
   return self._name
end

function ObjectID:create(name)
   return dp.ObjectID(name, self)
end

function ObjectID:parent()
   return self._parent
end
