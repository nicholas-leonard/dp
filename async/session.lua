------------------------------------------------------------------------
--[[ Session ]]--
-- Each session has its own states and mementos which are distributed through the system.
-- Different sessions may share the same objects, but never at the same time.
-- Sessions should avoid interfering with each other's progress.
-- Each process-session pair can be represented as a coroutine.
-- Sessions may spawn new sessions?
-- A session is a singleton globally available through the coroutine.
-- It points to the process Station which allows it to communicate commands across the network
------------------------------------------------------------------------
local Session = torch.class("dp.Session")
Session.isSession = true

function Session:__init(config)
   local args, id, local_map, shared_map = xlua.unpack(
         {config or {}},
         'Session', nil,
         {arg='id', type='dp.ObjectID'},
         {arg='local_map', type='dp.ObjectMap',
          help='Session-local object map'},
         {arg='shared_map', type='dp.ObjectMap'}
   )
   self._id = id
   self._local_map = local_map or dp.ObjectMap
   self._shared_map = shared_map or dp.ObjectMap
end

function Session:setup(station)
   self._station = station
end

function Session:getSharedWithLocalMemento(object_id)
   local object = self._shared_map:get(object_id)
   local memento = self._local_map:get(object_id)
   object:setMemento(memento)
   return object
end

function Session:remoteCall(command)
   self._station:send(command, session:id())
end

function Session:execute(command)
   command:execute(self)
end
