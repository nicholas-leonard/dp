async = require 'async'
------------------------------------------------------------------------
--[[ Command ]]--
-- Command design pattern
-- Serializable
-- An object that can be sent over the network
------------------------------------------------------------------------
local Command = torch.class("dp.Command")

function Command.__init(proxy_id)
   self._proxy_id = proxy_id
end

function Command:proxyId()
   return self._proxy_id
end

function Command:execute(session)
   error"Not Implemented"
end

------------------------------------------------------------------------
--[[ Forward ]]--
------------------------------------------------------------------------
local Forward, parent = torch.class("dp.Forward", "dp.Command")

function Forward:__init(model_id, input_state, carry_state, batch_state)
   self._model_id = model_id
   self._input_state = input_state
   self._carry_state = carry_state
   self._batch_state = batch_state
end

function Forward:execute(session)
   local session_map, shared_map = session:objectMaps()
   local model = getSharedWithLocalMemento(self._model_id)
   local output_state, carry_state = model:forward(
      self._input_state, self._carry_state, self._batch_state
   )
   return output_state, carry_state
end

------------------------------------------------------------------------
--[[ ObjectMap ]]--
-- A remote object is mapped to an id in an object map so that it can be retrieved.
-- A command is executed with this global object map as an argument.
-- A coroutine executes as command function with access to session and global object maps.
-- Each process listens and reacts to messages by unserializing and executing them on the object map.
------------------------------------------------------------------------
local ObjectMap = torch.class("dp.ObjectMap")

function ObjectMap:__init(config)
   self._map = {}
end

function ObjectMap:get(object_id)
   assert(object_id.isObjectID, "Expecting dp.ObjectID")
   return self._map[object_id]
end



------------------------------------------------------------------------
--[[ Station ]]--
-- Manages the transmission of messages accross the network.
------------------------------------------------------------------------
local Station = torch.class("dp.Station")

function Station:__init(config)
end

function Station:send(command, session_id)
   local destination = self:locate(command:id())
   async.tcp.client(
end


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

function Session:__init(config)
   local args, id, station, local_map, shared_map
      = xlua.unpack(
         {config or {}},
         'Session', nil,
         {arg='id', type='number'},
         {arg='station', type='dp.Station'},
         {arg='local_map', type='dp.ObjectMap',
          help='Session-local object map'},
         {arg='shared_map', type='dp.ObjectMap'}
   )
   self._id = id
   self._station = station
   self._local_map = local_map
   self._shared_map = shared_map
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

------------------------------------------------------------------------
--[[ ModelProxy ]]--
------------------------------------------------------------------------
local ModelProxy, parent = torch.class("dp.ModelProxy", "dp.Model")

function ModelProxy:forward(input_state, carry_state, batch_state)
   -- build Forward command
   local cmd = dp.Forward(self:id(), input_state, carry_state, batch_state)
   return session:remoteCall(cmd)
end

------------------------------------------------------------------------
--[[ RemoteModel ]]--
------------------------------------------------------------------------
local RemoteModel, parent = torch.class("dp.RemoteModel", "dp.Model")

function RemoteModel:__init(config)
   local args, station
      = xlua.unpack(
         {config or {}},
         'RemoteModel', nil,
         {arg='station', type='dp.Station'},
         {arg=
   )
   self._station = station
end

function RemoteModel:forward(input_state, carry_state, session_state)
   
end

function RemoteModel:setMemento(memento)
   self._state = memento:getState()
end

