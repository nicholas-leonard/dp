async = require 'async'
local separator = '<347 SPLIT 879>'

------------------------------------------------------------------------
--[[ Command ]]--
-- Command design pattern
-- Serializable
-- An object that can be sent over the network
------------------------------------------------------------------------
local Command = torch.class("dp.Command")

function Command.__init(subject_id)
   self._subject_id = subject_id
end

function Command:subjectId()
   return self._subject_id
end

function Command:execute(session)
   error"Not Implemented"
end

------------------------------------------------------------------------
--[[ Forward ]]--
------------------------------------------------------------------------
local Forward, parent = torch.class("dp.Forward", "dp.Command")

function Forward:__init(model_id, input_state, carry_state, batch_state)
   parent.init(model_id)
   self._input_state = input_state
   self._carry_state = carry_state
   self._batch_state = batch_state
end

function Forward:execute(session)
   local session_map, shared_map = session:objectMaps()
   local model = getSharedWithLocalMemento(self._subject_id)
   local output_state, carry_state = model:forward(
      self._input_state, self._carry_state, self._batch_state
   )
   return output_state, carry_state
end

------------------------------------------------------------------------
--[[ Locate ]]--
-- Used by stations to query master to locate remote objects
------------------------------------------------------------------------
local Locate, parent = torch.class("dp.Locate", "dp.Command")

function Locate:__init(master_id, addr)
   parent.init(master_id)
   self._addr = addr
end

function Locate:execute(session)
   local session_map, shared_map = session:objectMaps()
   local model = getSharedWithLocalMemento(self._subject_id)
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
--[[ MasterProxy ]]--
------------------------------------------------------------------------
local MasterProxy = torch.class("dp.MasterProxy")

function MasterProxy:__init(id, addr)
   self._id = id
   self._addr = addr
end

function MasterProxy:id()
   return self._id
end

function MasterProxy:addr()
   return self._addr
end

------------------------------------------------------------------------
--[[ Station ]]--
-- Manages the transmission of messages accross the network.
------------------------------------------------------------------------
local Station = torch.class("dp.Station")

function Station:__init(config)
   local args, master_proxy, directory = xlua.unpack(
      {config or {}}
      'Station', nil
      {arg='master_proxy', type='dp.MasterProxy', req=true,
       help='proxy of the Master process object'},
      {arg='directory', type='table'}
   )
   self._directory = directory or {}
   self._master_proxy = master_proxy
end

function Station:locate(object_id)
   local addr = self._directory[object_id]
   if not addr then
      local locate = dp.Locate(self._master_proxy:id(), addr)
      self:_send(locate, self._master_proxy:addr())
   end
end

function Station:send(command, session_id)
   local addr = self:locate(command:subjectId())
   command:setSessionId(session_id)
   self._send(command, addr)
end

function Station:_send(command, dest_addr)
   
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
   self._local_map = local_map or dp.ObjectMap
   self._shared_map = shared_map or dp.ObjectMap
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
local ModelProxy, parent = torch.class("dp.ModelProxy", "dp.BaseModel")

function ModelProxy:forward(input_state, carry_state, batch_state)
   -- build Forward command
   local cmd = dp.Forward(self:id(), input_state, carry_state, batch_state)
   return session:remoteCall(cmd)
end

------------------------------------------------------------------------
--[[ RemoteModel ]]--
------------------------------------------------------------------------
local RemoteModel, parent = torch.class("dp.RemoteModel", "dp.BaseModel")

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


------------------------------------------------------------------------
--[[ Master ]]--
-- Station
-- An object representing the master process of the system
------------------------------------------------------------------------
local Master, parent = torch.class("dp.Master", "dp.Station")

function Master:__init(config)
   local args, id, serial_mode = xlua.unpack(
      {config or {}},
      'Master', 
      'Represents the master process of the system.'
      {arg='id', type='dp.ObjectID', req=true},
      {arg='serial_mode', type='string', default='ascii'}
   )
   self._id = id
   self._mode = mode
   -- contains all sessions
   self._session_map = {}
   -- contains all master proxies (slaves)
   self._slave_map = {}
   
   self._server = async.tcp.listen({host='localhost', port=8483}, function(client)
      print('new connection:',client)
      client.onsplitdata(separator, function(data)
         print('received #data :', #data)
         local command = torch.deserialize(data, self._mode)
         local session = self:session(command:sessionId())
      end)
      client.onend(function()
         print('client ended')
      end)
      client.onclose(function()
         print('closed.')
         collectgarbage()
         print(collectgarbage("count") * 1024)
      end)
   end)
end

function Master:newSession()
   return Session{station=
end

function Master:session(session_id)
   return self._session_map[session_id]
end

------------------------------------------------------------------------
--[[ AsyncExperiment ]]--
-- A distributed asynchronous experiment
------------------------------------------------------------------------
local AsyncExperiment, parent = torch.class("dp.AsyncExperiment", "dp.Experiment")

function AsyncExperiment:setup(config)
   self._master = config.master or assert(false)
   parent.setup(self, config)
end
