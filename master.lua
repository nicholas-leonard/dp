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
   local args, id, local_map, shared_map
      = xlua.unpack(
         {config or {}},
         'Session', nil,
         {arg='id', type='number'},
         {arg='local_map', type='dp.ObjectMap',
          help='Session-local object map'},
         {arg='shared_map', type='dp.ObjectMap'}
   )
   self._id = id
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

function RemoteModel:setLocalMemento(memento)
   self._state = memento:getState()
end

function RemoteModel:set


------------------------------------------------------------------------
--[[ Master ]]--
-- Station
-- Not Serializable
-- An object representing the master process of the system
-- Should run in its own dedicated process for maximum concurrency
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
   -- server listens to incomming connections
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
--[[ AsyncPropagator ]]--
-- Serializable
-- A distributed asynchronous propagator
------------------------------------------------------------------------
local AsyncPropagator, parent = torch.class("dp.AsyncPropagator", "dp.Propagator")

function AsyncPropagator:setup(config)
   self._master_proxy = config.master_proxy or assert(false)
   parent.setup(self, config)
   for i in range(
   async.fiber(function() 
      self:propagateEpoch(batch, report, session)
   end)
end

-- TODO : refactor some stuff so as to not repeat code
function AsyncPropagator:propagateEpoch(dataset, report)
   self:resetLoss()
   if self._feedback then
      self._feedback:reset()
   end
   
   -- local vars
   local start_time = sys.clock()
   local last_batch
   
   if self._stats then
      print('==> epoch # '..(report.epoch + 1)..' for '..self:name())
   end
   
   for batch in self._sampler:sampleEpoch(dataset) do
   
      async.fiber(function() 
         self:propagateBatch(batch, report, session)
      end)
      
      if self._progress then
         -- disp progress
         xlua.progress(batch:batchIter(), batch:epochSize())
      end
      last_batch = batch
   end
   if self._progress and not self._stats then
      print"\n"
   end
   
   -- time taken
   self._epoch_duration = sys.clock() - start_time
   self._batch_duration = self._epoch_duration / last_batch:epochSize()
   self._example_speed = last_batch:epochSize() / self._epoch_duration
   self._num_batches = last_batch:epochSize() / last_batch:batchSize()
   self._batch_speed = (self._num_batches / self._epoch_duration)
   if self._stats then
      print("\n==> epoch size = "..last_batch:epochSize()..' examples')
      print("==> batch duration = "..(self._batch_duration*1000)..' ms')
      print("==> epoch duration = " ..self._epoch_duration..' s')
      print("==> example speed = "..self._example_speed..' examples/s')
      print("==> batch speed = "..self._batch_speed..' batches/s')
   end
end      
