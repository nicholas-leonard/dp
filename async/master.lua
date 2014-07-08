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
-- Not Serializable
-- An object representing the master process of the system
-- Should run in its own dedicated process for maximum concurrency
------------------------------------------------------------------------
local Master, parent = torch.class("dp.Master")
Master.isMaster = true

function Master:__init(config)
   local args, station_map = xlua.unpack(
      {config or {}},
      'Master', 
      'Represents the master process of the system.'
      {arg='station_map', type='table'}
   )
   -- contains all master proxies (slaves)
   self._station_map = {}
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
