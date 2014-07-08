------------------------------------------------------------------------
--[[ Station ]]--
-- Not Serializable
-- Manages the transmission of messages accross the network.
------------------------------------------------------------------------
local Station = torch.class("dp.Station")
Station.isStation = true

function Station:__init(config)
   local args, id, master_proxy, host, port, directory, serial_mode
      = xlua.unpack(
      {config or {}}
      'Station', nil
      {arg='id', type='dp.ObjectID', req=true},
      {arg='master_proxy', type='dp.MasterProxy', req=true,
       help='proxy of the Master process object'},
      {arg='host', type='string', req=true},
      {arg='port', type='number', req=true},
      {arg='directory', type='table'},
      {arg='serial_mode', type='string', default='ascii'}
   )
   self._id = id
   self._directory = directory or {}
   self._master_proxy = master_proxy
   self._host = host
   self._port = port
   self._session_map = {}
   self._serial_mode = serial_mode
   self:startServer()
end

-- How can we handle a JoinTable on inputs from multiple clients?
-- Option 1. Inputs would have to be accumulated before being able to 
-- send back a reply to each client. 
-- Option 2. ParallelProxies sends and receives the replises before 
-- forwarding them to a JoinTable.
-- Since option 2 requires the least amount of work and is the simplest
-- to use from a user perspective, we chose it. Option 1 would require 
-- a dramatic overhaul of our model containers where each model can 
-- chain commands to its own successors and predecessors.
function Station:startServer()
   require 'async'
   -- listens to incoming commands from the network
   self._server = async.tcp.listen({host=self._host, port=self._port}, function(client)
      -- encapsulated in a fiber (a glorified coroutine)
      async.fiber(function()
         client.sync()
         local data = client.readsplit(separator)
         local command = torch.deserialize(data, self._serial_mode)
         -- coroutine is yielded by _send() when waiting for async client.
         -- returns after final resume()
         local session = self:session(command:sessionId())
         -- execute command within a session :
         --  - yields after send (resumed by reply)
         --  - yields in AsyncJoin (resumed by reply)
         local reply = session:execute(command)
         -- send back reply to client
         local data = torch.serialize(reply, self._serial_mode)
         client.write(data)
         client.write(separator)
      end)
   end)
end

function Station:locate(object_id)
   local addr = self._directory[object_id]
   if not addr then
      local locate = dp.Locate(self._master_proxy:id(), addr)
      self:_send(locate, self._master_proxy:addr())
   end
end

function Station:newSession()
end

function Station:session(session_id)
   local session = self._session_map[session_id] or dp.Session
end

function Station:send(command, session_id)
   local addr = self:locate(command:subjectId())
   command:setSessionId(session_id)
   self._send(command, addr)
end

function Station:_send(command, dest_addr)
   local co = async.fiber.context().co
   local client = async.tcp.connect(dest_addr, function(client)
      local data = torch.serialize(command, self._serial_mode)
      client.onsplitdata(separator, function(data)
         local reply = torch.deserialize(data)
         -- send reply to yield (see below)
         coroutine.resume(co, reply)
         client.close()
      end)
      client.write(data)
      client.write(separator)
   end)
   -- this should be resumed by tcp client when reply is received
   local reply = coroutine.yield()
   return reply
end

-- used to send multiple commands
function Station:_sendMany(commands, dest_addrs)
   local co = async.fiber.context().co
   local nCommands = table.length(commands)
   local reply_counter = 0
   local clients = {}
   local replies = {}
   for k, command in pairs(commands) do
      local client = async.tcp.connect(dest_addrs[k], function(client)
         local data = torch.serialize(command, self._serial_mode)
         client.onsplitdata(separator, function(data)
            local reply = torch.deserialize(data)
            replies[k] = reply
            reply_counter = reply_counter + 1
            if reply_counter == nCommands then
               -- send replies to yield (see below)
               coroutine.resume(co, replies)
            end
            client.close()
         end)
         client.write(data)
         client.write(separator)
      end)
      clients[k] = command
   end
   -- this should be resumed by tcp client when reply is received
   local replies = coroutine.yield()
   return replies
end

