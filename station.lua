------------------------------------------------------------------------
--[[ Station ]]--
-- Not Serializable
-- Manages the transmission of messages accross the network.
------------------------------------------------------------------------
local Station = torch.class("dp.Station")

function Station:__init(config)
   local args, master_proxy, host, port, directory = xlua.unpack(
      {config or {}}
      'Station', nil
      {arg='master_proxy', type='dp.MasterProxy', req=true,
       help='proxy of the Master process object'},
      {arg='host', type='string', req=true},
      {arg='port', type='number', req=true},
      {arg='directory', type='table'}
   )
   self._directory = directory or {}
   self._master_proxy = master_proxy
   self._host = host
   self._port = port
   self._session_map = {}
   self:startServer()
end

   
function Station:startServer()
   require 'async'
   self._server = async.tcp.listen({host=self._host, port=self._port}, function(client)
      async.fiber(function()
         client.sync()
         --listens to incomming commands from the network
         print('new connection:',client)
         local function handleReply(reply)
            if coroutine.status(coroutine.running()) == 'dead' then
               
            end
         end
         local data = client.readsplit(separator)
         print('received #data :', #data)
         local command = torch.deserialize(data, mode)
         print('received command :', torch.typename(command))
         -- coroutine is yielded by _send() when waiting for async client.
         -- returns after final resume()
         local session = self:session(command:sessionId())
         -- execute command within a session (yields after send)
         local reply = session:execute(command)
         -- send back reply to client
         local data = torch.serialize(reply, self._mode)
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
   return self._session_map[session_id]
end

function Station:send(command, session_id)
   local addr = self:locate(command:subjectId())
   command:setSessionId(session_id)
   self._send(command, addr)
end

function Station:_send(command, dest_addr)
   local co = async.fiber.context().co
   local client = async.tcp.connect(dest_addr, function(client)
      local data = torch.serialize(command, self._mode)
      client.onreadsplit(separator, function(data)
         local reply = torch.deserialize(data)
         --send reply to yield (see below)
         coroutine.resume(co, reply)
         client.close()
      end)
      client.write(data)
      client.write(separator)
      
   end)
   -- this should be resumed by tcp client when reply is received
   -- yields to 
   local reply = coroutine.yield()
end

