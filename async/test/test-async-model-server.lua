require 'dp'
async = require 'async'

--Station
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
   self._server = async.tcp.listen({host=self._host, port=self._port}, function(client)
      --listens to incomming commands from the network
      print('new connection:',client)
      client.onsplitdata(separator, function(data)
         print('received #data :', #data)
         local command = torch.deserialize(data, mode)
         print('received command :', torch.typename(command))
         local session = self:session(command:sessionId())
         -- execute command within a session
         local reply = session:execute(command)
         local data = torch.serialize(reply)
         client.write(reply)
      end)
      client.onend(function()
         print('server ended')
      end)
      client.onclose(function()
         print('closed.')
         collectgarbage()
         print(collectgarbage("count") * 1024)
      end)
      client.onerr(function()
         print('error')
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
   local reply
   local client = async.tcp.connect(dest_addr, function(client)
      print('new connection:',client)
      client.onsplitdata(separator, function(data)
         print('received:', #data)
         reply = torch.deserialize(data, mode)
         print('received command :', torch.typename(command))
         client.close()
      end)
      client.onend(function()
         
         print('client ended')
      end)
      client.onclose(function()
         print('closed.')
      end)
      --client.write('test')
      local data = torch.serialize(command, mode)
      print('sending :', #data)
      client.write(data)
      client.write(separator)
   end)
end


async.go()
