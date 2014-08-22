async = require 'async'
require 'torch'

local mode = 'ascii'
local separator = '<347 SPLIT 879>'

-- simulates a station : receives and forwards to/from another station
local station = async.tcp.listen({host='localhost', port=8080}, function(client)
   async.fiber(function()
      client.sync()
      --listens to incomming commands from the network
      print('station: new connection:',client)
      local data = client.readsplit(separator)
      print('station: received #data :', #data)
      local tensor = torch.deserialize(data, mode)
      print('station: received tensor :', torch.typename(tensor))
      -- coroutine is yielded by _send() when waiting for async client.
      -- returns after final resume()
      local co = async.fiber.context().co
      ------- SEND --------
      -- execute command within a session (yields after send)
      local client2 = async.tcp.connect({host='localhost', port=8483}, function(client)
         local data = torch.serialize(tensor, mode)
         client.onsplitdata(separator, function(data)
            local reply = torch.deserialize(data)
            print('station: send reply to yield')
            coroutine.resume(co, reply)
            client.close()
         end)
         client.write(data)
         client.write(separator)
      end)
      -- this should be resumed by tcp client when reply is received
      local reply = coroutine.yield()
      print('station: after yield')
      -------- REPLY ----------
      -- send back reply to client
      local data = torch.serialize(reply, mode)
      client.write(data)
      client.write(separator)
      print('station: after write')
   end)
end)

local server = async.tcp.listen({host='localhost', port=8483}, function(client)
   async.fiber(function()
      client.sync()
      print('server: new connection:', client)
      local cache = {}
      local data = client.readsplit(separator)
      print('server: received data :', #data)
      local tensor = torch.deserialize(data, mode)
      print('server: received tensor :', torch.typename(tensor))
      data = torch.serialize(tensor, mode)
      print('server: sending data', #data)
      client.write(data)
      client.write(separator)
      print('server: wrote')
   end)
end)

async.go()
