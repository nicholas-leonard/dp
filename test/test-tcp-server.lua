require 'torch'
require 'os'
async = require 'async'

local separator = '<347 SPLIT 879>'
local sync = false
local mode = 'binary'

local recv = 0

-- tcp server
local server

if sync then
   -- build tcp server using fiber + sync
   server = async.tcp.listen({host='localhost', port=8483}, function(client)
      async.fiber(function()
         client.sync()
         print('new connection:', client)
         while true do
            local cache = {}
            local data = client.readsplit(separator)
            print('received data :', #data)
            recv = recv + 1
            local tensor = torch.deserialize(data, mode)
            print('received tensor :', tensor:size())
            data = torch.serialize(tensor, mode)
            print('sending data', #data)
            client.write(data)
            client.write(separator)
            print('wrote')
         end
      end)
   end)
else
   -- build tcp sever using async
   server = async.tcp.listen({host='localhost', port=8483}, function(client)
      print('new connection:',client)
      client.onsplitdata(separator, function(data)
         print('received #data :', #data)
         local tensor = torch.deserialize(data, mode)
         recv = recv + 1
         print('received tensor :', tensor:size(), recv)
         print('sending data', #data)
         --this will fail and exit without error when client kills before write could finish
         client.write(data)
         client.write(separator)
         print('wrote')
      end)
      client.onend(function()
         print('client ended')
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

async.go()
