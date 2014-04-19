require 'torch'
require 'os'
async = require 'async'

local separator = '<347 SPLIT 879>'
local sync = false
local mode = 'binary'

local recv = 0
local sent = 0

-- tcp server
local server = async.tcp.listen({host='localhost', port=8080}, function(client)
   print('new connection:',client)
   client.onsplitdata(separator, function(data)
      print('received #data :', #data)
      local tensor = torch.deserialize(data, mode)
      recv = recv + 1
      print('received tensor :', tensor:size(), recv)
      print('sending data', #data)
      local inner_client = async.tcp.connect('tcp://localhost:8484/', function(client2)
         print('new connection:',client2)
         client2.onsplitdata(separator, function(data)
            print('received:', #data)
            local tensor = torch.deserialize(data, mode)
            --print('received tensor :', tensor:size())
            recv = recv+1
         end)
         client2.onend(function()
            print('client ended')
         end)
         client2.onclose(function()
            print('closed.')
         end)
         --client.write('test')
         client2.write(torch.serialize(tensor, mode))
         client2.write(separator)
         print('wrote')
         sent=sent+1
      end)
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

async.go()
