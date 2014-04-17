require 'torch'
async = require 'async'

local separator = '<347 SPLIT 879>'

-- tcp server
local server = async.tcp.listen({host='localhost', port=8483}, function(client)
   async.fiber(function()
      client.sync()
      print('new connection:', client)
      while true do
         local cache = {}
         local data = client.readsplit(separator)
         print('received data :', #data)
         local tensor = torch.deserialize(data, 'ascii')
         print('received tensor :', tensor:size(), tensor:max())
         print('sending data', #data)
         client.write(data)
         client.write(separator)
      end
   end)
end)

async.go()
